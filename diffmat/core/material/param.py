from typing import Union, Tuple, Dict, Optional, Callable

from torch.nn.functional import relu, hardtanh
import torch as th

from diffmat.core.base import BaseParameter
from diffmat.core.function import FunctionGraph
from diffmat.core.types import ParamValue, IntParamValue, Constant, DeviceType
from diffmat.core.util import OL


class ConstantParameter(BaseParameter[ParamValue]):
    """A constant (hence non-optimizable) paramter in a differentiable material graph.
    """
    def __init__(self, name: str, data: ParamValue, **kwargs):
        """Initialize a constant material graph parameter.

        Args:
            name (str): Parameter name.
            data (ParamValue): Parameter value. Although the value can be a PyTorch tensor,
                the parameter will still remain constant throughout optimization.
            kwargs (Dict[str, Any], optional): Keyword arguments to pass into the parent class
                constructor.
        """
        super().__init__(name, data, **kwargs)

    def evaluate(self) -> ParamValue:
        """Obtain the parameter value.

        Returns:
            ParamValue: Parameter value.
        """
        data = self.data
        return data.clone() if isinstance(data, th.Tensor) \
               else data.copy() if isinstance(data, list) else data

    def set_value(self, value: Constant):
        """Set the parameter value from a constant.

        Args:
            value (Constant): Source parameter value.
        """
        if isinstance(self.data, th.Tensor):
            self.data.copy_(self._at(value))
        else:
            self.data = type(self.data)(value)

    def to_device(self, device: DeviceType = 'cpu'):
        """Move the material graph parameter to a specified device (e.g., CPU or GPU). The value
        will be moved if it is a PyTorch tensor.

        Args:
            device (DeviceType, optional): Target device ID. Defaults to 'cpu'.
        """
        self.data = self.data.to(device) if isinstance(self.data, th.Tensor) else self.data
        super().to_device(device)


class IntegerParameter(BaseParameter[IntParamValue]):
    """An integer (only optimizable by gradient-free optimizers) parameter in a differentiable
    material graph.
    """
    IS_OPTIMIZABLE_INTEGER = True

    def __init__(self, name: str, data: IntParamValue,
                 scale: Tuple[IntParamValue, IntParamValue] = [0, 1], **kwargs):
        """Initialize the integer-valued parameter.

        Args:
            name (str): Parameter name.
            data (IntParamValue): Parameter value.
            scale (Tuple[IntParamValue, IntParamValue], optional): Valid range of the parameter
                value ([low, high]). A parameter will not exceed its value range during
                optimization. Defaults to [0, 1].
            kwargs (Dict[str, Any], optional): Keyword arguments to pass into the parent class
                constructor.
        """
        super().__init__(name, data, **kwargs)

        # Record integer range
        low, high = scale
        self.scale = ([int(v) for v in low] if isinstance(low, (tuple, list)) else int(low),
                      [int(v) for v in high] if isinstance(high, (tuple, list)) else int(high))

    def evaluate(self) -> IntParamValue:
        """Obtain the integer parameter value, clamped by the range.

        Returns:
            IntParamValue: Parameter value within its default range.
        """
        data, (low, high) = self.data, self.scale

        # Clamp data value using the parameter range
        if isinstance(data, int):
            val = min(max(data, low), high)
        else:
            low = [low] * len(data) if isinstance(low, int) else low
            high = [high] * len(data) if isinstance(high, int) else high
            val = [min(max(vd, vl), vh) for vd, vl, vh in zip(data, low, high)]

        return val

    def set_value(self, value: IntParamValue):
        """Set the parameter value from a constant integer.

        Args:
            value (IntParamValue): Source parameter value.
        """
        self.data = [int(v) for v in value] if isinstance(value, (tuple, list)) else int(value)


class Parameter(BaseParameter[Optional[th.Tensor]]):
    """An optimizable parameter in a differentiable material graph.

    Static members:
        IS_OPTIMIZABLE (bool): Whether the parameter is considered optimizable. Defaults to True.
    """
    IS_OPTIMIZABLE = True

    def __init__(self, name: str, data: Optional[th.Tensor],
                 scale: Tuple[Union[float, th.Tensor], Union[float, th.Tensor]] = (0.0, 1.0),
                 relu_flag: bool = False, hardtanh_flag: bool = True, quantize: bool = False,
                 **kwargs):
        """Initialize an optimizable material graph parameter.

        Args:
            name (str): Parameter name.
            data (Optional[Tensor]): Parameter value. A None value should only be used when a node
                function that receives this parameter as input explicitly handles None values. For
                instance, the `anchor` parameter in Gradient Map and Curve nodes.
            scale (Tuple[float | Tensor, float | Tensor], optional): Valid range of the parameter
                value ([low, high]). A parameter will not exceed its value range during
                optimization. Defaults to (0.0, 1.0).
            relu_flag (bool, optional): Apply a ReLU operation to clamp the parameter value in
                every optimization iteration. Defaults to False.
            hardtanh_flag (bool, optional): Apply a hardtanh operation to clamp the parameter value
                in every optimization iteration. Defaults to True.
            quantize (bool, optional): Whether the parameter represents the continuous form of an
                originally discrete parameter. In that case, the parameter must be quantized to
                integers after optimization. Defaults to False.
            kwargs (Dict[str, Any], optional): Keyword arguments to pass into the parent class
                constructor.

        Raises:
            ValueError: The data input is not a PyTorch tensor.
        """
        if data is not None and not isinstance(data, th.Tensor):
            raise ValueError('Data input to an optimizable parameter must be a torch tensor')

        super().__init__(name, None, **kwargs)

        self.data = data.to(self.device) if data is not None else data

        low, high = scale
        self.scale = (low.to(self.device) if isinstance(low, th.Tensor) else low,
                      high.to(self.device) if isinstance(high, th.Tensor) else high)

        self.relu_flag = relu_flag
        self.hardtanh_flag = hardtanh_flag
        self.quantize = quantize

    def _unnormalize(self, value: th.Tensor) -> th.Tensor:
        """Unnormalize the stored parameter value to be used as a node function argument.

        Args:
            value (tensor): Normalized parameter value (in [0, 1]).

        Returns:
            Tensor: Unnormalized parameter value.
        """
        # Apply a linear mapping according to the value range
        low, high = self.scale
        if isinstance(low, th.Tensor) or low != 0:
            value_unnorm = value * (high - low) + low
        elif isinstance(high, th.Tensor) or high != 1:
            value_unnorm = value * high
        else:
            value_unnorm = value.clone()
        return value_unnorm

    def evaluate(self) -> Optional[th.Tensor]:
        """Obtain the parameter value and apply a nonliear function.

        Returns:
            Tensor: Parameter value within its user-provided range.
        """
        data = self.data
        if data is None:
            return data

        # Apply ReLU or hardtanh to clamp the parameter value
        with th.no_grad():
            if self.relu_flag:
                data = relu(data, inplace=True)
            elif self.hardtanh_flag:
                data = hardtanh(data, 0.0, 1.0, inplace=True)

        # Unnormalize the stored parameter as output
        return self._unnormalize(data)

    def set_value(self, value: Constant, normalize: bool = False):
        """Set the parameter value from a constant, with the option of normalizing the input.

        Args:
            value (Constant): Source parameter value.
            normalize (bool, optional): Switch for normalizing the input according to the parameter
                range.
        """
        value: th.Tensor = self._t(value)
        if normalize:
            low, high = self.scale
            if isinstance(low, float) and isinstance(high, float):
                value = (value - low) / max(high - low, 1e-8)
            else:
                value = (value - low) / (high - low).clamp_min(1e-8)

        # Update the parameter value without gradient
        with th.no_grad():
            self.data.copy_(value)

    def to_device(self, device: DeviceType = 'cpu'):
        """Move the material graph parameter to a specified device (e.g., CPU or GPU), including
        the parameter value and the scales.

        Args:
            device (DeviceType, optional): Target device ID. Defaults to 'cpu'.
        """
        # Move the parameter data
        if self.data is not None:
            self.data = self.data.to(device)

        # Move the scales
        low, high = self.scale
        self.scale = (low.to(device) if isinstance(low, th.Tensor) else low,
                      high.to(device) if isinstance(high, th.Tensor) else high)

        super().to_device(device)


class GradientMapAnchor(Parameter):
    """The 'anchors' parameter in a Gradient Map node.
    """
    def __init__(self, name: str, data: Optional[th.Tensor], **kwargs):
        """Initialize an optimizable Gradient Map anchor parameter.

        Args:
            name (str): Parameter name.
            data (Optional[Tensor]): Parameter value. Supplying a None value effectively makes the
                Gradient Map node a simple grayscale-to-color converter.
            kwargs (Dict[str, Any], optional): Keyword arguments to pass into the parent class
                constructor.
        """
        super().__init__(name, data, **kwargs)

    def _unnormalize(self, value: th.Tensor) -> th.Tensor:
        """Unnormalize the position coordinates in an anchor array by calculating a normalized
        prefix sum.

        Args:
            value (Tensor): Normalized anchor array.

        Returns:
            Tensor: Anchor array in its original scale.
        """
        # Calculate the prefix sum to restore X positions and normalize them in case the last
        # coordinate exceeds [0, 1]
        unnorm_pos = th.cumsum(value.select(1, 0), 0)
        unnorm_pos = unnorm_pos / unnorm_pos[-1].clamp_min(1.0)

        # Assemble the anchor matrix and perform unscaling
        unnorm_value = value.clone()
        unnorm_value[:, 0] = unnorm_pos
        return unnorm_value


class CurveAnchor(Parameter):
    """The 'anchors' parameter in a Curve node.
    """
    def __init__(self, name: str, data: Optional[th.Tensor], **kwargs):
        """Initialize an optimizable Curve anchor parameter.

        Args:
            name (str): Parameter name.
            data (Optional[Tensor]): Parameter value. The Curve node will behave as an identity
                function if a None value is supplied.
            kwargs (Dict[str, Any], optional): Keyword arguments to pass into the parent class
                constructor.
        """
        super().__init__(name, data, **kwargs)

    def _unnormalize(self, value: th.Tensor) -> th.Tensor:
        """Unnormalize X positions and control point coordinates of an anchor array.

        Args:
            value (Tensor): Normalized anchor array.

        Returns:
            Tensor: Anchor array in its original scale.
        """
        # Calculate the prefix sum to restore X positions and normalize them in case the last
        # coordinate exceeds [0, 1]
        positions = th.cumsum(value.select(1, 0), 0)
        positions = positions / positions[-1].clamp_min(1.0)

        # Restore control point positions
        diff_pre = positions.diff(1, prepend=th.zeros(1, device=self.device))
        diff_app = positions.diff(1, append=th.ones(1, device=self.device))
        left_pos = th.clamp(positions - value.select(1, 2) * diff_pre, 0, 1)
        right_pos = th.clamp(value.select(1, 4) * diff_app + positions, 0, 1)

        # Assemble the anchor matrix
        anchors = value.clone()
        anchors[:, 0] = positions
        anchors[:, 2] = left_pos
        anchors[:, 4] = right_pos

        # Final per-value unnormalization
        return super()._unnormalize(anchors)


class DynamicParameter(BaseParameter[FunctionGraph]):
    """Material graph parameter defined with a value processor (function graph). The value is
    computed by evaluating the corresponding function graph.

    Static members:
        IS_DYNAMIC (bool): Whether the parameter holds a dynamic value as defined using a function
            graph instead of a literal or tensor. Defaults to True.
    """
    # Dynamic parameters are defined by value processors
    IS_DYNAMIC = True

    def __init__(self, name: str, data: FunctionGraph,
                 map_value: Optional[Callable[[ParamValue], ParamValue]] = None, **kwargs):
        """Initialize a constant material graph parameter.

        Args:
            name (str): Parameter name.
            data (FunctionGraph): The value processor (function graph) that encodes the parameter
                value.
            map_value (Optional[(ParamValue) -> ParamValue], optional): A value mapping function to
                succeed function graph execution. Provides addtional conversion into values
                acceptable to node functions (e.g., converts integers to string-valued options).
                Defaults to None (no conversion).
            kwargs (Dict[str, Any], optional): Keyword arguments to pass into the parent class
                constructor.
        """
        super().__init__(name, data, **kwargs)

        # Since dynamic parameters only computes values in Substance designer, an optional mapping
        # function could be given to convert the value to diffmat. Default to identity mapping
        self.map_value: Callable[[ParamValue], ParamValue] = \
            map_value if map_value else lambda _: _

        # Undefined output level
        self.output_level = -1

    def compile(self, var_levels: Dict[str, int]) -> int:
        """Compile the function graph associated with the dynamic parameter. This is a wrapper of
        the same method in the `FunctionGraph` class.

        Args:
            var_levels (Dict[str, int]): The value categories (levels) of named variables visible
                to the function graph.

        Returns:
            int: Value category of the dynamic parameter.
        """
        # Compile the function graph to infer output level
        self.output_level = self.data.compile(var_levels)
        return self.output_level

    def evaluate(self, var: Dict[str, ParamValue]) -> ParamValue:
        """Evaluate the function graph associated with the dynamic parameter. This is a wrapper of
        the same function in `class FunctionGraph`.

        Args:
            var (Dict[str, ParamValue]): Collection of named variables visible to the function
                graph.

        Returns:
            ParamValue: Dynamic parameter value (output of the function graph) mapped to valid
                formats for node function input.
        """
        # Automatically add the result into the source dictionary
        value = self.data.evaluate(var=var)
        return self.map_value(value)

    @property
    def output_level(self) -> int:
        """The value category of a dynamic parameter is not defined until compilation. Raise an
        error if this attribute is referenced before `compile`.

        Returns:
            int: Value category of the dynamic parameter.
        """
        if self._output_level < 0:
            raise RuntimeError("The dynamic parameter must be compiled to determine its output "
                               "level. Please invoke the 'compile' method first.")

        return self._output_level

    @output_level.setter
    def output_level(self, val: int):
        """Setter for the 'output_level' attribute.

        Args:
            val (int): Source output level.
        """
        self._output_level = val

    def set_value(self, _: Constant):
        """Set the parameter value from a constant. This is merely a dummy function to instantiate
        the abstract base class.
        """
        self.logger.warning('Attempt to set the value of a dynamic parameter. Ignored')

    def to_device(self, device: DeviceType = 'cpu'):
        """Move the function graph of the dynamic parameter to a specified device (e.g., CPU or
        GPU).

        Args:
            device (DeviceType, optional): Target device ID. Defaults to 'cpu'.
        """
        # Move the function graph
        self.data.to_device(device)

        super().to_device(device)
