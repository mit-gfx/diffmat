from xml.etree import ElementTree as ET
from typing import Optional, Union, Dict, List, Tuple

import torch as th

from diffmat.core.base import BaseParameter
from diffmat.core.material import (
    ConstantParameter, IntegerParameter, Parameter, GradientMapAnchor, CurveAnchor,
    DynamicParameter
)
from diffmat.translator import types as tp
from diffmat.translator.base import BaseParamTranslator
from diffmat.translator.function_trans import FunctionGraphTranslator
from diffmat.translator.types import Constant, ParamValue
from diffmat.translator.util import get_param_value, to_constant, is_optimizable


class ConstantParamTranslator(BaseParamTranslator):
    """Translator of an XML subtree to a constant (non-optimizable) material graph parameter.

    Static members:
        PARAM_CLASS (Type[Parameter]): Parameter class that the translator instantiates.
    """
    # Parameter object class that this translator produces
    PARAM_CLASS = ConstantParameter

    def __init__(self, root: Optional[ET.Element], name: Optional[str] = None,
                 sbs_name: Optional[str] = None, default: Optional[Constant] = None,
                 sbs_default: Optional[Constant] = None, requires_default: bool = True,
                 **param_kwargs):
        """Initialize the constant parameter translator.

        For detailed definitions of arguments `name`, `sbs_name`, `default`, and `sbs_default`,
        please refer to the constructor of `BaseParamTranslator`.

        Args:
            root (Optional[Element]]): Root node of the XML tree. The parameter will be translated
                to its default value if an XML tree is not given.
            name (Optional[str], optional): Parameter name in Diffmat. Defaults to None.
            sbs_name (Optional[str], optional): Parameter name in Substance Designer.
                Defaults to None.
            default (Optional[Constant], optional): Default parameter value in Diffmat storage.
                Defaults to None.
            sbs_default (Optional[Constant], optional): Default parameter value in Substance
                Designer. Defaults to None.
            requires_default (bool, optional): Whether a default parameter value (via either
                `default` or `sbs_default`) must be provided. Defaults to True.
            param_kwargs (Dict[str, Any], optional): keyword arguments that will be passed directly
                to the parameter object constructor during translation.

        Raises:
            ValueError: A default parameter value is not provided when `requires_default` is True.
        """
        # Raise error if the parameter should not be none and no default value is given
        if requires_default and default is None and sbs_default is None:
            raise ValueError('The default parameter value must be provided')

        super().__init__(root, name=name, sbs_name=sbs_name, default=default,
                         sbs_default=sbs_default, **param_kwargs)

        # Parameters that require default values can not be evaluated to None
        self.requires_default = requires_default

        # Define a simple identity type cast function
        def _t(x: Constant) -> Constant: return x
        self._t = _t

        # Process dynamic parameter
        self.function_trans: Optional[FunctionGraphTranslator] = None
        if self.type < 0:
            self.function_trans = \
                FunctionGraphTranslator(self.root.find('.//dynamicValue'), self.name)

    def _to_literal(self, value_str: str, type: int = 0) -> Constant:
        """Process string-valued Substance parameters into numbers or arrays (nothing special is
        done by default).

        Args:
            value_str (str): Parameter value in string format.
            type (int, optional): Parameter type specifier. See 'type numbers' in
                `diffmat/translator/types.py`. Defaults to 0.

        Returns:
            Constant: Parameter value in numerical format.
        """
        return to_constant(value_str, type)

    def _map(self, value: ParamValue) -> ParamValue:
        """Map a Substance parameter value to the corresponding Diffmat parameter value. The
        default behavior is identity mapping.

        Args:
            value (ParamValue): Parameter value in Substance Designer.

        Returns:
            ParamValue: Parameter value in Diffmat.
        """
        return value

    def _normalize(self, value: ParamValue) -> ParamValue:
        """Normalize a diffmat parameter value to [0, 1]. Constant parameters do not need
        normalization by default.

        Args:
            value (ParamValue): Parameter value in Diffmat.

        Returns:
            ParamValue: Normalized parameter value for Diffmat storage.
        """
        return value

    def _calc_value(self) -> Optional[ParamValue]:
        """Calculate the diffmat parameter value from the XML.

        Returns:
            Optional[ParamValue]: Parameter value for Diffmat storage (None or floating-point
                numbers normalized to [0, 1]).
        """
        _t = self._t                    # Type casting function
        _map = self._map                # Substance to diffmat value mapping function
        _normalize = self._normalize    # Normalization function

        # Obtain parameter value from XML subtree first
        value_str = ''
        if self.root:
            value_str = get_param_value(self.root, check_dynamic=True)

        # The value is converted from strings to literals, casted to the correct type, mapped to
        # diffmat convention, and finally normalized for storage and optimization
        if value_str:
            value = _normalize(_map(_t(self._to_literal(value_str, self.type))))

        # Use the default provided value if no XML is given
        elif self.default is not None:
            value = _t(self.default)

        # Convert the default value in Substance
        elif self.sbs_default is not None:
            value = _normalize(_map(_t(self.sbs_default)))

        # Leave the parameter as None (this should only happen to anchor parameters)
        else:
            value = None

        return value

    def translate(self, **obj_kwargs) -> Union[BaseParameter, DynamicParameter]:
        """Convert the parameter value to a Python object and instantiate the parameter.

        Args:
            obj_kwargs (Dict[str, Any], optional): Keyword arguments that will be passed to
                the instantiated parameter object and, additionally, a function graph translator
                when the parameter is dynamic.

        Raises:
            RuntimeError: Generating a constant parameter that holds a None value.

        Returns:
            BaseParameter | DynamicParameter: Translated parameter object.
        """
        # Handle dynamic parameter translation
        if self.type < 0:
            value = self.function_trans.translate(**obj_kwargs)
            return DynamicParameter(self.name, value, map_value=self._map)

        # Calculate constant parameter value and construct a parameter object
        value: Optional[ParamValue] = self._calc_value()
        if self.requires_default and value is None:
            raise RuntimeError("Parameter value None is not allowed for constant parameters. "
                               "Please check whether a default value is provided.")

        return self.PARAM_CLASS(self.name, value, **self.param_kwargs, **obj_kwargs)


class IntegerParamTranslator(ConstantParamTranslator):
    """Translator of an XML subtree to an integer-valued material graph parameter.

    Static members:
        PARAM_CLASS (Type[Parameter]): Parameter class that the translator instantiates.
    """
    # Parameter object class that this translator produces
    PARAM_CLASS = IntegerParameter

    def __init__(self, root: Optional[ET.Element], scale: Union[int, Tuple[int, int]] = 1,
                 **trans_and_param_kwargs):
        """Initialize the integer parameter translator.

        Args:
            root (Optional[Element]): Root node of the XML tree. The parameter will be translated
                to its default value if an XML tree is not given.
            scale (int | Tuple[int, int], optional): Default parameter value range. If a single
                integer is given, the range is `[0, scale]`. If a tuple of two integers is given,
                the range is `[scale[0], scale[1]]`. Defaults to 1.
            trans_and_param_kwargs (Dict[str, Any], optional): Keyword arguments for the parent
                class constructor and the translated parameter object.
        """
        super().__init__(root, **trans_and_param_kwargs)

        # Set the lower/upper bounds of the parameter
        if isinstance(scale, (list, tuple)):
            self.scale = (int(scale[0]), int(scale[1]))
        else:
            self.scale = (0, int(scale))

        # Pass the parameter range to the constructed parameter object
        self.param_kwargs['scale'] = self.scale

        # Verify that the parameter indeed has integer values
        if self.type not in (tp.DYNAMIC, tp.INT, tp.INT2, tp.INT3, tp.INT4):
            raise ValueError(f'Parameter {self.name} has a non-integer value type: {self.type}')

    def _normalize(self, value: Union[int, List[int]]) -> int:
        """Adjust the parameter range according to the input value.

        Args:
            value (Union[int, List[int]]): Integer parameter value (scalar or vector).

        Returns:
            int: The input value.
        """
        # Detect if the parameter value is out of bound. Substance handles an out-of-bound
        # parameter by automatically adjusting the scale so that the parameter becomes the midpoint
        # of the expanded value range. Note that the scale may vary among elements in a vector
        low, high = self.scale

        if isinstance(value, int):
            new_low = value * 2 - high if value < low else low
            new_high = value * 2 - low if value > high else high
        else:
            new_low = [v * 2 - high if v < low else low for v in value]
            new_high = [v * 2 - low if v > high else high for v in value]

        # Save the new value range
        self.scale = new_low, new_high

        # Update the parameter range of the constructed parameter object
        self.param_kwargs['scale'] = self.scale

        return value


class ParamTranslator(ConstantParamTranslator):
    """Translator of an XML subtree to an optimizable material graph parameter.

    Static members:
        PARAM_CLASS (Type[Parameter]): Parameter class that the translator instantiates.
    """
    # Parameter object class that this translator produces
    PARAM_CLASS = Parameter

    def __init__(self, root: Optional[ET.Element], scale: Union[float, Tuple[float, float]] = 1.0,
                 quantize: bool = False, **trans_and_param_kwargs):
        """Initialize the parameter translator.

        Args:
            root (Optional[Element]): Root node of the XML tree. The parameter will be translated
                to its default value if an XML tree is not given.
            scale (float | Tuple[float, float], optional): Parameter value range during
                optimization (one float: [0, val]; two floats: [val_0, val_1]). Defaults to 1.0.
            quantize (bool, optional): Whether the parameter represents the continuous form of an
                originally discrete parameter. In that case, the parameter must be quantized to
                integers after optimization. Defaults to False.
            trans_and_param_kwargs (Dict[str, Any], optional): Keyword arguments for the parent
                translator class constructor and the translated parameter object.

        Raises:
            ValueError: Content of the XML tree implies that the parameter is not optimizable.
            RuntimeError: Attempt to optimize an integer parameter without setting the `quantize`
                flag.
        """
        super().__init__(root, **trans_and_param_kwargs)

        # Set the lower/upper bounds of the parameter
        if isinstance(scale, (list, tuple)):
            self.scale = (float(scale[0]), float(scale[1]))
        else:
            self.scale = (0.0, float(scale))

        # The quantization flag controls whether the result will be rounded to the nearest integer
        # This is effective for continuous optimization of integer-valued parameters
        self.quantize = quantize

        # Pass the quantization flag and the parameter range to the constructed parameter object
        self.param_kwargs.update(quantize=quantize, scale=self.scale)

        # Change the type cast function into tensor creation
        def _t(x) -> th.Tensor:
            return th.as_tensor(x, dtype=th.float32)
        self._t = _t

        # Verify that the parameter is optimizable
        if not (self.type in (tp.DYNAMIC, tp.OPTIONAL, tp.INT) or is_optimizable(self.type)):
            raise ValueError(f'Parameter {self.name} has a non-optimizable or unrecognized '
                             f'parameter type: {self.type}')
        elif self.type == tp.INT and not quantize:
            raise RuntimeError(f'Attempt to optimize an integer variable without quantization')

    def _normalize(self, value: th.Tensor) -> th.Tensor:
        """Linearly map a Substance parameter value to the corresponding diffmat parameter value.

        Args:
            value (ParamValue): Parameter value in Diffmat.

        Returns:
            ParamValue: Normalized parameter value for Diffmat storage.
        """
        # Detect if the parameter value is out of bound. Substance handles an out-of-bound
        # parameter by automatically adjusting the scale so that the parameter becomes the midpoint
        # of the expanded value range. Note that the scale may vary among elements in a vector
        low, high = self.scale
        new_low = th.where(value < low, value * 2 - high, self._t(low))
        new_high = th.where(value > high, value * 2 - low, self._t(high))
        norm_value = (value - new_low) / (new_high - new_low)

        # Save the new value range
        self.scale = (new_low.item() if new_low.numel() == 1 else new_low,
                      new_high.item() if new_high.numel() == 1 else new_high)

        # Update the parameter range of the constructed parameter object
        self.param_kwargs['scale'] = self.scale

        return norm_value


class ListIndexPT(ConstantParamTranslator):
    """Parameter translator that interprets the parameter value as a list index.
    """
    def __init__(self, root: Optional[ET.Element], source_list: List[Constant],
                 **trans_and_param_kwargs):
        """Initialize the parameter translator.

        Args:
            root (Optional[Element]): Root node of the XML tree. The parameter will be translated
                to its default value if an XML tree is not given.
            source_list (List[Constant]): List of possible parameter values for a Diffmat node
                function that will be indexed by the integer-valued parameter.  
            trans_and_param_kwargs (Dict[str, Any], optional): Keyword arguments for the parent
                translator class constructor and the translated parameter object.
        """
        super().__init__(root, **trans_and_param_kwargs)

        self.source_list = source_list

    def _map(self, value: Constant) -> Constant:
        """Index the source list using the parameter value.

        Args:
            value (Constant): List index.

        Returns:
            Constant: Paramter value item from the list.
        """
        return self.source_list[int(value)]


class GradientMapAnchorPT(ParamTranslator):
    """Parameter translator for color anchors in a gradient map node.

    Static members:
        PARAM_CLASS (Type[Parameter]): Parameter class that the translator instantiates.
    """
    # Parameter object class that this translator produces
    PARAM_CLASS = GradientMapAnchor

    def __init__(self, root: Optional[ET.Element], **trans_and_param_kwargs):
        """Initialize the parameter translator.

        Args:
            root (Optional[Element]): Root node of the XML tree. The parameter will be translated
                to its default value (None) if an XML tree is not given.
            trans_and_param_kwargs (Dict[str, Any], optional): Keyword arguments for the parent
                translator class constructor and the translated parameter object.
        """
        super().__init__(root, **trans_and_param_kwargs, requires_default=False)

        # Parameter 'interpolate' can only be inferred from gradient anchors so we process it here
        # rather than using a different translator, not to complicate back translation
        # --------
        # This field is currently deprecated since the latest Substance Designer version declares
        # it as an individual parameter
        self.interpolate = True

    def _to_literal(self, value_str: List[Dict[str, str]], _) -> th.Tensor:
        """Organize color gradient anchors into a 2D torch tensor.

        Args:
            value_str (List[Dict[str, str]]): The source parameter array in string format,
                organized by a list of records that correspond to parameter array cells.
            _ (Any): Unused placeholder.

        Raises:
            ValueError: The input list is empty.

        Returns:
            Tensor: The anchor array in tensor format.
        """
        # Ensure input value is not empty
        if not value_str:
            raise ValueError('The input cell array must not be empty')

        # Extract positions and colors from respective fields in array cells
        _t = self._t
        positions = _t([to_constant(cell['position'], tp.FLOAT) for cell in value_str])
        colors = _t([to_constant(cell['value'], tp.FLOAT4) for cell in value_str])
        if colors.ndim < 2:
            colors = colors.unsqueeze(1)

        # Assemble positions and colors into the 'anchors' matrix and sort the rows in ascending
        # position order
        anchors = th.hstack((positions.unsqueeze(1), colors))
        anchors = anchors.take_along_dim(anchors.narrow(1, 0, 1).argsort(0), 0)

        return anchors

    def _normalize(self, value: th.Tensor) -> th.Tensor:
        """Convert ascending position coordinates to non-negative finite differences.

        Args:
            value (Tensor): Anchor array in tensor format.

        Returns:
            Tensor: Normalized anchor array for parameter storage.
        """
        norm_value = value.clone()
        norm_value[:, 0] = th.diff(norm_value[:, 0], prepend=th.zeros(1))
        return norm_value


class CurveAnchorPT(ParamTranslator):
    """Parameter translator for tone mapping anchors in a curve node.

    Static members:
        PARAM_CLASS (Type[Parameter]): Parameter class that the translator instantiates.
    """
    # Parameter object class that this translator produces
    PARAM_CLASS = CurveAnchor

    def __init__(self, root: Optional[ET.Element], **trans_and_param_kwargs):
        """Initialize the parameter translator.

        Args:
            root (Optional[Element]): Root node of the XML tree. The parameter will be translated
                to its default value (None) if an XML tree is not given.
            trans_and_param_kwargs (Dict[str, Any], optional): Keyword arguments for the parent
                translator class constructor and the translated parameter object.
        """
        super().__init__(root, **trans_and_param_kwargs, requires_default=False)

    def _to_literal(self, value_str: List[Dict[str, str]], _) -> th.Tensor:
        """Organize curve anchors into a 2D torch tensor.

        Args:
            value_str (List[Dict[str, str]]): The source parameter array in string format,
                organized by a list of records that correspond to parameter array cells.
            _ (Any): Unused placeholder.

        Returns:
            Tensor: The anchor array in tensor format.
        """
        # Extract positions and Bezier control points from respective fields in array cells
        _t = self._t
        positions = [cell['position'] for cell in value_str]
        left_cps = [cell['position'] if int(cell['isLeftBroken']) else cell['left'] \
                    for cell in value_str]
        right_cps = [cell['position'] if int(cell['isRightBroken']) else cell['right'] \
                     for cell in value_str]

        # Convert the strings to tensor
        for str_list in (positions, left_cps, right_cps):
            str_list[:] = [to_constant(s, tp.FLOAT2) for s in str_list]

        # Assemble positions and control points into the 'anchors' matrix and sort the rows in
        # ascending position order
        anchors = th.hstack((_t(positions), _t(left_cps), _t(right_cps)))
        anchors = anchors.take_along_dim(anchors.narrow(1, 0, 1).argsort(0), 0)

        return anchors

    def _normalize(self, value: th.Tensor) -> th.Tensor:
        """Convert ascending position coordinates to non-negative finite differences.

        Args:
            value (Tensor): Anchor array in tensor format.

        Returns:
            Tensor: Normalized anchor array for parameter storage.
        """
        # Compute finite differences for X positions
        positions, left_pos, right_pos = tuple(value[:, i] for i in (0, 2, 4))
        diff_pre = positions.diff(1, prepend=th.zeros(1))
        diff_app = positions.diff(1, append=th.ones(1))

        # Represent the X positions of control points using their offsets from anchors
        left_off = ((positions - left_pos) / diff_pre.clamp_min(1e-16)).clamp_(0, 1)
        right_off = ((right_pos - positions) / diff_app.clamp_min(1e-16)).clamp_(0, 1)

        # Update the value tensor
        norm_value = value.clone()
        norm_value[:, 0] = diff_pre
        norm_value[:, 2] = left_off
        norm_value[:, 4] = right_off

        # Final per-value normalization to [0, 1]
        return super()._normalize(norm_value)
