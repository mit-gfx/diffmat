from abc import abstractmethod
from typing import Union, Optional, List, Tuple, Dict, Callable, Iterator
import random

import torch as th

from diffmat.core.base import BaseNode, BaseParameter
from diffmat.core.types import (
    Constant, ParamValue, IntParamValue, NodeSummary, MultiInputDict, MultiOutputDict, DeviceType
)
from diffmat.core.util import OL, FILTER_OFF, FILTER_NO, FILTER_YES, to_const
from .util import (
    get_parameters, get_parameters_as_config, set_parameters_from_config,
    get_integer_parameters, num_integer_parameters, set_integer_parameters_from_list,
    get_integer_parameters_as_config, set_integer_parameters_from_config, timed_func
)


class BaseMaterialNode(BaseNode[List[BaseParameter], MultiInputDict, MultiOutputDict]):
    """A base class for differentiable material nodes where parameters are represented by objects.
    """
    def __init__(self, name: str, type: str, res: int, params: List[BaseParameter] = [],
                 inputs: MultiInputDict = {}, outputs: MultiOutputDict = {},
                 seed: int = 0, allow_ablation: bool = False, is_generator: bool = False,
                 **kwargs):
        """Initialize the base material node object (including internal node parameters).

        Args:
            name (str): Material node name.
            res (int): Output texture resolution (after log2).
            params (List[BaseParameter], optional): List of node parameters. Defaults to [].
            inputs (MultiInputDict, optional): Mapping from input connector names to corresponding
                output slots of predecessor nodes. Defaults to {}.
            outputs (MultiOutputDict, optional): Mapping from output connector names to a list of
                successor nodes. Defaults to {}.
            seed (int, optional): Random seed to node function. Defaults to 0.
            allow_ablation (bool, optional): Switch for allowing ablation of the node. This flag
                mainly applies to FX-Map, Pixel Processor, and most generator nodes that were
                not supported in earlier versions of DiffMat. Setting the flag allows them to be
                replaced by dummy nodes or excluded from node parameter optimization, which is
                helpful to ablation studies. Defaults to False.
            is_generator (bool, optional): Indicates whether the node is a generator node.
                Defaults to False.
            kwargs (Dict[str, Any]): Keyword arguments to pass into the parent class constructor.
        """
        super().__init__(name, type, params, inputs, outputs, **kwargs)

        self.res = res
        self.seed = seed
        self.allow_ablation = allow_ablation
        self.is_generator = is_generator

        # Internal node parameters
        self.internal_params: Dict[str, Constant] = {
            '$size': [float(1 << res), float(1 << res)],
            '$sizelog2': [float(res), float(res)],
            '$normalformat': 0,
            '$tiling': 0,
        }

    def compile(self, exposed_param_levels: Dict[str, int] = {},
                master_seed: int = 0, inherit_seed: bool = True) -> Dict[str, int]:
        """Compile function graphs inside dynamic node parameters, and acquire the value categories
        of all named variables effective to this node for static type checking.

        Args:
            exposed_param_levels (Dict[str, int], optional): Value category mapping of exposed
                parameters in a material graph. Defaults to {}.
            master_seed (int, optional): Graph-wide random seed, to which per-node random seeds
                serve as offsets in the seed value. Defaults to 0.
            inherit_seed (bool, optional): Switch for overwriting the internal random seed using
                the provided `master_seed`. Defaults to True.

        Returns:
            Dict[str, int]: Value category mapping of named variables accessible from this node.
        """
        # Add the level information of internal parameters and non-dynamic parameters
        var_levels = exposed_param_levels.copy()
        var_levels.update({key: OL.get_level(val) for key, val in self.internal_params.items()})

        # Inherit the graph-level random seed
        if inherit_seed:
            self.seed = master_seed

        # Initialize the random number generator
        rng_state = random.getstate()
        random.seed(self.seed)

        for param in (p for p in self.params if p.IS_DYNAMIC):
            param.compile(var_levels)

        # Reset the random number generator
        random.setstate(rng_state)

        return var_levels

    def _evaluate_node_params(
            self, exposed_params: Dict[str, ParamValue] = {}
        ) -> Tuple[Dict[str, Optional[ParamValue]], Dict[str, ParamValue]]:
        """Compute the values of node parameters (include dynamic ones). Also returns the
        collection of variables effective in this node.

        Args:
            exposed_params (Dict[str, ParamValue], optional): Name-to-value mapping for exposed
                parameters in the material graph. Defaults to {}.

        Returns:
            Dict[str, Optional[ParamValue]]: Node parameter value dictionary.
            Dict[str, ParamValue]: Named variables value dictionary.
        """
        # Initialize the dictionary that maps node parameter names to values
        node_params: Dict[str, Optional[ParamValue]] = {}

        # Evaluate dynamic parameters (be aware of inter-parameter dependency)
        var = exposed_params.copy()
        var.update(self.internal_params)

        for param in self.params:
            if param.IS_DYNAMIC:
                value = param.evaluate(var)
                if isinstance(value, th.Tensor) and value.dtype == th.long:
                    value = to_const(value)
            else:
                value = param.evaluate()
            node_params[param.name] = value

            # Update the tiling variable since it can be referenced as an internal parameter
            if param.name == 'tiling':
                var['$tiling'] = value

        return node_params, var

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Union[th.Tensor, Tuple[th.Tensor, ...]]:
        """Node function wrapper. See `functional.py` for actual implementations.
        """
        ...

    def benchmark_forward(
            self, *args, **kwargs
        ) -> Tuple[Union[th.Tensor, Tuple[th.Tensor, ...]], float]:
        """Wrapper of the `evaluate` method with additional timing support.
        """
        gpu_mode = self.device.type == 'cuda'
        return timed_func(self.evaluate, args, kwargs, gpu_mode=gpu_mode)

    def benchmark_backward(
            self, *args, **kwargs
        ) -> Tuple[Union[th.Tensor, Tuple[th.Tensor, ...]], float]:
        """Wrapper of the `evaluate` method that independently times the backward pass.
        """
        # Wrapper function of `evaluate` that performs a backward pass
        def func(*args, **kwargs) -> Union[th.Tensor, Tuple[th.Tensor, ...]]:
            ret = self.evaluate(*args, **kwargs, benchmarking=True)
            obj = sum(v.sum() for v in ret if isinstance(v, th.Tensor)) \
                  if isinstance(ret, (tuple, list)) else ret.sum()
            obj.backward()
            return ret

        gpu_mode = self.device.type == 'cuda'
        return timed_func(func, args, kwargs, gpu_mode=gpu_mode)

    def train(self):
        """Switch to training mode where all optimizable parameters require gradient.
        """
        for param in self.parameters(filter_requires_grad=FILTER_NO):
            param.requires_grad_(True)

    def eval(self):
        """Switch to evaluation mode where no optimizable parameter requires gradient.
        """
        for param in self.parameters():
            param.requires_grad_(False)

    def _filter_params(self, filter_generator: int = FILTER_OFF) -> List[BaseParameter]:
        """Return node parameters filtered by a set of predefined rules.

        Args:
            filter_generator (int, optional): Option for returning node parameters on the
                condition that the node is (not) a generator. Valid cases are:
                    `1 = yes` means parameters will be returned only if the node is a generator;
                    `0 = no` means parameters will be returned only if the node is not a generator;
                    `-1 = off` means node parameters will always be returned.
                Defaults to -1.

        Returns:
            List[BaseParameter]: List of node parameters, if not empty.
        """
        # Determine whether the node has non-empty inputs and check against the filter
        has_input = any(self.inputs.values())
        if filter_generator < 0 or filter_generator == (self.is_generator or not has_input):
            return self.params

        return []

    def parameters(self, filter_generator: int = FILTER_OFF,
                   filter_requires_grad: int = FILTER_YES, detach: bool = False,
                   flatten: bool = False) -> Iterator[th.Tensor]:
        """Return an iterator over optimizable, continuous parameter values in the material node
        (tensor views rather than copies).

        Args:
            filter_generator (int, optional): Option for node parameter visibility contigent on
                whether the node is (not) a generator node. Valid cases are:
                    `1 = yes` means parameters are visible only if the node is a generator;
                    `0 = no` means parameters are visible only if the node is not a generator;
                    `-1 = off` means node parameters are always visible.
                Defaults to `off`.
            filter_requires_grad (int, optional): Option for filtering out parameters that require
                gradient. Valid cases are:
                    `1 = yes` means parameters that require gradient are returned;
                    `0 = no` means parameters that don't require gradient are returned;
                    `-1 = off` means all parameters are returned.
                Defaults to `yes`.
            detach (bool, optional): Whether returned tensor views are detached (i.e., don't
                require gradient). Defaults to False.
            flatten (bool, optional): Whether returned tensor views are flattened.
                Defaults to False.

        Yields:
            Iterator[Tensor]: Tensor views of optimizable node parameter values.
        """
        return get_parameters(self._filter_params(filter_generator=filter_generator),
                              filter_requires_grad=filter_requires_grad,
                              detach=detach, flatten=flatten)

    def get_parameters_as_config(self, filter_generator: int = FILTER_OFF,
                                 constant: bool = False) -> Dict[str, Dict[str, ParamValue]]:
        """Return parameter values of the material node as a dict-type configuration in the
        following format:
        ```yaml
        {param_name}: # x many
            value: {param_value}
            normalize: False/True # optional for optimizable parameters
        ```

        Args:
            filter_generator (int, optional): See the `parameters` method for details.
                Defaults to `-1 = off`.
            constant (bool, optional): Whether to convert parameter values to literals (float,
                int, or bool-typed values). Defaults to False.
        """
        return get_parameters_as_config(
            self._filter_params(filter_generator=filter_generator), constant=constant)

    def set_parameters_from_config(self, config: Dict[str, Dict[str, ParamValue]]):
        """Set parameter values of the material node from a nested dict-type configuration in the
        following format:
        ```yaml
        {param_name}: # x many
            value: {param_value}
            normalize: False/True # optional for optimizable parameters
        ```

        Args:
            config (Dict[str, Dict[str, ParamValue]]): Parameter configuration as outlined above.
        """
        set_parameters_from_config(self.params, config)

    def integer_parameters(self, filter_generator: int = FILTER_OFF) -> Iterator[IntParamValue]:
        """An iterator that traverses all optimizable integer parameters in a material node.

        Args:
            filter_generator (int, optional): See the `parameters` method for details.
                Defaults to `-1 = off`.

        Yields:
            Iterator[IntParamValue]: Integer parameter values.
        """
        return get_integer_parameters(self._filter_params(filter_generator=filter_generator))

    def num_integer_parameters(self, **kwargs) -> int:
        """Count the number of optimizable integer parameters in the material node.

        Args:
            kwargs (Dict[str, Any], optional): Keyword arguments to pass into the `_filter_params`
                method.

        Returns:
            int: Number of optimizable integer parameters.
        """
        return num_integer_parameters(self._filter_params(**kwargs))

    def set_integer_parameters_from_list(self, values: List[int], **kwargs):
        """Set optimizable integer parameter values of the material node from an integer list.

        Args:
            values (List[int]): List of integer parameter values.
            kwargs (Dict[str, Any], optional): Keyword arguments to pass into the `_filter_params`
                method.
        """
        set_integer_parameters_from_list(self._filter_params(**kwargs), values)

    def get_integer_parameters_as_config(self, **kwargs) -> Dict[str, Dict[str, IntParamValue]]:
        """Return optimizable integer parameter values of the material node as a dict-type
        configuration in the following format:
        ```yaml
        {param_name}: # x many
          value: {param_value}
          low: {param_low_bound}
          high: {param_high_bound}
        ```

        Args:
            kwargs (Dict[str, Any], optional): Keyword arguments to pass into the `_filter_params`
                method.

        Returns:
            Dict[str, Dict[str, IntParamValue]]: Integer parameter configuration as outlined above.
        """
        return get_integer_parameters_as_config(self._filter_params(**kwargs))

    def set_integer_parameters_from_config(self, config: Dict[str, Dict[str, IntParamValue]]):
        """Set optimizable integer parameter values of the material node from a nested dict-type
        configuration in the following format:
        ```yaml
        {param_name}: # x many
          value: {param_value}
        ```

        Args:
            config (Dict[str, Dict[str, IntParamValue]]): Integer parameter configuration as
                outlined above.
        """
        set_integer_parameters_from_config(self.params, config)

    def summarize(self) -> NodeSummary:
        """Generate a summary of node status, including name, I/O, and parameters.

        Returns:
            NodeSummary: A dictionary that summarizes essential information of the node, including
                name, input connections, and node parameter values.
        """
        get_variable_name: Callable[[str, str], str] = \
            lambda name, output: f'{name}_{output}' if output else name

        return {
            'name': self.name,
            'input': [get_variable_name(*val) if val is not None else None \
                      for val in self.inputs.values()],
            'param': dict(tuple(p.summarize().values()) for p in self.params)
        }

    def to_device(self, device: DeviceType = 'cpu'):
        """Move the material graph node to a specified device (e.g., CPU or GPU).

        Args:
            device (DeviceType, optional): Target device ID. Defaults to 'cpu'.
        """
        # Move data members to the target device
        for param in self.params:
            param.to_device(device)

        super().to_device(device)
