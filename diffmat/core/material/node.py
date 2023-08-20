from typing import Dict, List, Tuple, Optional, Union
import inspect

import torch as th

from diffmat.core.base import BaseParameter
from diffmat.core.types import NodeFunction, ParamValue, Constant, MultiInputDict, MultiOutputDict
from .base import BaseMaterialNode


class MaterialNode(BaseMaterialNode):
    """Differentiable material graph node class.
    """
    def __init__(self, name: str, type: str, res: int, func: NodeFunction,
                 params: List[BaseParameter] = [], inputs: MultiInputDict = {},
                 outputs: MultiOutputDict = {}, seed: int = 0, **kwargs):
        """Initialize a material graph node.

        Args:
            name (str): Material node name.
            res (int): Output texture resolution (after log2).
            func (NodeFunction): Node function implementation (defined in `functional.py`).
            params (List[BaseParameter], optional): Node parameters. Defaults to [].
            inputs (MultiInputDict, optional): Mapping from input connector names to corresponding
                output slots of predecessor nodes. Defaults to {}.
            outputs (MultiOutputDict, optional): Mapping from output connector names to a list of
                successor nodes. Defaults to {}.
            seed (int, optional): Random seed to node function. Defaults to 0.
            kwargs (Dict[str, Any]): Keyword arguments to pass into the parent class constructor.
        """
        super().__init__(name, type, res, params=params, inputs=inputs, outputs=outputs,
                         seed=seed, **kwargs)

        self.func = func

        # Inspect the node function to determine whether or not some global or node-specific
        # options should be passed to the node function
        self._inspect_node_function()

    def _inspect_node_function(self):
        """Inspect the node function to check global options like 'use_alpha', 'res_h', and
        'res_w', which should be provided by either the graph or the node itself.
        """
        # Extract the arguments of the node function (with unwrapping)
        func_params = inspect.signature(self.func).parameters

        # Identify arguments that may be global options, i.e., they are not covered by parameter
        # translators or input connections
        all_params = {name for name, param in func_params.items()
                      if param.default is not param.empty}
        all_params -= self.inputs.keys()
        covered_param_names = set(p.name for p in self.params)

        self._required_options = all_params - covered_param_names
        self._required_params = all_params & covered_param_names

    def evaluate(self, *img_list: Optional[th.Tensor], exposed_params: Dict[str, ParamValue] = {},
                 benchmarking: bool = False, **options: Constant) -> \
                     Union[th.Tensor, Tuple[th.Tensor, ...]]:
        """Evaluate the node using the stored node function.

        Args:
            img_list (List[Tensor], optional): Input texture maps. Defaults to [].
            exposed_params (Dict[str, ParamValue], optional): Exposed parameter values of the
                material graph. Defaults to {}.
            benchmarking (bool, optional): Whether or not to benchmark runtime. Defaults to False.
            options (Dict[str, Constant], optional): Global options from the material graph, for
                example, the `use_alpha` flag that enables the alpha channel.

        Returns:
            Tensor or Tuple[Tensor]: Output texture map(s).
        """
        # Compute the dictionary that maps node parameter names to values
        node_params, _ = self._evaluate_node_params(exposed_params)
        seed_offset: int = node_params.get('seed', 0)

        # Discard parameters not needed by the node function
        node_params = {key: val for key, val in node_params.items()
                       if key in self._required_params}

        # For benchmarking mode, detach input images and node parameters from the original graph
        # to construct an independent computation graph for the node
        if benchmarking:
            _ct = lambda v: v.detach().clone().requires_grad_() \
                            if isinstance(v, th.Tensor) and th.is_floating_point(v) else v
            img_list = [_ct(img) for img in img_list]
            node_params = {key: _ct(val) for key, val in node_params.items()}

        # Add node-specific options to the global option dictionary and only keep those required
        options = options.copy()
        options.update({
            'res_h': 1 << self.res, 'res_w': 1 << self.res,
            'device': self.device
        })
        options = {key: val for key, val in options.items() if key in self._required_options}

        # Use a temporary random seed environment
        with self.temp_rng(seed = self.seed + seed_offset):

            # Node function profiling
            with self.timer(f'Node {self.name}', log_level='debug'):
                ret = self.func(*img_list, **node_params, **options)

        return ret


class ExternalInputNode(MaterialNode):
    """Subclass of nodes that read externally generated textures as input.
    """
    def __init__(self, name: str, type: str, res: int, func: NodeFunction,
                 inputs: MultiInputDict = {}, outputs: MultiOutputDict = {}, **kwargs):
        """Initialize an external input node.

        Args:
            name (str): External input node name.
            res (int): Output texture resolution (after log2).
            func (NodeFunction): Node function implementation.
            params (List[BaseParameter], optional): Node parameters. Defaults to [].
            inputs (MultiInputDict, optional): Mapping from input connector names to corresponding
                output slots of predecessor nodes. Defaults to {}.
            outputs (MultiOutputDict, optional): Mapping from output connector names to a list of
                successor nodes. Defaults to {}.
            seed (int, optional): Random seed to node function. Defaults to 0.
            kwargs (Dict[str, Any]): Keyword arguments to pass into the parent class constructor.
        """
        super().__init__(name, type, res, func, inputs=inputs, outputs=outputs, **kwargs)

    def evaluate(self, external_input_dict: Dict[str, th.Tensor], **options: Constant) -> \
            Union[th.Tensor, Tuple[th.Tensor, ...]]:
        """Evaluate the node to fetch one or more external input textures.

        Args:
            external_input_dict (Dict[str, Tensor]): Dictionary of all external input textures,
                i.e., those read from dependency files or generated by the Substance Automation
                Toolkit.
            options (Dict[str, Constant], optional): Global options from the material graph, for
                example, the `use_alpha` flag that enables the alpha channel.

        Returns:
            Tensor or Tuple[Tensor]: Loaded external input texture(s).
        """
        return self.func(external_input_dict, use_alpha=options['use_alpha'])
