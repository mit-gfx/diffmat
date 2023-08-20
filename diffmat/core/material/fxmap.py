from typing import Union, Optional, List, Dict
import itertools

import torch as th

from diffmat.core.base import BaseParameter
from diffmat.core.fxmap import FXMapExecutorV2 as FXE, FXMapGraph
from diffmat.core.fxmap.util import get_opacity
from diffmat.core.types import Constant, ParamValue, MultiInputDict, MultiOutputDict
from diffmat.core.util import FILTER_OFF
from .base import BaseMaterialNode
from .util import input_check_all_positional


class FXMap(BaseMaterialNode):
    """FX-map node class.
    """
    def __init__(self, name: str, type: str, res: int, func: Optional[FXMapGraph] = None,
                 params: List[BaseParameter] = [], inputs: MultiInputDict = {},
                 outputs: MultiOutputDict = {}, seed: int = 0, **kwargs):
        """Initialize the FX-map node object.

        Args:
            name (str): FX-map node name.
            res (int): Output texture resolution (after log2).
            func (Optional[FXMapGraph], optional): FX-map graph. Defaults to None.
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
        if func:
            self.link_as_parent(func)

        # Instantiate an FX-map executor which carries out pattern generation tasks from FX-map
        # nodes
        kwargs['parent'] = self
        kwargs.pop('allow_ablation', None)
        kwargs.pop('is_generator', None)
        self.executor = FXE(res, **kwargs)

    def compile(self, exposed_param_levels: Dict[str, int] = {}, master_seed: int = 0,
                inherit_seed: bool = True) -> Dict[str, int]:
        """Compile the FX-map node, including the FX-map graph and its parameters.

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
        # Compile the node parameters to get the active variable set
        var_levels = super().compile(exposed_param_levels, master_seed, inherit_seed)

        # Compile the FX-map graph
        if self.func:
            self.func.compile(var_levels)

        return var_levels

    @input_check_all_positional(class_method=True)
    def evaluate(self, img_bg: Optional[th.Tensor], *img_list: Optional[th.Tensor],
                 exposed_params: Dict[str, ParamValue] = {}, benchmarking: bool = False,
                 **_: Constant) -> th.Tensor:
        """Evaluate the FX-map node by traversing the FX-map graph and generating/compositing
        atomic patterns.

        Args:
            img_bg (Optional[Tensor]): Background image. Defaults to None.
            img_list (List[Tensor], optional): Input texture maps. Defaults to [].
            exposed_params (Dict[str, ParamValue], optional): Exposed parameter values of the
                material graph. Defaults to {}.
            benchmarking (bool, optional): Whether or not to benchmark runtime. Defaults to False.

        Returns:
            Tensor: Output texture map.
        """
        # Evaluate the node parameters first
        node_params, var = self._evaluate_node_params(exposed_params)

        # For benchmarking mode, detach input images and node parameters from the original graph
        # to construct an independent computation graph for the node
        if benchmarking:
            _ct = lambda v: v.detach().clone().requires_grad_() \
                            if isinstance(v, th.Tensor) and th.is_floating_point(v) else v
            img_list = [_ct(img) for img in img_list]
            var = {key: _ct(val) for key, val in var.items()}

        # Initialize the executor (which also sets the background)
        mode = node_params['mode'] if img_bg is None else \
               'gray' if img_bg.shape[1] == 1 else 'color'
        self.executor.reset(
            img_bg, *img_list, mode=mode, background_color=node_params['background_color'])

        if self.func:
            # Evaluate the FX-map graph to collect pattern generation jobs at the executor
            with self.temp_rng(seed=self.seed + node_params.get('seed', 0)):
                with self.timer(f'Node {self.name} (collection)', log_level='debug'):
                    self.func.evaluate(self.executor, var=var)

            # Compute blending opacity values at each depth level. The values are multiplied by a
            # global opacity
            max_depth = self.func.max_depth()
            blending_opacity = get_opacity(node_params['roughness'], max_depth)
            blending_opacity = self._t(blending_opacity) * node_params['opacity']

            # Run the FX-map executor to produce the output image
            with self.timer(f'Node {self.name} (execution)', log_level='debug'):
                img_out = self.executor.evaluate(blending_opacity)

        return img_out

    def _filter_params(self, filter_generator: int = FILTER_OFF) -> List[BaseParameter]:
        """Return node parameters filtered by a set of predefined rules.

        Args:
            filter_generator (int, optional): See `BaseMaterialNode._filter_params` for definition.
                Defaults to `-1 = off`.

        Returns:
            List[BaseParameter]: List of node parameters, if not empty.
        """
        # Determine whether the node has non-empty inputs and check against the filter
        has_input = any(self.inputs.values())
        if filter_generator < 0 or filter_generator == (self.is_generator or not has_input):
            return list(itertools.chain(self.params, *(gn.params for gn in self.func.nodes)))

        return []

    def to_device(self, device: Union[str, th.device]):
        """Move the FX-map node to a specified device (e.g., CPU or GPU).

        Args:
            device (Union[str, th.device]): Target device.
        """
        # Move parameters and set the device attribute
        super().to_device(device)

        # Move the FX-map graph
        if self.func:
            self.func.to_device(device)
