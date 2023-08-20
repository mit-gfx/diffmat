from typing import List, Tuple, Dict, Optional, Union

import torch as th

from diffmat.core.base import BaseParameter
from diffmat.core.function import FunctionGraph
from diffmat.core.types import Constant, ParamValue, MultiInputDict, MultiOutputDict
from .base import BaseMaterialNode
from .functional import resize_image_color
from .util import input_check_all_positional


class PixelProcessor(BaseMaterialNode):
    """Pixel processor node class.

    Static members:
        POS_FACTORY (Dict[Tuple[int, int], th.Tensor]): Factory of '$pos' matrices with various
            sizes. '$pos' refers to per-pixel center coordinates and is widely used in image
            sampling nodes.
    """
    # Factory of '$pos' matrices with various sizes. '$pos' refers to per-pixel center coordinates
    # and is widely used in images sampling nodes
    POS_FACTORY: Dict[Tuple[int, int], th.Tensor] = {}

    def __init__(self, name: str, type: str, res: int, func: Optional[FunctionGraph] = None,
                 params: List[BaseParameter] = [], inputs: MultiInputDict = {},
                 outputs: MultiOutputDict = {}, seed: int = 0, **kwargs):
        """Initialize the pixel processor node.

        Args:
            name (str): Pixel processor node name.
            res (int): Output texture resolution (after log2).
            func (Optional[FunctionGraph], optional): Per-pixel function graph. Defaults to None.
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

        # Get the '$pos' matrix
        self.internal_params['$pos'] = self._get_pos_matrix(1 << res, 1 << res)

    def _get_pos_matrix(self, res_h: int, res_w: int) -> th.Tensor:
        """Generate or fetch the '$pos' matrix of a pixel processor node, which stores per-pixel
        center coordinates in a shape of (1, H, W, 2).

        Args:
            res_h (int): Height of the output texture.
            res_w (int): Width of the output texture.

        Returns:
            th.Tensor: per-pixel '$pos' values.
        """
        size = (res_h, res_w)
        pos = self.POS_FACTORY.get(size)

        # Construct a new '$pos' matrix if the factory does not have it
        if pos is None:
            x_lin = th.linspace(0.5 / res_w, 1 - 0.5 / res_w, res_w, device=self.device)
            y_lin = th.linspace(0.5 / res_h, 1 - 0.5 / res_h, res_h, device=self.device)
            pos = th.stack(th.meshgrid(x_lin, y_lin, indexing='xy'), dim=2)
            self.POS_FACTORY[size] = pos

        return pos.to(self.device)

    def compile(self, exposed_param_levels: Dict[str, int], master_seed: int = 0,
                inherit_seed: bool = True) -> Dict[str, int]:
        """Compile dynamic node parameters and the per-pixel function.

        Args:
            exposed_param_levels (Dict[str, int]): Value category mapping of exposed parameters in
                a material graph.
            master_seed (int, optional): Graph-wide random seed, to which per-node random seeds
                serve as offsets in the seed value. Defaults to 0.
            inherit_seed (bool, optional): Switch for overwriting the internal random seed using
                the provided `master_seed`. Defaults to True.

        Returns:
            Dict[str, int]: Value category mapping of named variables accessible from this node.
        """
        # Compile dynamic node parameters
        all_levels = super().compile(exposed_param_levels, master_seed, inherit_seed)

        # Compile the per-pixel function
        if self.func:
            self.func.compile(all_levels)

        return all_levels

    @input_check_all_positional(class_method=True)
    def evaluate(self, *img_list: Optional[th.Tensor], exposed_params: Dict[str, ParamValue] = {},
                 benchmarking: bool = False, **options: Constant) -> th.Tensor:
        """Evaluate the pixel processor by executing the per-pixel function.

        Args:
            img_list (List[Tensor], optional): Input texture maps. Defaults to [].
            exposed_params (Dict[str, ParamValue], optional): Exposed parameter values of the
                material graph. Defaults to {}.
            benchmarking (bool, optional): Whether or not to benchmark runtime. Defaults to False.
            options (Dict[str, Constant], optional): Global options from the material graph, for
                example, the `use_alpha` flag that enables the alpha channel.

        Returns:
            Tensor: Output texture map.
        """
        # Compute the dictionary that maps node parameter names to values
        node_params, var = self._evaluate_node_params(exposed_params)

        # For benchmarking mode, detach input images and node parameters from the original graph
        # to construct an independent computation graph for the node
        if benchmarking:
            _ct = lambda v: v.detach().clone().requires_grad_() \
                            if isinstance(v, th.Tensor) and th.is_floating_point(v) else v
            img_list = [_ct(img) for img in img_list]
            var = {key: _ct(val) for key, val in var.items()}

        # Output a black image when a per-pixel function is not provided
        img_size = 1 << self.res
        grayscale = node_params['mode'] == 'gray'
        use_alpha = options['use_alpha']

        if self.func is None:
            num_channels = 1 if grayscale else 4 if use_alpha else 3
            img = th.zeros(1, num_channels, img_size, img_size, device=self.device)
            if num_channels == 4:
                img[:, :3] = 1.0

            return img

        # Evaluate the per-pixel function in the local random number generator environment
        with self.temp_rng(seed=self.seed + node_params.get('seed', 0)):
            with self.timer(f'Node {self.name}', log_level='debug'):
                result: th.Tensor = self.func.evaluate(*img_list, var=var)

        # The per-pixel flag in the function graph makes sure that the current result is at least
        # a 3D tensor but its layout is HxWxC, so it should be converted to a 4D BxCxHxW tensor
        result = result.unsqueeze(0) if result.ndim == 3 else result
        img = result.movedim(3, 1).expand(1, -1, img_size, img_size)

        # Convert the output to grayscale or color image
        num_channels = 1 if grayscale else 4 if use_alpha else 3
        img = resize_image_color(img, num_channels)

        return img.clamp(0.0, 1.0)

    def to_device(self, device: Union[str, th.device]):
        """Move the pixel processor node to a specified device (CPU or GPU).

        Args:
            device (Union[str, th.device]): Target device.
        """
        # Move the per-pixel function and the parameters
        if self.func:
            self.func.to_device(device)

        super().to_device()
