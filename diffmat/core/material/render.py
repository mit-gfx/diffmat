from functools import partial
from typing import Dict, Tuple, List, Union, Optional, Iterator

import torch as th

from diffmat.core.base import BaseEvaluableObject
from diffmat.core.types import DeviceType
from .util import input_check_all_positional, color_input_check, grayscale_input_check
from .param import ConstantParameter, Parameter


# Render helper functions
def brdf(n_dot_h: th.Tensor, alpha: th.Tensor, f0: th.Tensor, eps: float = 1e-12) -> th.Tensor:
    """Compute the specular BRDF function.

    Args:
        n_dot_h (Tensor): Dot products of normal vectors and half vectors.
        alpha (Tensor): Roughness (or squared roughness) map.
        f0 (Tensor): Ambient light intensity.
        eps (float, optional): A small epsilon that thresholds denominators to prevent division
            by zero. Defaults to 1e-12.

    Returns:
        Tensor: BRDF values.
    """
    # Tensor creation function
    _t = partial(th.as_tensor, device=n_dot_h.device)

    # Compute the GGX normal distribution function
    alpha_sq = alpha ** 2
    numerator = th.where(n_dot_h > 0, alpha_sq, _t(0.0))
    denominator = (n_dot_h ** 2 * (alpha_sq - 1) + 1) ** 2 * th.pi
    ndf = numerator / denominator.clamp_min(eps)

    # Assume D = ndf, F = f0, and G = n_dot_h ** 2, we have BRDF = D * F / 4
    return f0 * ndf * 0.25


class Renderer(BaseEvaluableObject):
    """Differentiable physics-based renderer using SVBRDF maps.

    Static members:
        CHANNELS (Dict[str, Tuple[bool, float]]): Supported types of output SVBRDF maps in a
            procedural material graph.
    """
    # Description of supported output channels and their format
    #   channel: (is color, default value)
    # where 'is color' is a bool - True = color; False = grayscale
    # default value is either 1.0 or 0.0
    CHANNELS: Dict[str, Tuple[bool, float]] = {
        'basecolor': (True, 0.0),
        'normal': (True, 1.0),
        'roughness': (False, 1.0),
        'metallic': (False, 0.0),
        # 'height': (False, 0.5),
        # 'ambientocclusion': (False, 1.0),
        'opacity': (False, 1.0)
    }

    def __init__(self, size: float = 30.0, camera: List[float] = [0.0, 0.0, 25.0],
                 light_color: List[float] = [3300.0, 3300.0, 3300.0], f0: float = 0.04,
                 normal_format: str = 'dx', optimizable: bool = False,
                 device: DeviceType = 'cpu', **kwargs):
        """Initialize the differentiable renderer.

        Args:
            size (float, optional): Real-world size of the texture. Defaults to 30.0.
            camera (List[float], optional): Position of the camera relative to the texture center.
                The texture always resides on the X-Y plane in center alignment.
                Defaults to [0.0, 0.0, 25.0].
            light_color (List[float], optional): Light intensity in RGB.
                Defaults to [3300.0, 3300.0, 3300.0].
            f0 (float, optional): Normalized ambient light intensity. Defaults to 0.04.
            normal_format (str, optional): Controls how the renderer interprets the format of
                normal maps (DirectX 'dx' or OpenGL 'gl'). Defaults to 'dx'.
            optimizable (bool, optional): Whether texture size and light intensity are regarded as
                optimizable. Defaults to False.
            device (DeviceType, optional): Target device ID where the renderer is placed.
                Defaults to 'cpu'.
        """
        super().__init__(device=device, **kwargs)

        # Tensor creation helper function
        _t = self._at

        # Create constant parameters or optimizable parameters according to the optimization flag
        if optimizable:
            size_scale = (0.0, 300.0)
            light_color_scale = (0.0, 10000.0)
            self.size = Parameter(
                'size', _t(size) / size_scale[1], scale=size_scale, device=device)
            self.light_color = Parameter(
                'light_color', _t(light_color) / light_color_scale[1], scale=light_color_scale,
                device=device)
        else:
            self.size = ConstantParameter('size', size, device=device)
            self.light_color = ConstantParameter('light_color', light_color, device=device)

        # Camera and base reflectivity are always fixed
        self.camera = ConstantParameter('camera', camera, device=device)
        self.f0 = ConstantParameter('f0', f0, device=device)

        self.normal_format = normal_format
        self.optimizable = optimizable
        self.device = device

    @input_check_all_positional(class_method=True)
    def evaluate(self, *tensors: th.Tensor, eps: float = 1e-12) -> th.Tensor:
        """Generate a rendered image from SVBRDF maps of an input texture.

        Args:
            tensors (Sequence[Tensor], optional): Sequence of input SVBRDF maps. Each map is
                interpreted per the order defined in `Renderer.CHANNELS`.
            eps (float, optional): A small epsilon that thresholds denominators to prevent division
                by zero. Defaults to 1e-12.

        Returns:
            Tensor: Rendered image using input SVBRDF maps.
        """
        # Check input validity
        for i, (label, (is_color, _)) in enumerate(self.CHANNELS.items()):
            check_func = color_input_check if is_color else grayscale_input_check
            check_func(tensors[i], label)

        # Discard the alpha channel of basecolor and normal, map the basecolor to gamma space, and
        # scale the normal image to [-1, 1]
        albedo, normal, roughness, metallic, opacity, *_ = tensors
        albedo = albedo.narrow(1, 0, 3) ** 2.2
        normal = ((normal.narrow(1, 0, 3) - 0.5) * 2.0)

        # Process DirectX normal format by default
        if self.normal_format == 'dx':
            normal.select(1, 1).neg_()

        # Read render parameters
        _t = self._at
        size: Union[float, th.Tensor] = self.size.evaluate()
        light_color = _t(self.light_color.evaluate()).view(3, 1, 1)
        camera = _t(self.camera.evaluate()).view(3, 1, 1)
        f0: float = self.f0.evaluate()

        # Account for metallicity - increase base reflectivity and decrease albedo
        f0 = th.lerp(_t(f0), albedo, metallic)
        albedo = albedo * (1.0 - metallic)

        # Calculate 3D pixel center positions (image lies on the x-y plane)
        img_size: int = albedo.shape[2]
        x_coords = th.linspace(0.5 / img_size - 0.5, 0.5 - 0.5 / img_size, img_size,
                               device=self.device)
        x_coords = x_coords * size

        x, y = th.meshgrid(x_coords, x_coords, indexing='xy')
        pos = th.stack((x, -y, th.zeros_like(x)))

        # Calculate view directions (half vectors from camera to each pixel center)
        omega = camera - pos
        dist: th.Tensor = th.norm(omega, dim=0, keepdim=True)
        half = omega / dist.clamp_min(eps)

        # Get the diffuse term and clamp to [0, 1]
        n_dot_h = (normal * half).sum(1, keepdim=True)
        geometry_times_light = n_dot_h * light_color / (dist * dist).clamp_min(eps)
        diffuse = geometry_times_light * albedo / th.pi
        diffuse = th.clamp(diffuse, 0.0, 1.0)

        # Get the specular term using the BRDF function
        specular = geometry_times_light * brdf(n_dot_h, roughness ** 2, f0, eps=eps)
        specular = th.clamp(specular, 0.0, 1.0)

        # Calculate final rendering
        rendering = th.clamp(diffuse + specular, eps, 1.0) ** (1 / 2.2)
        rendering = th.lerp(f0, rendering, opacity)
        return rendering

    # Alias of the evaluate function for direct invocation
    __call__ = evaluate

    def parameters(self, detach: bool = False, flatten: bool = False) -> Iterator[th.Tensor]:
        """An iterator over the optimizable parameter values in the material node (views rather
        than copies).

        Args:
            detach (bool, optional): Whether returned tensor views are detached (i.e., don't
                require gradient). Defaults to False.
            flatten (bool, optional): Whether returned tensor views are flattened.
                Defaults to False.

        Yields:
            Iterator[Tensor]: Tensor views of optimizable rendering parameters.
        """
        if self.optimizable:
            for param in (self.size, self.light_color):
                data: th.Tensor = param.data
                data = data.detach() if detach else data
                data = data.flatten() if flatten else data
                yield data

    def num_parameters(self) -> int:
        """Count the number of optimizable parameter values (floating-point numbers) in the
        material node.

        Returns:
            int: Number of optimizable parameter elements.
        """
        return sum(view.shape[0] for view in self.parameters(detach=True, flatten=True))

    def get_parameters_as_tensor(self) -> Optional[th.Tensor]:
        """Get the values of optimizable rendering parameters as a 1D torch tensor.

        Returns:
            Optional[Tensor]: Flattened concatenation of optimizable rendering parameter values,
                or None if they are not optimizable.
        """
        if self.optimizable:
            return th.cat(list(self.parameters(detach=True, flatten=True)))
        else:
            return None

    def set_parameters_from_tensor(self, values: th.Tensor):
        """Set the optimizable rendering parameters from a 1D torch tensor.

        Args:
            values (Tensor): Source parameter values.

        Raises:
            ValueError: The input is not a 1D PyTorch tensor.
            RuntimeError: The method is invoked whereas rendering parameters are not optimizable.
            ValueError: Input tensor does not match the number of optimizable parameters in size.
        """
        # Check if the input is a 1D torch tensor
        if not isinstance(values, th.Tensor) or values.ndim != 1:
            raise ValueError('The input must be a 1D torch tensor.')

        # Check if the number of provided parameters match the number of optimizable ones
        values = values.detach()
        if not self.optimizable:
            raise RuntimeError('Rendering parameters are not optimizable')

        num_params = self.num_parameters()
        if values.shape[0] != num_params:
            raise ValueError(f'The size of the input tensor ({values.shape[0]}) does not match '
                             f'the optimizable parameters ({num_params}) in the renderer')

        # Update the parameter values
        pos = 0
        for view in self.parameters(detach=True, flatten=True):
            view.copy_(values.narrow(0, pos, view.shape[0]))
            pos += view.shape[0]

    def to_device(self, device: DeviceType = 'cpu'):
        """Move rendering parameters to a target device (CPU or GPU).

        Args:
            device (DeviceType, optional): Target device ID. Defaults to 'cpu'.
        """
        for param in (self.size, self.light_color, self.camera, self.f0):
            param.to_device(device)

        super().to_device(device)
