import torch as th
import numpy as np

from diffmat.core.base import BaseEvaluableObject
from diffmat.core.types import FloatValue, FloatVector, DeviceType
from diffmat.core.util import to_const, to_tensor
from .patterns import ATOMIC_PATTERNS


class AlternativeRand(BaseEvaluableObject):
    """An alternative random sampler that works independently from the default random seed
    environment.
    """
    def __init__(self, seed: int = 0, device: DeviceType = 'cpu', **kwargs):
        """Initialize the alternative random number generator.
        """
        super().__init__(device=device, **kwargs)

        self.seed = seed

        # Initialize the random number generator state without disturbing the environment
        with self.temp_rng(seed=seed) as state:
            self.rng_state = state

    def evaluate(self, *args, **kwargs) -> th.Tensor:
        """Generate uniform random values in [0, 1), the same as `th.rand`. 
        """
        # Switch the rng state to the internal state
        with self.temp_rng(self.rng_state):
            rands = th.rand(*args, **kwargs)

            # Record the new internal state
            self.rng_state = self.get_rng_state()

        return rands

    rand = evaluate


def get_opacity(roughness: FloatValue, max_depth: int) -> FloatVector:
    """Compute the blending opacity at every depth level of the FX-map graph.
    """
    # Compute the normalized blending opacities
    if to_const(roughness) == 0.0:
        opacity = np.ones(max_depth + 1)
    elif isinstance(roughness, (float, np.ndarray)):
        opacity = np.logspace(0, max_depth, max_depth + 1, base=roughness)
        opacity = opacity / opacity.sum()
    else:
        device = roughness.device
        opacity = roughness ** th.linspace(0, max_depth, max_depth + 1, device=device)
        opacity = opacity / opacity.sum()

    return opacity


def get_pattern_pos(depth: int, switch: int = 7) -> th.Tensor:
    """Compute pattern center positions at an FX-map depth level in rendering order.
    """
    # Process smaller arrays in Numpy
    pos = np.full((1, 2), 0.5, dtype=np.float32)
    offsets = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=np.float32) * 0.25
    for _ in range(min(depth, switch)):
        pos = (np.expand_dims(pos, 1) + offsets).reshape(-1, 2)
        offsets *= 0.5

    # Process larger arrays in PyTorch
    pos = to_tensor(pos)

    if depth > switch:
        offsets = to_tensor(offsets)
        for _ in range(depth - switch):
            pos = (pos.unsqueeze(1) + offsets).view(-1, 2)
            offsets *= 0.5

    return pos


def get_group_pos(scale: int = 1, relative: bool = False) -> th.Tensor:
    """Compute global center positions for noise generator nodes controlled by the 'scale'
    parameter. To accommodate floating-point scale values, an additional weight mask is returned
    to indicate the fraction of contribution from each position.
    """
    # Start from the center
    pos = np.zeros((1, 2), dtype=np.int64)
    direction = np.array([-1, 0])
    length = 1
    rot_matrix = np.array([[0, -1], [1, 0]])

    # Follow the outward spiral
    num_target = (scale + (scale + 1) % 2) ** 2

    while pos.shape[0] < num_target:
        pos_advances = direction * np.expand_dims(np.arange(1, length + 1), 1)
        pos = np.append(pos, pos[-1] + pos_advances, axis=0)
        direction = rot_matrix @ direction
        if not direction[1]:
            length += 1

    # Truncate the pos matrix and normalize to [0, 1]
    pos = pos[:num_target] / scale + 0.5

    # Discard out-of-bound positions
    pos = pos[(pos < 1.0).all(axis=1)]
    if pos.shape[0] != scale ** 2:
        raise RuntimeError('Scaled positions calculation failed. Please check.')

    return to_tensor(pos - 0.5 if relative else pos)


def get_pattern_image(pattern_type: str, res_h: int, res_w: int, mode: str = 'gray',
                      var: FloatValue = 0.0) -> th.Tensor:
    """Generate an image of an atomic pattern.
    """
    if mode not in ('gray', 'color'):
        raise ValueError("The color mode must be either 'color' or 'gray'")

    # Create the pixel coordinates grid
    x_coords = th.linspace(1 / res_h - 1, 1 - 1 / res_h, res_h)
    y_coords = th.linspace(1 / res_w - 1, 1 - 1 / res_w, res_w) if res_w != res_h else x_coords
    grid = th.stack(th.meshgrid(x_coords, y_coords, indexing='xy'), dim=2).unsqueeze(0)

    # Generate the pattern image
    ret = ATOMIC_PATTERNS[pattern_type](grid, var)

    # Obtain the grayscale output which is also an alpha mask for the color output
    alpha = ret[0] if isinstance(ret, tuple) else ret
    if mode == 'gray':
        return alpha.movedim(-1, -3)

    # Construct the color output. The RGB channels are full-white inside the pattern.
    mask = ret[1] if isinstance(ret, tuple) else \
        ((grid > -1) & (grid <= 1)).all(dim=-1, keepdim=True).float()
    output = th.cat((mask.expand(-1, -1, -1, 3), alpha), dim=-1)
    return output.movedim(-1, -3)
