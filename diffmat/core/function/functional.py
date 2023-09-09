from typing import List, Sequence, Union

import torch as th
import math
import random

from diffmat.core.operator import grid_sample
from diffmat.core.types import FloatValue, FloatVector
from diffmat.core.util import to_tensor


def cat_broadcast(tensors: Sequence[th.Tensor], dim: int = 0) -> th.Tensor:
    """An extension of the 'torch.cat' function with shape broadcasting.
    """
    ndims = max(t.ndim for t in tensors)
    hd_tensor = next(t for t in tensors if t.ndim == ndims)
    return th.cat([t.expand_as(hd_tensor) for t in tensors], dim=dim)


def image_sample(img_list: List[th.Tensor], pos: FloatVector, img_index: int,
                 mode: int) -> th.Tensor:
    """Image sampling function tailored to color/grayscale sampling nodes in pixel processors.
    """
    # Retrieve the input image
    img_in = img_list[img_index]
    B = img_in.shape[0]

    # Deduce whether to execute per-pixel
    per_pixel = isinstance(pos, th.Tensor) and pos.ndim > 1 

    # For non-per-pixel functions, the input image must not be batched
    if not per_pixel and B > 1:
        raise ValueError('Non-per-pixel sampling function only allows one image as input')

    # Convert the position to a 4D torch tensor
    img_grid = to_tensor(pos).expand((B, -1, -1, 2) if per_pixel else (B, 1, 1, 2))

    # After sampling, convert the output tensor from BCHW to BHWC, or back to 1D for
    # non-per-pixel functions
    mode_str = 'bilinear' if mode else 'nearest'
    img_out = grid_sample(img_in, img_grid, mode=mode_str, sbs_format=True)
    img_out = img_out.movedim(1, -1) if per_pixel else th.atleast_1d(img_out.squeeze())

    return img_out


def rotate_position(position: FloatVector, angle: FloatValue) -> FloatVector:
    """Non-atomic function: Rotate Position
    """
    R = rotation_matrix(angle)
    return (R.unflatten(-1, (2, 2)) * position.unsqueeze(-2)).sum(dim=-1) \
           if isinstance(position, th.Tensor) \
           else [v1 * v2 for v1, v2 in zip(R, position * 2)]


def rotation_matrix(angle: FloatValue) -> FloatVector:
    """Non-atomic function: Rotation Matrix
    """
    is_tensor = isinstance(angle, th.Tensor)
    cos_angle, sin_angle \
        = (angle.cos(), angle.sin()) if is_tensor else (math.cos(angle), math.sin(angle))
    R = [cos_angle, -sin_angle, sin_angle, cos_angle]
    return th.cat(R, dim=-1) if is_tensor else R


def direction_to_normal(direction: FloatValue, slope_angle: FloatValue,
                        y_up: Union[bool, th.Tensor]) -> FloatVector:
    """Convert direction into a normal vector.
    """
    # Convert turns to radians
    dpi = math.pi * 2.0
    direction, slope_angle = direction * dpi, slope_angle * dpi

    # Calculate normal vector
    y_flip = (th.where(y_up, -1.0, 1.0) if isinstance(y_up, th.Tensor)
              else -1.0 if y_up else 1.0)
    if isinstance(direction, th.Tensor):
        cos_dir, sin_dir = -th.cos(direction), th.sin(direction) * y_flip
    else:
        cos_dir, sin_dir = -math.cos(direction), math.sin(direction) * y_flip

    # Amplify the normal vector using the slope angle
    sin_slope = (th.sin(slope_angle) if isinstance(slope_angle, th.Tensor)
                 else math.sin(slope_angle))
    norm_x = cos_dir * sin_slope * 0.5 + 0.5
    norm_y = sin_dir * sin_slope * 0.5 + 0.5

    # Assemble the final vector
    tensors = [n for n in (norm_x, norm_y) if isinstance(n, th.Tensor)]
    tensors_3d = [n for n in tensors if n.ndim >= 3]

    if tensors_3d:
        ret = th.cat((*(to_tensor(n).expand_as(tensors[0]) for n in (norm_x, norm_y)),
                      th.ones(*tensors[0].shape[:-1], 2)), dim=-1)
    elif tensors:
        ret = th.cat((*(th.atleast_1d(to_tensor(n)) for n in (norm_x, norm_y)),
                      th.ones(2)), dim=-1)
    else:
        ret = [norm_x, norm_y, 1.0, 1.0]

    return ret


def global_random(input: FloatValue, seed: Union[int, th.Tensor]) -> FloatValue:
    """A random number generator that produces consistent results given an input seed.
    """
    r = (1234 * 2 ** 35) / ((seed + 4321) % 65536 + 4321)
    r = ((r % 1e6 - r % 100) * 0.01) ** 2
    r = (r % 1e6 - r % 100) * 1e-6
    return input * r


def rand(per_pixel: bool = False, size: List[int] = []) -> FloatValue:
    """Random number generation function.
    """
    return th.rand(*size, 1) if per_pixel else random.random()


def discrete_ab(a: FloatValue, b: FloatValue, prob: FloatValue, per_pixel: bool = False,
                size: List[int] = []) -> FloatValue:
    """Discrete random number generation function.
    """
    rands = rand(per_pixel=per_pixel, size=size)
    if per_pixel or any(isinstance(v, th.Tensor) for v in (a, b, prob)):
        ret = th.where(th.as_tensor(prob <= rands), a, b)
    else:
        ret = a if prob <= rands else b

    return ret
