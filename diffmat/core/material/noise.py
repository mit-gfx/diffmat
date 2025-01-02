from typing import Union, Optional, Tuple, List, Callable
import math

from torch.nn.functional import grid_sample as grid_sample_impl, affine_grid
import torch as th
import numpy as np
import numpy.typing as npt

from diffmat.core.fxmap import \
    FXMapExecutorV2 as FXE, DenseFXMapExecutorV2 as FXEDense, ChainFXMapComposer as Composer
from diffmat.core.fxmap.util import get_group_pos, get_pattern_pos, get_opacity, ATOMIC_PATTERNS
from diffmat.core.log import get_logger
from diffmat.core.types import FloatValue, FloatVector, DeviceType, FXMapJobArray
from diffmat.core.util import to_tensor, to_numpy, to_const, check_arg_choice
from .functional import blend, d_blur, levels, transform_2d, color_from_rgb, color_to_rgb
from .util import input_check, input_check_all_positional, grayscale_input_check, color_input_check


# Logger for the noise module
logger = get_logger('diffmat.core')


# --------------------------------------------------- #
#          Noise and pattern generator nodes          #
# --------------------------------------------------- #

def crystal_2(res_h: int = 256, res_w: int = 256, scale: int = 1, disorder: float = 0.0,
              device: DeviceType = 'cpu') -> th.Tensor:
    """Noise generator: Crystal 2
    """
    if res_h != res_w:
        raise ValueError('Only square textures are supported for now')

    pattern_size = to_tensor([1.0 / scale, 4.0 / scale])

    # Set up Job array function
    def job_arr_func(depth: int, pos: th.Tensor, branch_offsets: th.Tensor, rands: th.Tensor) -> \
            Tuple[FXMapJobArray, th.Tensor]:

        # Sample random color, offset, size, and rotation variations
        color_rands, branch_offset_rands, pattern_offset_rands, size_rands, rotation_rands = \
            rands.split((1, 2, 2, 2, 1), dim=-1)

        # Calculate pattern offsets
        branch_offsets = branch_offsets + \
            disorder_func(0.125 / scale, disorder, branch_offset_rands)
        pattern_offsets = disorder_func(0.0625 / scale, disorder, pattern_offset_rands)

        # Construct the job array
        job_arr = {
            'color': (color_rands * 2 - 1).flatten(0, -2),
            'offset': (pos + branch_offsets + pattern_offsets).flatten(0, -2),
            'size': ((size_rands + 1) * pattern_size).flatten(0, -2),
            'rotation': (rotation_rands * 0.135 + 0.9325).flatten(0, -2),
            'variation': [0.84],
            'depth': np.array([depth]),
            'blending': [FXE.BLEND_ADD],
        }

        return job_arr, branch_offsets

    # Invoke the chain FX-map composer
    depths = (3, 6)
    rand_sizes = [8] * len(range(*depths))

    composer = Composer(
        int(math.log2(res_h)), 'half_bell', background_color=128/255, roughness=0.18,
        global_opacity=53.81, device=device)
    fx_map = composer.evaluate(
        job_arr_func, depths, rand_sizes, scale=scale, keep_order=False)

    return fx_map


def clouds_2(seed: int = 0, res_h: int = 256, res_w: int = 256, scale: int = 1,
             disorder: float = 0.0, device: DeviceType = 'cpu') -> th.Tensor:
    """Noise generator: Clouds 2
    """
    if res_h != res_w:
        raise ValueError('Only square textures are supported for now')

    # --------------------
    # Low frequency FX-map
    # --------------------

    pattern_color = to_tensor(-1.0)
    pattern_size = to_tensor([5.0 / scale, 1.0 / scale])

    # Job array generation function
    def lf_job_arr_func(depth: int, pos: th.Tensor, branch_offsets: th.Tensor,
                        rands: th.Tensor) -> Tuple[FXMapJobArray, th.Tensor]:

        # Sample random color, offset, size, and rotation variations
        branch_offset_rands, rotation_rands = rands.split((2, 1), dim=-1)

        # Calculate cumulative branch offsets
        branch_offsets = branch_offsets + \
            disorder_func(0.25 / scale, disorder, branch_offset_rands)

        # Construct the job array
        job_arr = {
            'color': pattern_color.view(-1, 1),
            'offset': (pos + branch_offsets).view(-1, 2),
            'size': pattern_size.view(-1, 2),
            'rotation': rotation_rands.view(-1, 1),
            'variation': [0.0],
            'depth': np.array([depth]),
            'blending': [FXE.BLEND_ADD],
        }

        return job_arr, branch_offsets

    # Invoke the chain FX-map composer
    depths = (3, 5)
    rand_sizes = [3] * len(range(*depths))

    composer = Composer(
        int(math.log2(res_h)), 'pyramid', background_color=1.0, roughness=0.66,
        global_opacity=0.44, device=device)
    img_low_freq = composer.evaluate(
        lf_job_arr_func, depths, rand_sizes, scale=scale, keep_order=False)

    # --------------------
    # Mid frequency FX-map
    # --------------------

    # Reset random seed state for alignment of random numbers
    th.manual_seed(seed)

    pattern_color = th.ones([])

    def mf_job_arr_func(depth: int, pos: th.Tensor, branch_offsets: th.Tensor,
                        rands: th.Tensor) -> Tuple[Optional[FXMapJobArray], th.Tensor]:

        # Layer 5 has additional color variation
        branch_offset_rands, rotation_rands = rands.split((2, 1), dim=-1)

        # Calculate cumulative branch offsets
        branch_offsets = branch_offsets + \
            disorder_func(2 ** (1 - depth) / scale, disorder, branch_offset_rands)

        # Construct the job array
        job_arr = {
            'color': pattern_color.view(-1, 1),
            'offset': (pos + branch_offsets).view(-1, 2),
            'size': pattern_size.view(-1, 2),
            'rotation': rotation_rands.view(-1, 1),
            'variation': [0.0],
            'depth': np.array([depth]),
            'blending': [FXE.BLEND_ADD],
        } if depth >= 5 else None

        return job_arr, branch_offsets

    # Invoke the chain FX-map composer
    depths = (3, 7)
    rand_sizes = [3] * len(range(*depths))

    composer = Composer(
        int(math.log2(res_h)), 'pyramid', roughness=0.66, global_opacity=0.44, device=device)
    img_mid_freq = composer.evaluate(
        mf_job_arr_func, depths, rand_sizes, scale=scale, keep_order=False)

    # ------------------------------------------
    # High frequency FX-map (non-differentiable)
    # ------------------------------------------

    # Reset random seed state for alignment of random numbers
    th.manual_seed(seed)

    # Helper funciton for generating random offsets
    def random_offsets(rands: th.Tensor, alpha: float = 1.0) -> th.Tensor:
        radius_rands, angle_rands = rands.split(1, dim=-1)
        angle_rands *= 6.28
        disorder_vecs = th.cat((th.cos(angle_rands), th.sin(angle_rands)), dim=-1)
        return radius_rands * disorder_vecs * alpha

    def hf_job_arr_func(depth: int, pos: th.Tensor, branch_offsets: th.Tensor,
                        rands: th.Tensor) -> Tuple[Optional[FXMapJobArray], th.Tensor]:

        # Layer 5 has additional color variation
        branch_offset_rands, rotation_rands = rands.split((2, 1), dim=-1)

        # Calculate cumulative branch offsets
        branch_offsets = branch_offsets + \
            random_offsets(branch_offset_rands, 2 ** (1 - depth) / scale)

        # Construct the job array
        job_arr = {
            'color': pattern_color.view(-1, 1),
            'offset': (pos + branch_offsets).view(-1, 2),
            'size': pattern_size.view(-1, 2),
            'rotation': rotation_rands.view(-1, 1),
            'variation': [0.0],
            'depth': np.array([depth]),
            'blending': [FXE.BLEND_ADD],
        } if depth >= 7 else None

        return job_arr, branch_offsets

    # Initiate the chain FX-map composer
    depths = (3, 9)
    rand_sizes = [3] * len(range(*depths))
    conns = [0xf] * (len(range(*depths)) - 2) + [0x5]

    img_high_freq = composer.evaluate(
        hf_job_arr_func, depths, rand_sizes, conns=conns, scale=scale, keep_order=False)

    # Final composition
    img_out = (img_low_freq - img_mid_freq).clamp_min(0.0)
    img_out = (img_out - img_high_freq).clamp_min(1 / 3)
    img_out = (1 - 1.5 * (1 - img_out)).clamp(0.0, 1.0)

    return img_out


def bnw_spots_3(seed: int = 0, res_h: int = 256, res_w: int = 256, scale: int = 1,
                disorder: float = 0.0, device: DeviceType = 'cpu') -> th.Tensor:
    """Noise generator: BnW Spots 3
    """
    if res_h != res_w:
        raise ValueError('Only square textures are supported for now')

    # --------------------
    # Low frequency FX-map
    # --------------------

    pattern_size_y = to_tensor(2.0)

    def lf_job_arr_func(depth: int, pos: th.Tensor, branch_offsets: th.Tensor,
                        rands: th.Tensor) -> Tuple[FXMapJobArray, th.Tensor]:

        # Sample random pattern variations
        color_rands, offset_rands, size_rands, rotation_rands = \
            rands.split((1, depth - 1, 1, 1), dim=-1)

        if depth == 2:
            angle_rands = offset_rands * 6.28
            disorder_vecs = th.cat((th.cos(angle_rands), th.sin(angle_rands)), dim=-1)
            disorder_offsets = 2 ** -6 / scale * disorder_vecs
        else:
            disorder_offsets = disorder_func(2 ** -6 / scale, disorder, offset_rands)

        pattern_pos = pos + branch_offsets + disorder_offsets
        pattern_sizes = th.cat((size_rands + 1, pattern_size_y.expand_as(size_rands)), dim=-1)
        pattern_sizes = pattern_sizes / scale

        # Construct the job array
        job_arr = {
            'color': ((color_rands * 2 - 1) * (0.75 - 0.15 * (depth - 2))).view(-1, 1),
            'offset': pattern_pos.view(-1, 2),
            'size': pattern_sizes.view(-1, 2),
            'rotation': rotation_rands.view(-1, 1),
            'variation': [0.0],
            'depth': np.array([depth]),
            'blending': [FXE.BLEND_ADD],
        }

        return job_arr, branch_offsets

    # Invoke the chain FX-map composer
    depths = (2, 4)
    rand_sizes = [4, 5]

    composer = Composer(
        int(math.log2(res_h)), 'paraboloid', background_color=128/255, roughness=1.0,
        global_opacity=0.9, device=device)
    img_low_freq = composer.evaluate(
        lf_job_arr_func, depths, rand_sizes, scale=scale, keep_order=False)

    # --------------------
    # Mid frequency FX-map
    # --------------------

    # Reset random number generator state
    th.manual_seed(seed)

    def mf_job_arr_func(depth: int, pos: th.Tensor, branch_offsets: th.Tensor,
                        rands: th.Tensor) -> Tuple[FXMapJobArray, th.Tensor]:

        # Sample random pattern variations
        color_rands, offset_rands, size_rands, rotation_rands = rands.split((1, 2, 1, 1), dim=-1)
        branch_offsets = branch_offsets + disorder_func(2 ** -6 / scale, disorder, offset_rands)
        pattern_sizes = th.cat((size_rands + 1, pattern_size_y.expand_as(size_rands)), dim=-1)
        pattern_sizes = pattern_sizes / scale

        # Construct the job array
        job_arr = {
            'color': (color_rands * 2 - 1).view(-1, 1),
            'offset': (pos + branch_offsets).view(-1, 2),
            'size': pattern_sizes.view(-1, 2),
            'rotation': rotation_rands.view(-1, 1),
            'variation': [0.0],
            'depth': np.array([depth]),
            'blending': [FXE.BLEND_ADD],
        }

        return job_arr, branch_offsets

    # Invoke the chain FX-map composer
    depths = (4, 7)
    rand_sizes = [5] * len(range(*depths))

    composer = Composer(
        int(math.log2(res_h)), 'paraboloid', background_color=128/255, roughness=1.0,
        global_opacity=0.85, device=device)
    img_mid_freq = composer.evaluate(
        mf_job_arr_func, depths, rand_sizes, scale=scale, keep_order=False)

    # ------------------------------------------
    # High frequency FX-map (non-differentiable)
    # ------------------------------------------

    # Reset random number generator state
    th.manual_seed(seed)

    pattern_size_y = th.ones([])

    def hf_job_arr_func(depth: int, pos: th.Tensor, branch_offsets: th.Tensor,
                        rands: th.Tensor) -> Tuple[FXMapJobArray, th.Tensor]:

        # Sample random pattern variations
        color_rands, offset_rands, size_rands, rotation_rands = rands.split((1, 2, 1, 1), dim=-1)
        branch_offsets = branch_offsets + \
            disorder_func(2 ** (-6 if depth < 6 else -4) / scale, disorder, offset_rands)
        pattern_sizes = th.cat((size_rands + 1, pattern_size_y.expand_as(size_rands)), dim=-1)
        pattern_sizes = pattern_sizes / scale

        # Patterns starts at Layer 6
        job_arr = {
            'color': (color_rands * 2 - 1).view(-1, 1),
            'offset': (pos + branch_offsets).view(-1, 2),
            'size': pattern_sizes.view(-1, 2),
            'rotation': rotation_rands.view(-1, 1),
            'variation': [0.0],
            'depth': np.array([depth]),
            'blending': [FXE.BLEND_ADD],
        } if depth >= 6 else None

        return job_arr, branch_offsets

    # Invoke the chain FX-map composer
    depths = (1, 10)
    rand_sizes = [5] * len(range(*depths))

    composer = Composer(
        int(math.log2(res_h)), 'paraboloid', background_color=128/255, roughness=1.0,
        global_opacity=0.9, device=device)
    img_high_freq = composer.evaluate(
        hf_job_arr_func, depths, rand_sizes, scale=scale, keep_order=False)

    # Final composition
    img_out = blend(img_mid_freq, img_low_freq, blending_mode='add_sub', opacity=0.5)
    img_out = 0.13601 + (0.861399 - 0.13601) * img_out
    img_out = blend(img_high_freq, img_out, blending_mode='add_sub', opacity=0.35)

    return img_out


def perlin_noise(res_h: int = 256, res_w: int = 256, scale: int = 32, disorder: float = 0.0,
                 device: DeviceType = 'cpu') -> th.Tensor:
    """Noise generator: Perlin noise
    """
    if res_h != res_w:
        raise ValueError('Only square textures are supported for now')

    # Return a gray image with non-positive scale
    if scale <= 0:
        return th.full((1, 1, res_h, res_w), 0.5)

    # -----------------
    # Background FX-map
    # -----------------

    pattern_color = th.ones(1)

    # Calculate pattern transformations
    num_patterns = scale ** 2
    gaussian_size = to_tensor([2.82 / scale, 2.82 / scale])
    gaussian_offset = 0.125 / scale

    angle_rands = th.rand(num_patterns, 1) + disorder
    angle_rands_rad = angle_rands * (math.pi * 2)
    offset_vecs = th.cat((th.cos(angle_rands_rad), th.sin(angle_rands_rad)), dim=1)
    offset_vecs = offset_vecs * gaussian_offset
    scaled_pos = get_group_pos(scale)

    # Assemble the job array
    job_arr = {
        'color': pattern_color.view(-1, 1),
        'offset': scaled_pos + offset_vecs,
        'size': gaussian_size.view(-1, 2),
        'rotation': angle_rands,
        'variation': [0.0],
        'depth': [0],
        'blending': [FXE.BLEND_ADD],
    }

    # Initiate the FX-map executor
    executor = FXEDense(int(math.log2(res_h)), keep_order=False, device=device)
    blending_opacity = get_opacity(0.0, 0) * 0.44
    img_bg = executor.evaluate(blending_opacity, batched_jobs={'bell': job_arr})

    # -----------------
    # Foreground FX-map
    # -----------------

    # Apply negated offset vectors
    job_arr['offset'] = scaled_pos - offset_vecs

    # Initiate the FX-map executor
    executor = FXEDense(int(math.log2(res_h)), keep_order=False, device=device)
    blending_opacity = get_opacity(0.0, 0) * 0.44
    img_fg = executor.evaluate(blending_opacity, batched_jobs={'bell': job_arr})

    # Final composition
    img_out = ((img_bg - img_fg + 1) * 0.5).clamp(0.0, 1.0)
    img_out = levels(img_out, in_low=0.25, in_high=0.75)

    return img_out


def gradient_linear_1(res_h: int = 256, res_w: int = 256, tiling: int = 1,
                      rotation: int = 0) -> th.Tensor:
    """Pattern generator: gradient linear 1
    """
    if res_h != res_w:
        raise ValueError('Only square textures are supported for now')

    # Determine which direction the gradient goes
    rotation = rotation // 90 % 4
    dim = 1 - rotation % 2
    res = res_h if dim else res_w

    # Compute the linear gradient field
    L, R = 1 / res - 1, 1 - 1 / res
    coords = th.linspace(R, L, res) if rotation in (0, 3) else th.linspace(L, R, res)
    grid = coords.unsqueeze(1).expand(-1, res_w) if dim else coords.expand(res_h, -1)
    gradient = (grid * max(tiling, 1e-8) + 1) % 2 * 0.5

    return gradient.expand(1, 1, -1, -1)


def gradient_linear_2(res_h: int = 256, res_w: int = 256, tiling: int = 1,
                      rotation: int = 0) -> th.Tensor:
    """Pattern generator: gradient linear 2
    """
    if res_h != res_w:
        raise ValueError('Only square textures are supported for now')

    # Determine which direction the gradient goes
    dim = 1 - rotation // 90 % 2
    res = res_h if dim else res_w

    # Compute the linear gradient field
    coords = th.linspace(1 / res - 1, 1 - 1 / res, res)
    grid = coords.unsqueeze(1).expand(-1, res_w) if dim else coords.expand(res_h, -1)
    gradient = 1 - ((grid * max(tiling, 1e-8) + 1) % 2 - 1) ** 2

    return gradient.expand(1, 1, -1, -1)


def gradient_linear_3(res_h: int = 256, res_w: int = 256, tiling: int = 1,
                      position: FloatValue = 0.5, rotation: int = 0) -> th.Tensor:
    """Pattern generator: gradient linear 3
    """
    if res_h != res_w:
        raise ValueError('Only square textures are supported for now')

    # Determine which direction the gradient goes
    dim = 1 - rotation // 90 % 2
    res = res_h if dim else res_w

    # Compute the linear gradient fields
    coords = th.linspace(1 / res - 1, 1 - 1 / res, res)
    grid = coords.unsqueeze(1).expand(-1, res_w) if dim else coords.expand(res_h, -1)
    grad_pos = (grid * tiling + 1) % 2 * 0.5
    grad_neg = 1 - grad_pos

    # Normalize the gradient fields using the position parameter and combine them into the final
    # result
    grad_pos = grad_pos.clamp_max(position + 1e-8) / position.clamp_min(1e-8)
    grad_neg = grad_neg.clamp_max(1 - position + 1e-8) / (1 - position).clamp_min(1e-8)
    gradient = th.min(grad_pos, grad_neg).clamp(0.0, 1.0)

    return gradient.expand(1, 1, -1, -1)


@input_check_all_positional(channel_specs='-')
def tile_generator(img_bg: Optional[th.Tensor], *img_list: Optional[th.Tensor], mode: str = 'gray',
                   res_h: int = 256, res_w: int = 256, x_num: int = 10, y_num: int = 10,
                   pattern: str = 'brick', input_number: int = 1,
                   input_distribution: str = 'random', pattern_specific: FloatVector = [0.2, 0.0],
                   input_filter_mode: str = 'bilinear_mipmap', fixed_rotation: int = 0,
                   fixed_rotation_random: float = 0.0, quincunx_flip: bool = False,
                   symmetry_random: float = 0.0, symmetry_random_mode: str = 'both',
                   size_mode: str = 'interstice', middle_size: FloatVector = [0.5, 0.5],
                   interstice: FloatVector = [0.0, 0.0, 0.0, 0.0], size: FloatVector = [1.0, 1.0],
                   size_absolute: FloatVector = [0.1, 0.1], size_pixel: FloatVector = [1.0, 1.0],
                   size_random: FloatVector = [0.0, 0.0], scale: FloatValue = 1.0,
                   scale_random: FloatValue = 0.0, scale_random_seed: int = 0,
                   offset: FloatValue = 0.0, offset_random: FloatValue = 0.0,
                   offset_random_seed: int = 0, vertical_offset: bool = False,
                   position_random: FloatVector = [0.0, 0.0],
                   global_offset: FloatVector = [0.0, 0.0], rotation: FloatValue = 0.0,
                   rotation_random: FloatValue = 0.0, color: Union[FloatValue, FloatVector] = 1.0,
                   color_random: Union[FloatValue, FloatVector] = 0.0,
                   color_by_number: bool = False, color_by_scale: bool = False,
                   checker_mask: bool = False, horizontal_mask: bool = False,
                   vertical_mask: bool = False, random_mask: float = 0.0,
                   invert_mask: bool = False, blending_mode: str = 'add',
                   background_color: Union[FloatValue, FloatVector] = 0.0,
                   global_opacity: FloatValue = 1.0, reverse_order: bool = False,
                   device: DeviceType = 'cpu') -> th.Tensor:
    """Pattern generator: tile generator
    """
    check_arg_choice(mode, ('gray', 'color'), 'mode')

    if res_h != res_w:
        raise ValueError('Only square textures are supported for now')
    if input_number not in range(1, 7):
        raise ValueError('The number of input images must be from 1 to 6')
    if len(img_list) < input_number:
        raise ValueError(f'Expect {input_number} images but only {len(img_list)} provided')

    # Check input image format
    is_color = mode == 'color'
    next_image = next((img for img in [img_bg, *img_list] if img is not None), None)
    if next_image is not None:
        img_check = color_input_check if is_color else grayscale_input_check
        img_check(next_image, 'input_image')

    # Initiate FX-map executor
    executor = FXE(int(math.log2(res_h)), device=device)
    executor.reset(img_bg, *img_list, mode=mode, background_color=background_color)

    # Quantize X and Y amount
    # Only return the FX-map canvas (already applied with background color and image) if there is
    # no pattern to render
    if min(x_num, y_num) <= 0:
        return executor.canvas

    # Random number pools (the pool must be triangular so that changes in the number of tiles won't
    # affect the actual random number sequence
    cd = 4 if is_color else 1
    rands, rands_np = gen_rand_pool(x_num, y_num, 15 + cd).flatten(0, 1).split((11 + cd, 4), dim=1)
    rands_np = to_numpy(rands_np)

    size_rands, offset_rands, rotation_rands, color_rands, pattern_rands \
        = rands.split((7, 2, 1, cd, 1), dim=1)
    fixed_rot_rands, fixed_rot_rands_2, mask_rands, image_rands = np.split(rands_np, 4, axis=1)

    # Size
    interstice_rands, scale_rands, global_scale_rands, symmetry_rands \
        = size_rands.split((2, 2, 1, 2), dim=1)
    one = th.ones([])

    # Pre-compute X and Y tile indices
    nums, num_patterns = to_tensor([x_num, y_num]), x_num * y_num
    x_indices_np, y_indices_np = np.mgrid[:x_num, :y_num]
    indices = to_tensor(np.stack((x_indices_np, y_indices_np), axis=2))

    ## Mode-specific base sizes before scaling
    if size_mode in ('interstice', 'scale'):
        sizes = to_tensor([1 / x_num, 1 / y_num])
    elif size_mode == 'scale_square':
        sizes = to_tensor([1 / max(x_num, y_num)] * 2)
    elif size_mode in ('absolute', 'pixel'):
        sizes = to_tensor(size_absolute if size_mode == 'absolute' else size_pixel)
    else:
        raise ValueError(f'Unknown size mode: {size_mode}')

    ## Middle size
    if size_mode in ('interstice', 'scale', 'scale_square'):
        mid_scales, mid_pos = tile_middle_size(x_num, y_num, middle_size)
        sizes = sizes * mid_scales.view(-1, 2)

    ## Mode-specific size scaling
    if size_mode == 'interstice':
        inters = to_tensor(interstice)
        inters_size, inters_random = inters[::2], inters[1::2]
        inters_scales = th.lerp(one, interstice_rands, inters_random)
        sizes = sizes * (1 - inters_scales * inters_size)

    else:
        if size_mode in ('scale', 'scale_square'):
            sizes = sizes * to_tensor(size)
        size_scales = th.lerp(one, scale_rands, size_random)
        sizes = sizes * size_scales

    ## Final global scaling
    global_scale_rands = global_scale_rands * (1 + scale_random_seed) % 1
    scales_random = th.lerp(one, global_scale_rands, scale_random)
    sizes = sizes * to_tensor(scale) * scales_random

    ## Symmetry flip
    flips = th.where(symmetry_rands >= symmetry_random * 0.5, one, -one)
    if symmetry_random_mode == 'both':
        sizes = sizes * flips
    elif symmetry_random_mode == 'horizontal':
        sizes.select(1, 0).mul_(flips.select(1, 0))
    elif symmetry_random_mode == 'vertical':
        sizes.select(1, 1).mul_(flips.select(1, 1))
    else:
        raise ValueError(f'Unknown symmetry random mode: {symmetry_random_mode}')

    ## For pixel mode, round up to the next integer value
    if size_mode == 'pixel':
        sizes = sizes.ceil() / to_tensor([res_w, res_h])

    # Position
    ## Base position based on middle size
    if size_mode in ('interstice', 'scale', 'scale_square'):
        pos = mid_pos.unflatten(0, (x_num, y_num))
    else:
        pos = (indices + 0.5) / nums

    ## Row/column-spcific offset
    x_indices, y_indices = indices.unbind(dim=2)

    if not vertical_offset:
        x_offsets = y_indices * offset / x_num
        x_offset_rands = to_tensor(global_random(y_indices_np + offset_random_seed))
        pos[..., 0] += x_offsets + x_offset_rands * offset_random * (y_indices + 1) / y_num

    else:
        y_offsets = x_indices * offset / y_num
        y_offset_rands = to_tensor(global_random(x_indices_np + offset_random_seed))
        pos[..., 1] += y_offsets + y_offset_rands * offset_random * (x_indices + 1) / x_num

    ## Global offset
    pos = pos.reshape(-1, 2)
    pos = th.flipud(pos) if reverse_order else pos
    pos = pos + (offset_rands * 2 - 1) * to_tensor(position_random) / nums
    pos = pos + to_tensor(global_offset)

    # Rotation
    ## Fixed rotation [0, 1.5]
    rotations_np: npt.NDArray[np.float32] \
        = np.where(fixed_rot_rands < fixed_rotation_random, fixed_rot_rands_2 * 4, 0.0)
    rotations_np = rotations_np.astype(np.float32) + fixed_rotation / 90
    rotations_np = (np.floor(rotations_np) * 0.25).reshape(x_num, y_num)

    ## Quincunx flip
    if quincunx_flip:
        rotations_np[::2, 1::2] += 0.25
        rotations_np[1::2, ::2] -= 0.25
        rotations_np[1::2, 1::2] += 0.5
    rotations_np = rotations_np.reshape(-1, 1)

    ## Flip the size of 90/270 fixed-rotated patterns
    flip_mask = rotations_np % 0.5 > 0
    sizes = th.where(th.as_tensor(flip_mask), th.fliplr(sizes), sizes)

    ## Continuous rotation
    rotations = to_tensor(rotations_np)
    rotations = rotations + rotation + (rotation_rands * 2 - 1) * rotation_random

    # Color
    colors = to_tensor(color)

    ## Luminance parameterization function
    def color_amp(lums: th.Tensor) -> th.Tensor:
        if color_by_number:
            lums = lums * th.linspace(1 / num_patterns, 1.0, num_patterns).unsqueeze(1)
        if color_by_scale:
            lums = lums * scales_random.clamp(0.0, 1.0)
        return lums

    ## Color randomness
    if is_color:
        rgb, a = colors.split((3, 1))
        h, s, l = color_from_rgb(rgb).unbind()
        hv, sv, lv, av = to_tensor(color_random).unbind()
        hr, sr, lr, ar = color_rands.split(1, dim=1)
        hsl = th.cat((h + hv * (hr - 0.5), th.lerp(s, sr, sv),
                      th.lerp(color_amp(l), lr, lv)), dim=1)
        colors = th.cat((color_to_rgb(hsl, dim=1), a * th.lerp(one, ar, av)), dim=1)
    else:
        colors = th.lerp(color_amp(colors), color_rands, color_random)

    # Mask
    x_par, y_par = (x_indices_np & 1).astype(bool), (y_indices_np & 1).astype(bool)

    mask_np = np.ones_like(x_par)
    mask_np = mask_np & (x_par == y_par) if checker_mask else mask_np
    mask_np = mask_np & ~y_par if horizontal_mask else mask_np
    mask_np = mask_np & x_par if vertical_mask else mask_np
    mask_np = mask_np & (mask_rands >= to_const(random_mask)).reshape(x_num, y_num)
    mask_np = np.ravel(~mask_np if invert_mask else mask_np)
    mask = th.as_tensor(mask_np)
    colors = colors * mask.unsqueeze(1)

    # Pattern
    ## Variation
    variable_patterns = ('brick', 'capsule', 'crescent', 'gradation', 'gradation_offset',
                         'half_bell', 'waves')
    enable_variation = pattern in variable_patterns
    if enable_variation:
        var, var_random = to_tensor(pattern_specific).unbind()
        variations = var * th.lerp(one, pattern_rands, var_random)
    else:
        variations = [0.0]

    ## Image index
    is_image = pattern == 'image'

    if not is_image:
        image_indices: List[str] = [0]
    elif input_distribution == 'random':
        image_indices: npt.NDArray[np.int64] = (image_rands * input_number).astype(np.int64)
        image_indices = image_indices.ravel()
    elif input_distribution == 'cycle':
        image_indices: npt.NDArray[np.int64] = np.arange(num_patterns) % input_number
    else:
        raise ValueError(f'Unknown input distribution mode: {input_distribution}')

    ## Blending mode
    ## For `add_sub` blending mode and image patterns, all empty inputs are considered as full
    ## zeroes
    enable_mask = blending_mode != 'add_sub'
    if not is_color:
        colors = colors - background_color
        if enable_mask:
            colors = colors.clamp(0.0, 1.0)
        elif is_image and any(img is None for img in img_list[:input_number]):
            img_zeros = th.zeros(1, cd, res_h, res_w)
            img_list = [img if img is not None else img_zeros for img in img_list]

    # Assemble job array
    job_arr: FXMapJobArray = {
        'color': colors[mask].view(-1, cd) if enable_mask else colors,
        'offset': pos[mask].view(-1, 2) if enable_mask else pos,
        'size': sizes[mask].view(-1, 2) if enable_mask else sizes,
        'rotation': rotations[mask].view(-1, 1) if enable_mask else rotations,
        'variation': variations[mask].view(-1, 1) if enable_variation and enable_mask \
                     else variations,
        'depth': [0],
        'blending': [FXE.BLEND_DICT['max' if blending_mode == 'max' else 'add']],
        'filtering': [FXE.FILTER_DICT[input_filter_mode]],
        'image_index': image_indices[mask_np].tolist() if is_image and enable_mask \
                       else to_const(image_indices)
    }

    # Execute FX-map
    blending_opacity = th.atleast_1d(to_tensor(global_opacity))
    fx_map = executor.evaluate(blending_opacity, batched_jobs={pattern: job_arr})

    return fx_map


@input_check_all_positional(channel_specs='c')
def tile_generator_color(img_bg: Optional[th.Tensor], *img_list: Optional[th.Tensor],
                         res_h: int = 256, res_w: int = 256, x_num: int = 10, y_num: int = 10,
                         pattern: str = 'brick', input_number: int = 1,
                         input_distribution: str = 'random',
                         pattern_specific: FloatVector = [0.2, 0.0],
                         input_filter_mode: str = 'bilinear_mipmap', fixed_rotation: int = 0,
                         fixed_rotation_random: float = 0.0, quincunx_flip: bool = False,
                         symmetry_random: float = 0.0, symmetry_random_mode: str = 'both',
                         size_mode: str = 'interstice', middle_size: FloatVector = [0.5, 0.5],
                         interstice: FloatVector = [0.0, 0.0, 0.0, 0.0],
                         size: FloatVector = [1.0, 1.0], size_absolute: FloatVector = [0.1, 0.1],
                         size_pixel: FloatVector = [1.0, 1.0],
                         size_random: FloatVector = [0.0, 0.0], scale: FloatValue = 1.0,
                         scale_random: FloatValue = 0.0, scale_random_seed: int = 0,
                         offset: FloatValue = 0.0, offset_random: FloatValue = 0.0,
                         offset_random_seed: int = 0, vertical_offset: bool = False,
                         position_random: FloatVector = [0.0, 0.0],
                         global_offset: FloatVector = [0.0, 0.0], rotation: FloatValue = 0.0,
                         rotation_random: FloatValue = 0.0,
                         color: FloatVector = [1.0, 1.0, 1.0, 1.0],
                         color_random: FloatVector = [0.0, 0.0, 0.0, 0.0],
                         color_by_number: bool = False, color_by_scale: bool = False,
                         checker_mask: bool = False, horizontal_mask: bool = False,
                         vertical_mask: bool = False, random_mask: float = 0.0,
                         invert_mask: bool = False, blending_mode: str = 'add',
                         background_color: FloatVector = [0.0, 0.0, 0.0, 0.0],
                         global_opacity: FloatValue = 1.0, reverse_order: bool = False,
                         device: DeviceType = 'cpu') -> th.Tensor:
    """Pattern generator: tile generator color
    """
    return tile_generator(img_bg, *img_list, mode='color', res_h=res_h, res_w=res_w, x_num=x_num,
                          y_num=y_num, pattern=pattern, input_number=input_number,
                          input_distribution=input_distribution, pattern_specific=pattern_specific,
                          input_filter_mode=input_filter_mode, fixed_rotation=fixed_rotation,
                          fixed_rotation_random=fixed_rotation_random, quincunx_flip=quincunx_flip,
                          symmetry_random=symmetry_random,
                          symmetry_random_mode=symmetry_random_mode, size_mode=size_mode,
                          middle_size=middle_size, interstice=interstice, size=size,
                          size_absolute=size_absolute, size_pixel=size_pixel,
                          size_random=size_random, scale=scale, scale_random=scale_random,
                          scale_random_seed=scale_random_seed, offset=offset,
                          offset_random=offset_random, offset_random_seed=offset_random_seed,
                          vertical_offset=vertical_offset, position_random=position_random,
                          global_offset=global_offset, rotation=rotation,
                          rotation_random=rotation_random, color=color, color_random=color_random,
                          color_by_number=color_by_number, color_by_scale=color_by_scale,
                          checker_mask=checker_mask, horizontal_mask=horizontal_mask,
                          vertical_mask=vertical_mask, random_mask=random_mask,
                          invert_mask=invert_mask, blending_mode=blending_mode,
                          background_color=background_color, global_opacity=global_opacity,
                          reverse_order=reverse_order, device=device)


@input_check(1)
def splatter(img_in: th.Tensor, mode: str = 'gray', pattern_width: FloatValue = 100.0,
             pattern_height: FloatValue = 100.0, rotation: FloatValue = 0.0,
             rotation_variation: FloatValue = 0.0, opacity: FloatValue = 1.0,
             offset_x: FloatValue = 0.0, offset_y: FloatValue = 0.0, disorder: FloatValue = 0.0,
             octave: int = 4, disorder_angle: FloatValue = 0.0, disorder_random: bool = False,
             size_variation: FloatValue = 0.0, filter_mode: str = 'bilinear_mipmap',
             output_min: FloatValue = 0.0, output_max: FloatValue = 1.0,
             background_color: Union[FloatValue, FloatVector] = 0.0,
             color_variation: FloatValue = 0.0, device: DeviceType = 'cpu') -> th.Tensor:
    """Pattern generator: splatter
    """
    check_arg_choice(mode, ('gray', 'color'), 'mode')

    # Input validity check
    res_h, res_w = img_in.shape[2], img_in.shape[3]
    if res_h != res_w:
        raise ValueError('Only square textures are supported for now')

    # Input image format check
    is_color = mode == 'color'
    img_check = color_input_check if is_color else grayscale_input_check
    img_check(img_in, 'input image')

    # Random number pool
    num_patterns, cd = 1 << (octave << 1), 3 if is_color else 1
    rands = th.rand(num_patterns, 5 + cd)
    color_rands, offset_rands, size_rands, rotation_rands = rands.split((cd, 2, 2, 1), dim=1)

    # Color
    colors = 1 - to_tensor(color_variation) * color_rands
    if is_color:
        colors = th.cat((colors, th.ones_like(colors.narrow(1, 0, 1))), dim=1)

    # Position
    pos = get_pattern_pos(octave) + th.stack((to_tensor(offset_x), to_tensor(offset_y))) * 0.01

    ## Position randomness (disorder)
    pos_rands, angle_rands = offset_rands.split(1, dim=1)
    disorder_radii = disorder * 1e-3 * pos_rands
    disorder_angles = th.atleast_2d(to_tensor(disorder_angle / 58))
    disorder_angles = disorder_angles * angle_rands if disorder_random else disorder_angles
    pos = pos + disorder_radii * th.cat((th.cos(disorder_angles), th.sin(disorder_angles)), dim=1)

    # Size
    sizes = th.stack((to_tensor(pattern_width), to_tensor(pattern_height)))
    sizes = sizes + (size_rands.narrow(1, 0, 1) - size_rands.narrow(1, 1, 1)) * size_variation
    sizes = sizes * 0.01

    # Rotation
    rotations = (rotation + rotation_rands * rotation_variation) / 360

    # Assemble job array
    job_arr: FXMapJobArray = dict(
        color=colors, offset=pos, size=sizes, rotation=rotations, variation=[0.0], depth=[octave],
        blending=[FXE.BLEND_MAX_COPY], filtering=[FXE.FILTER_DICT[filter_mode]], image_index=[0]
    )

    # Initiate FX-map executor
    executor = FXE(int(math.log2(res_h)), device=device)
    executor.reset(None, img_in, mode=mode, background_color=background_color)
    blending_opacity = to_tensor(opacity).expand(octave + 1)
    fx_map = executor.evaluate(blending_opacity, batched_jobs={'image': job_arr})

    # Output adjustment
    fx_map = levels(fx_map, in_low=output_min, in_high=output_max)

    return fx_map


@input_check(2, channel_specs='cg')
def splatter_color(img_rgb: th.Tensor, img_a: th.Tensor,
                   pattern_width: FloatValue = 100.0, pattern_height: FloatValue = 100.0,
                   rotation: FloatValue = 0.0, rotation_variation: FloatValue = 0.0,
                   opacity: FloatValue = 1.0, offset_x: FloatValue = 0.0,
                   offset_y: FloatValue = 0.0, disorder: FloatValue = 0.0, octave: int = 4,
                   disorder_angle: FloatValue = 0.0, disorder_random: bool = False,
                   size_variation: FloatValue = 0.0, filter_mode: str = 'bilinear_mipmap',
                   output_min: FloatValue = 0.0, output_max: FloatValue = 1.0,
                   background_color: FloatVector = [0.0, 0.0, 0.0, 1.0],
                   color_variation: FloatValue = 0.0, device: DeviceType = 'cpu') -> th.Tensor:
    """Pattern generator: splatter color
    """
    # Assemble image input for splatter function
    img_in = th.cat((img_rgb.narrow(1, 0, 3), img_a), dim=1)

    # Call the splatter function
    img_splatter = splatter(
        img_in, mode='color', pattern_width=pattern_width, pattern_height=pattern_height,
        rotation=rotation, rotation_variation=rotation_variation, opacity=opacity,
        offset_x=offset_x, offset_y=offset_y, disorder=disorder, octave=octave,
        disorder_angle=disorder_angle, disorder_random=disorder_random,
        size_variation=size_variation, filter_mode=filter_mode, output_min=output_min,
        output_max=output_max, background_color=background_color, color_variation=color_variation,
        device=device)

    return img_splatter


@input_check_all_positional(channel_specs='g')
def splatter_circular(img_bg: Optional[th.Tensor], *img_list: Optional[th.Tensor],
                      res_h: int = 256, res_w: int = 256, pattern_num: int = 10,
                      pattern_num_random: FloatValue = 0.0, pattern_num_min: int = 1,
                      ring_num: int = 1, pattern: str = 'paraboloid', input_number: int = 1,
                      input_distribution: str = 'random',
                      input_filter_mode: str = 'bilinear_mipmap',
                      pattern_specific: FloatValue = 0.0, symmetry_random: float = 0.0,
                      symmetry_random_mode: str = 'both', radius: FloatValue = 0.25,
                      radius_random: FloatValue = 0.0, radius_multiplier: FloatValue = 1.0,
                      angle_random: FloatValue = 0.0, spiral_factor: FloatValue = 0.0,
                      spread: FloatValue = 1.0, directional_offset: FloatValue = 0.0,
                      global_offset: FloatVector = [0.0, 0.0], connect_patterns: bool = False,
                      size_connected: FloatVector = [1.0, 0.1], size: FloatVector = [0.1, 0.1],
                      size_random: FloatVector = [0.0, 0.0], scale: FloatValue = 1.0,
                      scale_random: FloatValue = 0.0, scale_by_pattern_number: FloatValue = 0.0,
                      scale_by_pattern_number_invert: bool = False,
                      scale_by_ring_number: FloatValue = 0.0,
                      scale_by_ring_number_invert: bool = False,
                      pattern_rotation: FloatValue = 0.0,
                      pattern_rotation_random: FloatValue = 0.0,
                      pattern_rotation_pivot: str = 'center', center_orientation: bool = True,
                      ring_rotation: FloatValue = 0.0, ring_rotation_random: FloatValue = 0.0,
                      ring_rotation_offset: FloatValue = 0.0, color: FloatValue = 1.0,
                      color_random: FloatValue = 0.0, color_by_scale: FloatValue = 0.0,
                      color_by_pattern_number: FloatValue = 0.0,
                      color_by_pattern_number_invert: bool = False,
                      color_by_ring_number: FloatValue = 0.0,
                      color_by_ring_number_invert: bool = False, random_mask: float = 0.0,
                      background_color: FloatValue = 0.0, blending_mode: str = 'add',
                      global_opacity: FloatValue = 1.0, device: DeviceType = 'cpu') -> th.Tensor:
    """Pattern generator: splatter circular
    """
    if res_h != res_w:
        raise ValueError('Only square textures are supported for now')
    if input_number not in range(1, 7):
        raise ValueError('The number of input images must be from 1 to 6')
    if len(img_list) < input_number:
        raise ValueError(f'Expect {input_number} images but only {len(img_list)} provided')

    # Helper function for polar to cartesian coordinates conversion
    def cartesian(radius: th.Tensor, angle: th.Tensor, add_90_deg: bool = False) -> th.Tensor:
        return radius * th.cat((-angle.sin(), angle.cos()), dim=-1) if add_90_deg else \
               radius * th.cat((angle.cos(), angle.sin()), dim=-1)

    # Helper function for converting turning numbers to radians
    def _rad(a: th.Tensor) -> th.Tensor:
        return a * (math.pi * 2)

    # Initiate FX-map executor
    executor = FXE(int(math.log2(res_h)), device=device)
    executor.reset(img_bg, *img_list, background_color=background_color)

    # Return the FX-map canvas (already applied with background color and image) if there is no
    # pattern to render
    if ring_num <= 0:
        return executor.canvas

    # Ring-level random pool
    ring_radius_rands, pattern_num_rands, ring_rotation_rands, ring_angle_rands = \
        th.rand(ring_num, 1, 4).split(1, dim=2)
    one = th.ones([])

    # Calculate and quantize pattern number at each ring
    pattern_num_rands = (pattern_num * pattern_num_rands).clamp_min(pattern_num_min)
    pattern_num = th.lerp(to_tensor(pattern_num), pattern_num_rands, pattern_num_random)
    pattern_num_int = max(math.ceil(to_numpy(pattern_num).max()), 1)

    # Compute ring-level transformations
    ring_indices = th.linspace(0, ring_num - 1, ring_num).view(-1, 1, 1)
    ring_indices_inv = ring_num - ring_indices * radius_multiplier

    ## Radius
    ring_radii = ring_indices_inv * (radius / ring_num)

    ## Angle range
    pattern_num_1 = pattern_num.clamp_min(1.0)
    ring_rotations = ring_rotation + (ring_rotation_rands * 2 - 1) * ring_rotation_random
    ring_rotations = ring_rotations + ring_indices * (ring_rotation_offset / pattern_num_1)

    ring_angle_starts = ring_rotations + (1 - spread) * 0.5
    ring_angle_ends = ring_rotations - (1 - spread) * 0.5
    ring_first_angles = \
        ring_angle_starts + spread / pattern_num_1 * (1.0 if connect_patterns else 0.5)
    ring_first_angles = \
        ring_first_angles * th.lerp(one, ring_angle_rands * 0.666 + 0.667, angle_random)
    ring_angle_steps = spread / pattern_num_1

    ## First position for pattern connection
    ring_first_radii = ring_radii * th.lerp(one, ring_radius_rands, radius_random)
    ring_first_pos = cartesian(ring_first_radii * (1 - spiral_factor), _rad(ring_angle_starts))
    ring_last_pos = cartesian(ring_first_radii, _rad(ring_angle_ends))

    # Since fractional patterns are not rendered in Substance (rather than not defined like in
    # tile generators), we create a mask to remove those patterns
    pattern_num, pattern_indices = pattern_num.floor(), th.arange(pattern_num_int).unsqueeze(1)
    pattern_mask = pattern_indices < pattern_num
    pattern_num_1 = pattern_num.clamp_min(1.0)

    # Pattern-level random pool
    pattern_rand_pool = gen_rand_pool(ring_num, pattern_num_int, 11)
    rands, color_rands, mask_rands, image_rands = pattern_rand_pool.split((8, 1, 1, 1), dim=2)
    size_rands, scale_rands, symmetry_rands, radius_rands, angle_rands, rotation_rands = \
        rands.split((2, 1, 2, 1, 1, 1), dim=2)

    # For connected patterns, precompute all joint positions; this section also generates default
    # pattern positions without pattern connection
    angle_step_scales = th.lerp(one, angle_rands * 0.666 + 0.667, angle_random)
    angle_steps = th.cat((ring_first_angles, ring_angle_steps * angle_step_scales), dim=1)
    angles = th.cumsum(angle_steps, dim=1)[:, :-1]

    radii = ring_radii * th.lerp(one, ring_radii * radius_rands, radius_random)
    radii = radii * th.lerp(one, pattern_indices / pattern_num_1, spiral_factor)
    positions = cartesian(radii, _rad(angles))

    if connect_patterns:
        conn_joints = th.cat((ring_first_pos, positions[:, :-1], ring_last_pos), dim=1)
        conn_vecs = conn_joints.diff(dim=1)

    # Size
    if connect_patterns:
        conn_dists = conn_vecs.norm(dim=-1, keepdim=True)
        sizes_base = th.cat((conn_dists, th.ones_like(conn_dists)), dim=-1)
        sizes_base = sizes_base * to_tensor(size_connected)
    else:
        sizes_base = to_tensor(size)

    sizes = sizes_base * th.lerp(one, size_rands, size_random)
    scales = to_tensor(scale) * th.lerp(one, scale_rands, scale_random)
    sizes = sizes * scales

    ## Scaling by ring/pattern number
    scales_ring_num = ring_indices_inv if scale_by_ring_number_invert else ring_indices
    scales_ring_num = scales_ring_num / ring_num
    scales_pattern_num = \
        pattern_num - pattern_indices - 1 if scale_by_pattern_number_invert else pattern_indices
    scales_pattern_num = scales_pattern_num / pattern_num_1

    sizes = sizes * th.lerp(one, scales_ring_num, scale_by_ring_number)
    sizes = sizes * th.lerp(one, scales_pattern_num, scale_by_pattern_number)
    sizes_unflipped = sizes.clone()

    ## Symmetry flip
    flips = th.where(symmetry_rands >= symmetry_random * 0.5, one, -one)
    if symmetry_random_mode == 'both':
        sizes = sizes * flips
    elif symmetry_random_mode == 'horizontal':
        sizes.select(-1, 0).mul_(flips.select(-1, 0))
    elif symmetry_random_mode == 'vertical':
        sizes.select(-1, 1).mul_(flips.select(-1, 1))
    else:
        raise ValueError(f'Unknown symmetry random mode: {symmetry_random_mode}')

    # Rotation
    if connect_patterns:
        rotations_base = th.arctan2(conn_vecs.narrow(-1, 1, 1), conn_vecs.narrow(-1, 0, 1))
        rotations_base = rotations_base / (math.pi * 2)
    else:
        rotations_base = angles + 0.25 if center_orientation else th.zeros([])

    rotation_rands = rotation_rands * 2 - 1
    rotations = pattern_rotation + th.lerp(rotations_base, rotation_rands, pattern_rotation_random)

    # Position
    # Note that default positions when patterns are disconnected have been precomputed before
    # connection joints
    if connect_patterns:
        positions = (conn_joints[:, :-1] + conn_joints[:, 1:]) * 0.5

    ## Directional offsets
    rotations_rad = _rad(rotations)
    positions = positions + cartesian(directional_offset, rotations_rad)

    ## Pivotal pattern rotations
    if pattern_rotation_pivot != 'center':
        pivot_index = ['min_x', 'max_x', 'min_y', 'max_y'].index(pattern_rotation_pivot)
        pivot_dim = 0 if pivot_index < 2 else 1
        pivot_dists = (-0.5 if pivot_index in (0, 3) else 0.5) * \
                      sizes_unflipped.narrow(-1, pivot_dim, 1)
        vec1 = cartesian(pivot_dists, _rad(rotations_base), add_90_deg=pivot_dim)
        vec2 = cartesian(pivot_dists, rotations_rad, add_90_deg=pivot_dim)
        positions = positions + vec1 - vec2

    ## Global offset
    positions = positions + to_tensor(global_offset) + 0.5

    # Color
    colors = to_tensor(color)

    ## Color by ring/pattern number
    color_ring_num = ring_indices_inv if color_by_ring_number_invert else ring_indices + 1
    color_ring_num = color_ring_num / ring_num
    color_pattern_num = \
        pattern_num - pattern_indices if color_by_pattern_number_invert else pattern_indices + 1
    color_pattern_num = color_pattern_num / pattern_num_1

    colors = colors * th.lerp(one, color_pattern_num, color_by_pattern_number)
    colors = colors * th.lerp(one, color_ring_num, color_by_ring_number)

    ## Color by scale
    colors = colors * th.lerp(one, scales, color_by_scale)
    colors = colors * th.lerp(one, color_rands, color_random)

    # Mask
    mask = (pattern_mask * (mask_rands >= random_mask)).squeeze(-1)

    ## Image index
    is_image = pattern == 'image'

    if not is_image:
        image_indices: List[int] = [0]
    else:
        if input_distribution == 'random':
            image_indices = (to_numpy(image_rands) * input_number).astype(np.int64)
        elif input_distribution == 'by_pattern':
            image_indices: npt.NDArray[np.int64] = \
                np.tile(np.arange(pattern_num_int), (ring_num, 1)) % input_number
        elif input_distribution == 'by_ring':
            image_indices: npt.NDArray[np.int64] = \
                to_numpy(ring_indices_inv).astype(np.int64) % input_number
        else:
            raise ValueError(f'Unknown input distribution mode: {input_distribution}')

        image_indices = image_indices.ravel()

    # Assemble job array
    job_arr: FXMapJobArray = {
        'color': colors[mask].view(-1, 1),
        'offset': positions[mask].view(-1, 2),
        'size': sizes[mask].view(-1, 2),
        'rotation': rotations[mask].view(-1, 1),
        'variation': [pattern_specific],
        'depth': [0],
        'blending': [FXE.BLEND_DICT['max' if blending_mode == 'max' else 'add']],
        'filtering': [FXE.FILTER_DICT[input_filter_mode]],
        'image_index': image_indices[to_numpy(mask).ravel()].tolist() if is_image else \
                       image_indices,
    }

    # Execute FX-map
    blending_opacity = th.atleast_1d(to_tensor(global_opacity))
    fx_map = executor.evaluate(blending_opacity, batched_jobs={pattern: job_arr})

    return fx_map


def brick_generator(res_h: int = 256, res_w: int = 256, bricks: Tuple[int, int] = [4, 8],
                    bevel: FloatVector = [0.5, 0.5, 0.0, 0.0], keep_ratio: bool = True,
                    gap: FloatVector = [0.0, 0.0], middle_size: FloatVector = [0.5, 0.5],
                    height: FloatVector = [1.0, 1.0, 0.0, 0.0], slope: FloatVector = [0.0] * 4,
                    offset: FloatVector = [0.5, 0.0]) -> th.Tensor:
    """Pattern generator: brick generator
    """
    if res_h != res_w:
        raise ValueError('Only square textures are supported for now')

    # Unpack parameters
    num_x, num_y = bricks
    middle_x, middle_y = to_tensor(middle_size).unbind()
    bevel_x, bevel_y, bevel_smooth, bevel_round = to_tensor(bevel).unbind()
    gap_x, gap_y = to_tensor(gap).unbind()
    height_min, height_max, height_bal, height_var = to_tensor(height).unbind()
    slope_x, slope_y, slope_bal, slope_var = to_tensor(slope).unbind()
    offset, offset_random = to_tensor(offset).unbind()

    ## Compute brick indices and offsets from brick boundaries
    def get_brick_data(coords: th.Tensor, num: int, middle: th.Tensor) -> Tuple[th.Tensor, ...]:

        # Compute indices and offsets for adjacent brick pairs (using the `middle` parameter)
        size_pair = 1 / (num // 2 + (middle.clamp_min(1e-8) if num % 2 else 0))
        size_left, size_right = size_pair * middle, size_pair * (1 - middle)
        pair_inds, pair_coords = (coords / size_pair).long(), coords % size_pair

        # Compute indices, sizes, and offsets for individual bricks
        mid_mask = pair_coords >= size_left
        inds, inner_coords = pair_inds * 2 + mid_mask, pair_coords - mid_mask * size_left
        sizes = th.where(mid_mask, size_right, size_left)

        # Compute minimal brick size
        size_min = th.max(size_left, size_right) if num == 1 else th.min(size_left, size_right)

        return inds, inner_coords, sizes, size_min

    # Compute brick row indices for all pixels
    y_coords = th.linspace(0.5 / res_h, 1 - 0.5 / res_h, res_h)
    y_inds, y_inner_coords, y_sizes, y_size_min = get_brick_data(y_coords, num_y, middle_y)

    # Get corresponding X offsets
    x_offsets = y_inds / num_x * offset
    x_offset_rands = to_tensor(global_random(to_numpy(y_inds))) * offset_random
    x_offsets = x_offsets + x_offset_rands * (y_inds + 1) / num_y

    # Compute brick column indices for all pixels
    x_coords = (th.linspace(0.5 / res_w, 1 - 0.5 / res_w, res_w) - x_offsets.unsqueeze(1)) % 1
    x_inds, x_inner_coords, x_sizes, x_size_min = get_brick_data(x_coords, num_x, middle_x)
    size_min = th.min(y_size_min, x_size_min)

    ## Calculate beveled brick pattern
    def brick_bevel(coords: th.Tensor, sizes: th.Tensor, bevel: th.Tensor, bevel_smooth: th.Tensor,
                    gap: th.Tensor, brick_size: th.Tensor, vertical: bool = False) -> th.Tensor:

        # Apply interstices to obtain the brick pattern
        interstice_bevel = bevel.clamp_max(1 - gap) * (size_min if keep_ratio else brick_size)
        interstice = interstice_bevel * 0.25 + gap * brick_size * 0.5
        bricks = ((coords >= interstice) & (coords < sizes - interstice)).float()

        # Expand the pattern into a 4D image
        bricks = bricks.expand(res_h, res_w).unflatten(0, (1, 1, -1))

        # Apply the two-pass beveling effect
        angle = 0.25 if vertical else 0.0
        intensity = interstice_bevel * bevel_smooth * 0.5 * 64
        bricks = d_blur(bricks, intensity=intensity, angle=angle)
        intensity = interstice_bevel * (1 - bevel_smooth * 0.5) * 64
        bricks = d_blur(bricks, intensity=intensity, angle=angle)

        return bricks

    # Calculate the base brick pattern
    brick_x = brick_bevel(x_inner_coords, x_sizes, bevel_x, bevel_smooth, gap_x, x_size_min)
    brick_y = brick_bevel(y_inner_coords.view(-1, 1), y_sizes.view(-1, 1), bevel_y, bevel_smooth,
                          gap_y, y_size_min, vertical=True)
    brick = th.lerp(th.min(brick_x, brick_y), brick_x * brick_y, bevel_round)

    # Prepare the pixel-to-brick sampling function
    indices = th.stack((x_inds, y_inds.view(-1, 1).expand_as(x_inds)), dim=-1)
    brick_grid = ((indices + 0.5) / to_tensor(bricks) * 2 - 1).unsqueeze(0)

    def brick_sample(img: th.Tensor) -> th.Tensor:
        img = img.unflatten(0, (1, 1, -1))
        return grid_sample_impl(img, brick_grid, mode='nearest', align_corners=False)

    ## Calculate per-pixel slope blending mask
    def slope_mask(slope_rands: th.Tensor) -> th.Tensor:
        mask = (slope_rands > (slope_var + 1) * 0.5) == (slope_rands > 0.5)
        return brick_sample(mask.float()).bool()

    ## Calculate per-pixel slope map
    def slope_map(coords: th.Tensor, sizes: th.Tensor, slope: th.Tensor,
                  slope_mask: th.Tensor, vertical: bool = False) -> th.Tensor:

        # Compute slope maps
        slope_pos = (coords / sizes.clamp_min(1e-8)).clamp_max(1.0)
        slope_neg = 1 - slope_pos
        img_bg, img_fg = (slope_neg, slope_pos) if (to_const(slope) < 0) != vertical else \
                         (slope_pos, slope_neg)

        # Obtain the blended slope map
        img_slope = th.where(slope_mask, img_fg, img_bg)
        img_slope = 1 - (1 - img_slope) * slope.abs()

        return img_slope

    # Compute color slopes in X and Y directions
    color_rands = th.rand((num_x, num_y)).T
    x_slope_rands, y_slope_rands = color_rands % 0.5 * 2, (color_rands - 0.25) % 0.5 * 2
    img_slope_x = slope_map(x_inner_coords, x_sizes, slope_x, slope_mask(x_slope_rands))
    img_slope_y = slope_map(y_inner_coords.view(-1, 1), y_sizes.view(-1, 1), slope_y,
                            slope_mask(y_slope_rands), vertical=True)

    # Balance between X and Y slopes
    slope_bal_mask = th.lerp(
        slope_bal.clamp(0.0, 1.0), (slope_bal + 1).clamp(0.0, 1.0), 1 - y_slope_rands)
    img_slope = th.lerp(img_slope_y, img_slope_x, brick_sample(slope_bal_mask))
    brick = brick * img_slope

    # Compute positive and negative brick heights
    h_min, h_max = th.min(height_min, height_max), th.max(height_min, height_max)
    h_min_r, h_min_nr = h_min.clamp_min(0.0), -h_min.clamp_max(0.0)
    h_max_r, h_max_nr = h_max.clamp_min(0.0), -h_max.clamp_max(0.0)
    h_var_r, h_var_nr = height_var.clamp_min(0.0), -height_var.clamp_max(0.0)
    h_ratio = h_min_nr / (1 + h_max_r)

    height_rands = 1 - x_slope_rands
    height_bal_mask = (1 - height_rands % 0.5 * 2) > (height_bal + 1) * 0.5

    height_pos = th.where(height_bal_mask, h_min_r, h_max_r) * (1 - h_ratio)
    height_pos = th.lerp(height_pos, height_pos * height_rands, h_var_r)
    height_pos = th.lerp(height_pos, height_pos * x_slope_rands, h_var_nr)

    height_neg = th.where(height_bal_mask, h_min_nr, h_max_nr) * h_ratio
    height_neg = th.lerp(height_neg, height_neg * height_rands, h_var_r)
    height_neg = th.lerp(height_neg, height_neg * x_slope_rands, h_var_nr)

    # Apply brick heights
    img_out = (brick * brick_sample(height_pos) + h_ratio).clamp_max(1.0)
    img_out = (img_out - brick * brick_sample(height_neg)).clamp_min(0.0)

    return img_out


def stripes(res_h: int = 256, res_w: int = 256, num: int = 10, width: FloatValue = 0.5,
            softness: FloatValue = 0.0, shift: int = 10, align: str = 'edges',
            filtering: bool = True) -> th.Tensor:
    """Pattern generator: stripes
    """
    if res_h != res_w:
        raise ValueError('Only square textures are supported for now')

    # Generate the stripes pattern analytically
    x_coords = th.linspace(0.5 / res_w, 1 - 0.5 / res_w, res_w)
    y_coords = th.linspace(0.5 / res_h, 1 - 0.5 / res_h, res_h).unsqueeze(1)
    y_offset = 0.5 if not (num + shift) % 2 and align == 'center' else 0

    y_coords = (y_coords * num - x_coords * shift + y_offset) % 1
    width_smooth = width * (1 - softness * 0.5)
    stripes = (y_coords >= (1 - width_smooth) * 0.5) & (y_coords < (1 + width_smooth) * 0.5)
    stripes = stripes.float().unflatten(0, (1, 1, -1))

    # Apply filtering
    if filtering:
        stripes = d_blur(stripes, intensity=num * 0.5 / max(abs(shift), 1))
        stripes = d_blur(stripes, intensity=max(abs(shift), 1) * 0.5 / num, angle=0.25)

    # Apply softness
    stripes = d_blur(stripes, intensity=width * softness * (64 / num), angle=0.25)

    return stripes


@input_check(1, channel_specs='g')
def arc_pavement(img_in: Optional[th.Tensor] = None, res_h: int = 256, res_w: int = 256,
                 scale: int = 1, pattern_num: int = 12, pattern_num_random: FloatValue = 0.0,
                 pattern_num_min: int = 1, arc_num: int = 14, pattern: str = 'square',
                 pattern_scale: FloatValue = 1.0, pattern_width: FloatValue = 0.8,
                 pattern_height: FloatValue = 0.9, pattern_width_random: FloatValue = 0.0,
                 pattern_height_random: FloatValue = 0.0, pattern_spacing_random: FloatValue = 0.0,
                 pattern_height_decay: FloatValue = 0.25, color_random: FloatValue = 0.0,
                 device: DeviceType = 'cpu') -> th.Tensor:
    """Pattern generator: arc pavement
    """
    if res_h != res_w:
        raise ValueError('Only square textures are supported for now')

    # Initiate FX-map executor
    executor = FXE(int(math.log2(res_h)), device=device)
    executor.reset(None, img_in)

    # Return the FX-map canvas if there is no pattern to render
    if pattern_num < 0 or arc_num <= 0:
        return executor.canvas

    # Sample the number of patterns per arc
    num_groups = scale ** 2
    pattern_num_rands = th.rand(num_groups, arc_num, 2) * pattern_num_random
    num = pattern_num - (pattern_num - pattern_num_min) * pattern_num_rands
    num = (num.ceil().long() - th.arange(2)).clamp_min(1).view(num_groups, -1, 1)
    max_num = to_numpy(num).max()

    # Generate pattern-wise random number pool
    rand_pool = gen_rand_pool(num_groups, arc_num * 2, max_num, 4)
    spacing_rands, width_rands, height_rands, color_rands = rand_pool.split(1, dim=-1)

    # Compute pattern connector positions on all arcs
    spacing = th.arange(max_num + 1.0).tile(num_groups, arc_num * 2, 1)
    spacing_rands = (spacing_rands.squeeze(-1) - 0.5) * pattern_spacing_random
    spacing_rands.masked_fill_(th.arange(1, max_num + 1) >= num, 0.0)
    spacing[...,1:] += spacing_rands

    radius, initial_angle = 0.25 * math.sqrt(2.0) / scale, math.pi * (1.25 - 0.25 / pattern_num)
    angles = initial_angle + 0.5 * math.pi * spacing / num
    conn_pos = radius * th.stack((th.cos(angles), th.sin(angles)), dim=-1)

    # Compute pattern center positions (apply group offsets)
    conn_vec = conn_pos.diff(dim=-2)
    positions = conn_pos[...,:-1,:] + conn_vec * 0.5
    arc_inds = th.stack(th.meshgrid(th.arange(2), th.arange(arc_num), indexing='ij'), dim=-1)
    arc_offsets = (arc_inds / to_tensor([2 * scale, arc_num * scale])).view(-1, 1, 2)
    positions = positions + arc_offsets + get_group_pos(scale).unflatten(-1, (1, 1, -1))

    # Compute pattern sizes
    one = th.ones([])
    widths = pattern_width * conn_vec.norm(dim=-1, keepdim=True)
    widths = widths * th.lerp(one, width_rands, pattern_width_random)
    heights = pattern_height / (arc_num * scale)
    heights = heights * th.lerp(one, height_rands, pattern_height_random)
    heights_decay = 0.5 - 0.5 * th.cos(th.arange(max_num) / num * (math.pi * 2))
    heights = heights * th.lerp(one, heights_decay.unsqueeze(-1), pattern_height_decay)
    sizes = th.cat((widths, heights), dim=-1) * pattern_scale

    # Compute pattern rotations
    rotations = th.arctan2(conn_vec[..., 1:], conn_vec[..., :1]) / (math.pi * 2)

    # Compute pattern colors
    colors = th.lerp(one, color_rands, color_random)

    # Obtain the pattern validity mask and construct the FX-map job array
    mask = th.arange(max_num) < num
    job_arr: FXMapJobArray = {
        'color': colors[mask].view(-1, 1),
        'offset': positions[mask].view(-1, 2),
        'size': sizes[mask].view(-1, 2),
        'rotation': rotations[mask].view(-1, 1),
        'variation': [0.0],
        'depth': [0],
        'blending': [FXE.BLEND_DICT['max']],
        'filtering': [FXE.FILTER_DICT['nearest']],
        'image_index': [0],
    }

    # Execute FX-map
    fx_map = executor.evaluate([1.0], batched_jobs={pattern: job_arr})

    return fx_map


def shape(res_h: int = 256, res_w: int = 256, tiling: int = 1, pattern: str = 'square',
          variation: FloatValue = 0.0, scale: FloatValue = 1.0, size: FloatVector = [1.0, 1.0],
          angle: FloatValue = 0.0, rotation_45: bool = False) -> th.Tensor:
    """Pattern generator: shape
    """
    if res_h != res_w:
        raise ValueError('Only square textures are supported for now')
    if not tiling:
        return th.zeros(1, 1, res_h, res_w)

    # hemisphere pattern function
    def hemisphere(grid: th.Tensor, _: FloatValue = 0.0) -> th.Tensor:
        return th.sqrt(1 - (grid ** 2).sum(dim=-1, keepdim=True).clamp_max(1.0))

    # Prepare the atomic pattern function dictionary
    pattern_func_dict = ATOMIC_PATTERNS.copy()
    pattern_func_dict['hemisphere'] = hemisphere
    del pattern_func_dict['gradation_offset']

    if pattern not in pattern_func_dict:
        raise ValueError(f'Unknown pattern type: {pattern}')

    # Convert parameters to tensors
    scale, size, angle = to_tensor(scale), to_tensor(size), to_tensor(angle * (math.pi * 2))
    if pattern == 'hemisphere':
        size = size.clamp_min(1e-8) * scale.clamp_min(1e-8)
    else:
        size = th.where(size >= 0, size.clamp_min(1e-8), size.clamp_max(-1e-8)) * \
               th.where(scale >= 0, scale.clamp_min(1e-8), scale.clamp_max(-1e-8))

    # Obtain the affine coordinate grid
    cos_angle, sin_angle = angle.cos(), angle.sin()
    rot_matrix = th.stack((cos_angle, sin_angle, -sin_angle, cos_angle)).view(2, 2)
    theta = th.hstack((rot_matrix / size.unsqueeze(1), th.zeros(2, 1))).unsqueeze(0)
    grid = affine_grid(theta, [1, 1, res_h, res_w], align_corners=False)

    # Compute the shape pattern and optionally rotate 45 degrees
    img_pattern = pattern_func_dict[pattern](grid, variation)
    img_pattern = img_pattern[0] if isinstance(img_pattern, (tuple, list)) else img_pattern
    img_pattern = img_pattern.movedim(-1, -3)
    if rotation_45:
        img_pattern = transform_2d(img_pattern, matrix22=[1.0, -1.0, 1.0, 1.0])

    # Apply tiling
    if tiling != 1:
        offset = [(tiling + 1) % 2 * 0.5] * 2
        img_pattern = transform_2d(img_pattern, matrix22=[tiling, 0, 0, tiling], offset=offset)

    return img_pattern


@input_check_all_positional(channel_specs='ggggcg')
def tile_sampler(img_bg: Optional[th.Tensor], img_scale: Optional[th.Tensor],
                 img_disp: Optional[th.Tensor], img_rotation: Optional[th.Tensor],
                 img_vector: Optional[th.Tensor], img_color: Optional[th.Tensor],
                 img_mask: Optional[th.Tensor], img_pattern_dist: Optional[th.Tensor],
                 *img_list: Optional[th.Tensor], res_h: int = 256, res_w: int = 256,
                 x_num: int = 16, y_num: int = 16, pattern: str = 'square', input_number: int = 1,
                 input_distribution: str = 'random', input_filter_mode: str = 'bilinear_mipmap',
                 pattern_var: FloatValue = 0.0, pattern_var_random: FloatValue = 0.0,
                 fixed_rotation: int = 0, fixed_rotation_random: float = 0.0,
                 symmetry_random: float = 0.0, symmetry_random_mode: str = 'both',
                 size_mode: str = 'scale', size: FloatVector = [1.0, 1.0],
                 size_absolute: FloatVector = [0.1, 0.1], size_pixel: FloatVector = [1.0, 1.0],
                 size_random: FloatVector = [0.0, 0.0], scale: FloatValue = 0.8,
                 scale_random: FloatValue = 0.0, scale_map: FloatValue = 0.0,
                 scale_vector_map: FloatValue = 0.0, scale_map_effect: str = 'both',
                 position_random: FloatValue = 0.0, offset: FloatValue = 0.0,
                 offset_type: str = 'x_alt', global_offset: FloatVector = [0.0, 0.0],
                 disp_map: FloatValue = 0.0, disp_angle: FloatValue = 0.0,
                 disp_vector_map: FloatValue = 0.0, rotation: FloatValue = 0.0,
                 rotation_random: FloatValue = 0.0, rotation_map: FloatValue = 0.0,
                 rotation_vector_map: FloatValue = 0.0, mask_map_threshold: float = 0.0,
                 mask_map_invert: bool = False, mask_map_sampling: str = 'center',
                 mask_random: float = 0.0, mask_invert: bool = False, blending_mode: str = 'max',
                 color: FloatValue = 1.0, color_random: FloatValue = 0.0,
                 color_scale_mode: str = 'input', color_scale: FloatValue = 0.0,
                 global_opacity: FloatValue = 1.0, background_color: FloatValue = 1.0,
                 reverse_order: bool = False, device: DeviceType = 'cpu') -> th.Tensor:
    """Pattern generator: Tile Sampler
    """
    if res_h != res_w:
        raise ValueError('Only square textures are supported for now')
    if input_number not in range(1, 7):
        raise ValueError('The number of input images must be from 1 to 6')
    if len(img_list) < input_number:
        raise ValueError(f'Expect {input_number} images but only {len(img_list)} provided')

    # Initiate FX-map executor
    executor = FXE(int(math.log2(res_h)), device=device)
    executor.reset(img_bg, *img_list, background_color=background_color)

    # Quantize X and Y amount
    # Only return the FX-map canvas (already applied with background color and image) if there is
    # no pattern to render
    if min(x_num, y_num) <= 0:
        return executor.canvas

    # Random number pools (the pool must be triangular so that changes in the number of tiles won't
    # affect the actual random number sequence
    rands, rands_np = gen_rand_pool(y_num, x_num, 12).split((10, 2), dim=-1)
    size_rands, offset_rands, rotation_rands, color_rands, pattern_rands, mask_rands \
        = rands.split((5, 1, 1, 1, 1, 1), dim=-1)
    fixed_rot_rands, image_rands = np.split(to_numpy(rands_np), 2, axis=-1)

    # Base sampling position
    x_indices, y_indices = th.meshgrid(
        th.arange(x_num - 1, -1, -1) if reverse_order else th.arange(x_num),
        th.arange(y_num - 1, -1, -1) if reverse_order else th.arange(y_num), indexing='xy')
    indices = th.stack((x_indices, y_indices), dim=-1)
    pos = (indices + 0.5 + to_tensor(global_offset)) / to_tensor([x_num, y_num])

    ## Row/column-specific offset
    if offset_type in ('x_alt', 'x_global'):
        pos[..., 0] += (y_indices % 2 if offset_type == 'x_alt' else y_indices) * offset / x_num
    elif offset_type in ('y_alt', 'y_global'):
        pos[..., 1] += (x_indices % 2 if offset_type == 'y_alt' else x_indices) * offset / y_num
    else:
        raise ValueError(f'Unknown offset type: {offset_type}')

    ## Position random
    pos_rand_angles = offset_rands * (math.pi * 2.0)
    pos_rand_vecs = th.cat((th.cos(pos_rand_angles), th.sin(pos_rand_angles)), dim=-1)
    pos = (pos + pos_rand_vecs * (position_random / max(x_num, y_num))) % 1

    # Apply vector and warp maps
    pos_sample = (pos * 2 - 1).unsqueeze(0)
    sample_nn: Callable[[th.Tensor, th.Tensor], th.Tensor] \
        = lambda img, grid: grid_sample_impl(
            img, grid, mode='nearest', align_corners=False).squeeze(0).movedim(0, -1)

    ## Vector map displacement
    vector_map = sample_nn(img_vector, pos_sample)[..., :2] * 2 - 1 \
                 if img_vector is not None else None
    pos = pos + vector_map * disp_vector_map if vector_map is not None else pos - disp_vector_map

    ## Warp displacement
    if img_disp is not None:
        warp_rad = sample_nn(img_disp, pos_sample) * -disp_map
        warp_angle = to_tensor(disp_angle * (math.pi * 2.0))
        pos = pos + warp_rad * th.stack((th.cos(warp_angle), th.sin(warp_angle)))

    ## Update sampling coordinates using normalized positions
    pos = pos % 1
    pos_sample = (pos * 2 - 1).unsqueeze(0)

    # Rotation
    ## Base rotation
    rotations_np = np.floor(fixed_rot_rands * 4).astype(np.int64)
    rotations_np = np.where(fixed_rot_rands * (rotations_np + 1) % 1 > fixed_rotation_random,
                            fixed_rotation // 90, rotations_np)
    rotations = rotation + to_tensor(rotations_np * 0.25)
    rotations = rotations + (rotation_rands * 2 - 1) * rotation_random

    ## Apply vector and rotation maps
    rotations_param = th.arctan2(*vector_map.split(1, dim=-1)[::-1]) / (math.pi * 2.0) \
                      if vector_map is not None else -0.375
    rotations = rotations + rotations_param * rotation_vector_map
    rotations = rotations + sample_nn(img_rotation, pos_sample) * rotation_map \
                if img_rotation is not None else rotations

    # Size
    size_rands, scale_rands, symmetry_rands = size_rands.split((2, 1, 2), dim=-1)
    one = th.ones([])

    ## Mode-specific size
    if size_mode == 'scale':
        sizes = to_tensor(size) * to_tensor([1 / x_num, 1 / y_num])
    elif size_mode == 'scale_square':
        sizes = to_tensor(size) * to_tensor([1 / max(x_num, y_num)] * 2)
    elif size_mode in ('absolute', 'pixel'):
        sizes = to_tensor(size_absolute if size_mode == 'absolute' else size_pixel)
    else:
        raise ValueError(f'Unknown size mode: {size_mode}')

    ## Apply vector and scale maps
    scales_param = th.lerp(one, vector_map.norm(dim=-1, keepdim=True), scale_vector_map) \
                   if vector_map is not None else scale_vector_map * (math.sqrt(2.0) - 1) + 1
    scales_param = scales_param * th.lerp(one, sample_nn(img_scale, pos_sample), scale_map) \
                   if img_scale is not None else scales_param * (1 - scale_map)

    if scale_map_effect == 'both':
        sizes = sizes * scales_param
    elif scale_map_effect == 'x':
        sizes[..., 0] *= scales_param.squeeze(-1)
    elif scale_map_effect == 'y':
        sizes[..., 1] *= scales_param.squeeze(-1)
    else:
        raise ValueError(f'Unknown scale parametrization mode: {scale_map_effect}')

    ## Flip the size of 90/270 fixed-rotated patterns
    sizes = th.where(th.as_tensor((rotations_np % 2).astype(bool)), th.flip(sizes, [-1]), sizes)

    ## Global scaling
    scales_rand = th.lerp(one, scale_rands, scale_random)
    sizes = sizes * th.lerp(one, size_rands, size_random) * (scale * scales_rand)

    ## For pixel mode, round up to the next integer value
    if size_mode == 'pixel':
        sizes = sizes.ceil() / to_tensor([res_w, res_h])

    ## Symmetry flip
    flips = th.where(symmetry_rands >= symmetry_random * 0.5, one, -one)
    if symmetry_random_mode == 'both':
        sizes = sizes * flips
    elif symmetry_random_mode == 'horizontal':
        sizes[..., 0] *= flips[..., 0]
    elif symmetry_random_mode == 'vertical':
        sizes[..., 1] *= flips[..., 1]
    else:
        raise ValueError(f'Unknown symmetry random mode: {symmetry_random_mode}')

    # Color
    ## Color parametrization
    if color_scale_mode == 'input':
        colors_param = sample_nn(img_color, pos_sample) if img_color is not None else th.zeros([])
    elif color_scale_mode == 'scale':
        colors_param = scales_param * scales_rand
    elif color_scale_mode in ('row', 'column'):
        x_param, y_param = th.meshgrid(
            th.linspace(1 / x_num, 1, x_num), th.linspace(1 / y_num, 1, y_num), indexing='xy')
        colors_param = (y_param if color_scale_mode == 'row' else x_param).unsqueeze(-1)
    elif color_scale_mode == 'number':
        colors_param = th.linspace(1 / (x_num * y_num), 1, x_num * y_num).view(y_num, x_num, 1)
    else:
        raise ValueError(f'Unknown color parametrization mode: {color_scale_mode}')

    ## Base color
    colors = th.lerp(to_tensor(color), colors_param, color_scale)
    colors = th.lerp(colors, color_rands, color_random)

    ## Account for background color
    if blending_mode == 'add_sub':
        colors = colors - background_color

    # Mask
    ## Sample mask map
    if mask_map_sampling not in ('center', 'bbox'):
        raise ValueError(f'Unknown mask map sampling mode: {mask_map_sampling}')

    if img_mask is None:
        mask_map = 1.0 if mask_map_invert else 0.0
    else:
        img_mask = 1.0 - img_mask if mask_map_invert else img_mask

        ## Mask map sampling
        if mask_map_sampling == 'center':
            mask_map = sample_nn(img_mask, pos_sample).squeeze(-1)
        else:
            bbox_scale = to_tensor([[0.4, 0.4], [0.4, -0.4], [-0.4, 0.4], [-0.4, -0.4]])
            pos_bbox = (sizes.unsqueeze(2) * bbox_scale + pos.unsqueeze(2)).movedim(2, 0) % 1
            mask_map = grid_sample_impl(
                img_mask.expand(4, -1, -1, -1), pos_bbox * 2 - 1, mode='nearest',
                align_corners=False)
            mask_map = mask_map.squeeze(1).min(dim=0)[0]

    ## Calculate final mask
    mask = (mask_rands > mask_random).squeeze(-1) & (mask_map >= mask_map_threshold)
    mask = ~mask if mask_invert else mask

    # Pattern distribution
    image_indices: List[int] = [0]

    if input_distribution == 'random':
        image_indices = (image_rands.squeeze(-1) * input_number).astype(np.int64)
    elif input_distribution == 'cycle':
        image_indices = np.arange(x_num * y_num).reshape(y_num, x_num) % input_number
    elif input_distribution == 'map' and img_pattern_dist is not None:
        dist_map = sample_nn(img_pattern_dist, pos_sample)
        image_indices = (to_numpy(dist_map) * input_number).astype(np.int64)
    elif input_distribution != 'map':
        raise ValueError(f'Unknown input pattern distribution mode: {input_distribution}')

    # Pattern variation
    variations = pattern_var * th.lerp(one, pattern_rands, pattern_var_random)

    # Assemble job array
    job_arr: FXMapJobArray = {
        'color': colors[mask].view(-1, 1),
        'offset': pos[mask].view(-1, 2),
        'size': sizes[mask].view(-1, 2),
        'rotation': rotations[mask].view(-1, 1),
        'variation': variations[mask].view(-1, 1),
        'depth': [0],
        'blending': [FXE.BLEND_DICT['max' if blending_mode == 'max' else 'add']],
        'filtering': [FXE.FILTER_DICT[input_filter_mode]],
        'image_index': image_indices[to_numpy(mask)] if not isinstance(image_indices, list) \
                       else image_indices,
    }

    # Execute FX-map
    blending_opacity = th.atleast_1d(to_tensor(global_opacity))
    fx_map = executor.evaluate(blending_opacity, batched_jobs={pattern: job_arr})

    return fx_map


# ------------------------------------------------------------------------ #
#          Helper functions for noise and pattern generator nodes          #
# ------------------------------------------------------------------------ #


def disorder_func(radius: FloatValue, angle: th.Tensor, rands: th.Tensor) -> th.Tensor:
    """A disorder function which creates random offset variations in noise generators.
    """
    radius_rands, angle_rands = rands.split(1, dim=-1)
    radii = radius_rands * radius
    angles = (angle + (angle_rands * 2 - 1)) * (math.pi * 2)
    return radii * th.cat((th.cos(angles), th.sin(angles)), dim=-1)


def gen_rand_pool(*shape: int) -> th.Tensor:
    """Generate a random number pool (up to 4D) such that all dimensions but the last one do not
    interrupt the random number sequence.
    """
    if len(shape) > 4:
        raise ValueError(f'The input shape should not exceed 4D, got {shape}')
    elif len(shape) <= 2 or min(shape) <= 0:
        return th.rand(shape)

    # Extract dimensions
    z_num, y_num, x_num, size = shape if len(shape) == 4 else (0, *shape)

    # Construct the triangular random number pool
    tri_size = (x_num + y_num) * (x_num + y_num - 1) // 2
    tri_pool = th.rand(tri_size, size) if not z_num else \
               th.stack([th.rand(tri_size, size) for _ in range(z_num)])

    # Re-index the triangular random number pool into a rectangular pool
    r_inds, c_inds = np.ogrid[:y_num, :x_num]
    r_inds = r_inds + c_inds
    tri_inds = th.as_tensor(r_inds * (r_inds + 1) // 2 + c_inds).long().ravel()

    return tri_pool.index_select(-2, tri_inds).unflatten(-2, (y_num, x_num))


def tile_middle_size(x_num: int, y_num: int, middle_size: FloatVector = [0.5, 0.5]) -> \
        Tuple[th.Tensor, th.Tensor]:
    """Calculates tile scales and positions from the `middle_size` parameter. Used in tile
    generator nodes.
    """
    middle_x, middle_y = (to_tensor(middle_size) * 2).unbind()
    one = th.ones([])

    # Compute scales after applying the middle size
    def get_mid_scale(num: int, mid: th.Tensor) -> th.Tensor:
        alt_mask = np.arange(num) % 2
        scale_0 = to_tensor(alt_mask * (num / max(num // 2, 1)))
        scale_2 = to_tensor((1 - alt_mask) * (num / ((num + 1) // 2)))
        return th.where(mid < 1, th.lerp(scale_0, one, mid), th.lerp(one, scale_2, mid - 1))

    # Compute positions after applying the middle size
    def get_mid_pos(num: int, mid: th.Tensor) -> th.Tensor:
        pos_0 = th.linspace(0, (num - 1) / max(num - num % 2, 1), num)
        pos_1 = th.linspace(0.5 / num, 1 - 0.5 / num, num)
        pos_2 = th.linspace(1 / (num + num % 2), num / (num + num % 2), num)
        return th.where(mid < 1, th.lerp(pos_0, pos_1, mid), th.lerp(pos_1, pos_2, mid - 1))

    # Middle size scaling
    mid_scale_x, mid_scale_y = get_mid_scale(x_num, middle_x), get_mid_scale(y_num, middle_y)
    mid_scale = th.stack(th.meshgrid(mid_scale_x, mid_scale_y, indexing='ij'), dim=-1)

    # Middle size positions
    mid_pos_x, mid_pos_y = get_mid_pos(x_num, middle_x), get_mid_pos(y_num, middle_y)
    mid_pos = th.stack(th.meshgrid(mid_pos_x, mid_pos_y, indexing='ij'), dim=-1)

    return mid_scale.view(-1, 2), mid_pos.view(-1, 2)


def global_random(seeds: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
    """A random number generator that produces consistent results given an input seed.
    """
    r = (1234 * 2 ** 35) / ((seeds + 4321) % 65536 + 4321)
    r = (((r % 1e6) - (r % 100)) * 0.01) ** 2
    r = ((r % 1e6) - (r % 100)) * 1e-6
    return r.astype(np.float32)
