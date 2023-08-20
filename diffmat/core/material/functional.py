from typing import Tuple, List, Callable, Union, Optional, Any
import math

from torch.nn.functional import conv2d, grid_sample as grid_sample_impl, affine_grid
from torchvision.ops import deform_conv2d
from scipy.ndimage.morphology import distance_transform_edt
import numpy as np
import torch as th

from diffmat.core.types import FloatValue, FloatVector, FloatArray, FXMapJobArray, NodeFunction
from diffmat.core.log import get_logger
from diffmat.core.operator import *
from diffmat.core.util import check_arg_choice, to_const, to_tensor, to_tensor_and_const
from .util import input_check, input_check_all_positional, color_input_check, grayscale_input_check


# Logger for the functional module
logger = get_logger('diffmat.core')


# ---------------------------------- #
#          Atomic functions          #
# ---------------------------------- #

@input_check(3, channel_specs='--g', reduction='any', reduction_range=2)
def blend(img_fg: Optional[th.Tensor] = None, img_bg: Optional[th.Tensor] = None,
          blend_mask: Optional[th.Tensor] = None, blending_mode: str = 'copy',
          alpha_blend: bool = True, cropping: List[float] = [0.0, 1.0, 0.0, 1.0],
          opacity: FloatValue = 1.0) -> th.Tensor:
    """Atomic node: Blend

    Args:
        img_fg (tensor, optional): Foreground image (G or RGB(A)). Defaults to None.
        img_bg (tensor, optional): Background image (G or RGB(A)). Defaults to None.
        blend_mask (tensor, optional): Blending alpha mask (G only). Defaults to None.
        blending_mode (str, optional): Color blending mode.
            copy | add | subtract | multiply | add_sub | max | min | divide | switch | overlay |
            screen | soft_light. Defaults to 'copy'.
        alpha_blend (bool, optional): Enable alpha blending for color inputs. Defaults to True.
        cropping (list, optional): Cropping mask for blended image ([left, right, top, bottom]).
            Defaults to [0.0, 1.0, 0.0, 1.0].
        opacity (float, optional): Alpha multiplier. Defaults to 1.0.

    Raises:
        ValueError: Unknown blending mode.

    Returns:
        Tensor: Blended image.
    """
    # Get foreground and background channels
    channels_fg = img_fg.shape[1] if img_fg is not None else 0
    channels_bg = img_bg.shape[1] if img_bg is not None else 0

    # Calculate blending weights
    opacity = to_tensor(opacity).clamp(0.0, 1.0)
    weight = blend_mask * opacity if blend_mask is not None else opacity

    # Empty inputs behave the same as zero
    zero = th.zeros([])

    # Switch mode: no alpha blending
    if blending_mode == 'switch':
        img_fg = img_fg if channels_fg else zero
        img_bg = img_bg if channels_bg else zero

        # Short-circuiting or linear interpolation
        opacity_const = to_const(opacity)
        if blend_mask is None and opacity_const in (0.0, 1.0):
            img_out = img_fg if opacity_const == 1.0 else img_bg
        else:
            img_out = th.lerp(img_bg, img_fg, weight)

    # For other modes, process RGB and alpha channels separately
    else:

        # Split the alpha channel
        use_alpha = max(channels_fg, channels_bg) == 4
        fg_alpha = img_fg[:,3:] if channels_fg == 4 else zero
        bg_alpha = img_bg[:,3:] if channels_bg == 4 else zero
        img_fg = zero if not channels_fg else img_fg[:,:3] if use_alpha else img_fg
        img_bg = zero if not channels_bg else img_bg[:,:3] if use_alpha else img_bg

        # Apply foreground alpha to blending weights
        weight = weight * fg_alpha if use_alpha else weight

        # Blend RGB channels in specified mode
        ## Copy mode
        if blending_mode == 'copy':
            img_out = th.lerp(img_bg, img_fg, weight)

        ## Add (linear dodge) mode
        elif blending_mode == 'add':
            img_out = img_fg * weight + img_bg

        ## Subtract mode
        elif blending_mode == 'subtract':
            img_out = img_bg - img_fg * weight

        ## Multiply mode
        elif blending_mode == 'multiply':
            img_out = th.lerp(img_bg, img_fg * img_bg, weight)

        ## Add Sub mode
        elif blending_mode == 'add_sub':
            img_out = (2.0 * img_fg - 1.0) * weight + img_bg

        ## Max (lighten) mode
        elif blending_mode == 'max':
            img_out = th.lerp(img_bg, th.max(img_fg, img_bg), weight)

        ## Min (darken) mode
        elif blending_mode == 'min':
            img_out = th.lerp(img_bg, th.min(img_fg, img_bg), weight)

        ## Divide mode
        elif blending_mode == 'divide':
            img_out = th.lerp(img_bg, img_bg / (img_fg + 1e-15), weight)

        ## Overlay mode
        elif blending_mode == 'overlay':
            img_below = 2.0 * img_fg * img_bg
            img_above = 1.0 - 2.0 * (1.0 - img_fg) * (1.0 - img_bg)
            img_fg = th.where(img_bg < 0.5, img_below, img_above)
            img_out = th.lerp(img_bg, img_fg, weight)

        ## Screen mode
        elif blending_mode == 'screen':
            img_fg = 1.0 - (1.0 - img_fg) * (1.0 - img_bg)
            img_out = th.lerp(img_bg, img_fg, weight)

        ## Soft light mode
        elif blending_mode == 'soft_light':
            interp_pos = th.lerp(img_bg, th.sqrt(img_bg + 1e-16), img_fg * 2 - 1.0)
            interp_neg = th.lerp(img_bg ** 2, img_bg, img_fg * 2)
            img_fg = th.where(img_fg > 0.5, interp_pos, interp_neg)
            img_out = th.lerp(img_bg, img_fg, weight)

        ## Unknown mode
        else:
            raise ValueError(f'Unknown blending mode: {blending_mode}')

        # Blend alpha channels
        if use_alpha:
            bg_alpha = bg_alpha if channels_bg else bg_alpha.expand_as(fg_alpha)
            out_alpha = th.lerp(bg_alpha, th.ones([]), weight) if alpha_blend else bg_alpha
            img_out = th.cat((img_out, out_alpha), dim=1)

    # Clamp the result to [0, 1]
    img_out = img_out.clamp(0.0, 1.0)

    # Apply cropping
    if list(cropping) == [0.0, 1.0, 0.0, 1.0]:
        img_out_crop = img_out
    else:
        start_row = math.floor(cropping[2] * img_out.shape[2])
        end_row = math.floor(cropping[3] * img_out.shape[2])
        start_col = math.floor(cropping[0] * img_out.shape[3])
        end_col = math.floor(cropping[1] * img_out.shape[3])

        img_out_crop = img_bg.expand_as(img_out).clone()
        img_out_crop[..., start_row:end_row, start_col:end_col] = \
            img_out[..., start_row:end_row, start_col:end_col]

    return img_out_crop


@input_check(1)
def blur(img_in: th.Tensor, intensity: FloatValue = 10.0) -> th.Tensor:
    """Atomic node: Blur (simple box blur)

    Args:
        img_in (tensor): Input image.
        intensity (float, optional): Box filter side length, defaults to 10.0.

    Returns:
        Tensor: Blurred image.
    """
    num_group, img_size = img_in.shape[1], img_in.shape[2]

    # Process input parameters
    intensity, intensity_const = to_tensor_and_const(intensity * img_size / 256.0)
    kernel_len = int(math.ceil(intensity_const + 0.5) * 2 - 1)
    if kernel_len <= 1:
        return img_in.clone()

    # Create 2D kernel
    kernel_rad = kernel_len // 2
    blur_idx = to_tensor([-abs(i) for i in range(-kernel_rad, kernel_rad + 1)])
    blur_1d = th.clamp(blur_idx + intensity + 0.5, 0.0, 1.0)
    blur_row = blur_1d.expand(num_group, 1, 1, -1)
    blur_col = blur_1d.unsqueeze(1).expand(num_group, 1, -1, 1)

    # Perform depth-wise convolution without implicit padding
    img_in = pad2d(img_in, (kernel_rad, kernel_rad))
    img_out = conv2d(img_in, blur_row, groups=num_group)
    img_out = conv2d(img_out, blur_col, groups=num_group)
    img_out = th.clamp(img_out / (intensity ** 2 * 4.0), 0.0, 1.0)

    return img_out


@input_check(2, reduction='any', reduction_range=2)
def channel_shuffle(img_in: Optional[th.Tensor] = None, img_in_aux: Optional[th.Tensor] = None,
                    use_alpha: bool = False, channel_r: int = 0, channel_g: int = 1,
                    channel_b: int = 2, channel_a: int = 3) -> th.Tensor:
    """Atomic node: Channel Shuffle

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        img_in_aux (tensor, optional): Auxiliary input image (G or RGB(A)) for swapping channels
            between images. Defaults to None.
        use_alpha (bool, optional): Enable the alpha channel in output. Defaults to False.
        channel_r (int, optional): Red channel index from source images. Defaults to 0.
        channel_g (int, optional): Green channel index from source images. Defaults to 1.
        channel_b (int, optional): Blue channel index from source images. Defaults to 2.
        channel_a (int, optional): Alpha channel index from source images. Defaults to 3.

    Raises:
        ValueError: Shuffle index is out of bound or invalid.

    Returns:
        Tensor: Channel shuffled images.
    """
    # Assemble channel shuffle indices
    num_channels = 4 if use_alpha else 3
    shuffle_idx = [channel_r, channel_g, channel_b, channel_a][:num_channels]

    # Convert grayscale inputs to color
    if img_in is not None and img_in.shape[1] == 1:
        img_in = img_in.expand(-1, num_channels, -1, -1)
    if img_in_aux is not None and img_in_aux.shape[1] == 1:
        img_in_aux = img_in_aux.expand(-1, num_channels, -1, -1)

    # Output is identical to the first input by default
    img_out = img_in.clone()

    # Copy channels from source images to the output using assembled indices
    for i, idx in filter(lambda x: x[0] != x[1], enumerate(shuffle_idx)):
        if idx >= 0 and idx <= 3:
            source_img = img_in
        elif idx >= 4 and idx <= 7:
            source_img = img_in_aux
            idx -= 4
        else:
            raise ValueError(f'Invalid shuffle index: {shuffle_idx}')

        if source_img is not None and idx < source_img.shape[1]:
            img_out[:, i] = source_img[:, idx]

    return img_out


@input_check(1)
def curve(img_in: th.Tensor, anchors: Optional[FloatArray] = None) -> th.Tensor:
    """Atomic node: Curve

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        anchors (list or tensor, optional): Piece-wise Bezier curve anchors. Defaults to None.

    Raises:
        ValueError: Input anchor array has invalid shape.

    Returns:
        Tensor: Curve-mapped image.

    TODO:
        - Support per channel (including alpha) adjustment
    """
    # Split the alpha channel from the input image
    img_in, img_in_alpha = img_in.split(3, dim=1) if img_in.shape[1] == 4 else (img_in, None)

    # When the anchors are not provided, this node simply passes the input image
    if anchors is None:
        return img_in

    # Process input anchor table
    anchors = to_tensor(anchors)
    num_anchors = anchors.shape[0]
    if anchors.shape != (num_anchors, 6):
        raise ValueError(f'Invalid anchors shape: {list(anchors.shape)}')

    # Sort input anchors in ascendng X position order
    anchors = anchors[th.argsort(anchors[:, 0])]

    # Determine the size of the sample grid
    res_h, res_w = img_in.shape[2], img_in.shape[3]
    sample_size_t = max(res_h, res_w) * 2
    sample_size_x = sample_size_t

    # First sampling pass (parameter space)
    t = th.linspace(0.0, 1.0, sample_size_t)
    int_idx = th.sum((t.unsqueeze(1) >= anchors.select(1, 0)), 1)
    pre_mask, app_mask = int_idx == 0, int_idx == num_anchors
    int_idx = int_idx.clamp(1, num_anchors - 1) - 1

    anchor_pairs = th.stack((anchors[:-1], anchors[1:]), dim=1)[int_idx]
    p1 = anchor_pairs.select(1, 0).narrow(1, 0, 2).T
    p2 = anchor_pairs.select(1, 0).narrow(1, 4, 2).T
    p3 = anchor_pairs.select(1, 1).narrow(1, 2, 2).T
    p4 = anchor_pairs.select(1, 1).narrow(1, 0, 2).T
    A = p4 - p1 + (p2 - p3) * 3.0
    B = (p1 + p3 - p2 * 2.0) * 3.0
    C = (p2 - p1) * 3.0
    D = p1

    t_ = (t - p1[0]) / (p4[0] - p1[0]).clamp_min(1e-8)
    bz_t = (((A * t_) + B) * t_ + C) * t_ + D
    bz_t[0] = th.where(pre_mask | app_mask, t, bz_t[0])
    bz_t[1] = th.where(pre_mask, anchors[0, 1], bz_t[1])
    bz_t[1] = th.where(app_mask, anchors[-1, 1], bz_t[1])

    # Second sampling pass (x space)
    x = th.linspace(0.0, 1.0, sample_size_x)
    int_idx = th.sum((x.unsqueeze(1) >= bz_t[0]), 1)
    int_idx = int_idx.clamp(1, sample_size_t - 1) - 1

    bz_t_pairs = th.stack((bz_t[:,:-1], bz_t[:,1:])).index_select(2, int_idx)
    x_ = (x - bz_t_pairs[0, 0]) / (bz_t_pairs[1, 0] - bz_t_pairs[0, 0]).clamp_min(1e-8)
    bz_y = th.lerp(bz_t_pairs[0, 1], bz_t_pairs[1, 1], x_)

    # Third sampling pass (color space)
    bz_y = bz_y.expand(img_in.shape[0] * img_in.shape[1], 1, 1, sample_size_x)
    col_grid = img_in.view(img_in.shape[0] * img_in.shape[1], res_h, res_w, 1) * 2.0 - 1.0
    sample_grid = th.cat([col_grid, th.zeros_like(col_grid)], 3)
    img_out = grid_sample_impl(bz_y, sample_grid, align_corners=True)
    img_out = img_out.clamp(0, 1).view_as(img_in)

    # Append the original alpha channel
    if img_in_alpha is not None:
        img_out = th.cat([img_out, img_in_alpha], dim=1)

    return img_out


@input_check(1)
def d_blur(img_in: th.Tensor, intensity: FloatValue = 10.0, angle: FloatValue = 0.0) -> th.Tensor:
    """Atomic node: Directional Blur

    Args:
        img_in (tensor): Input image.
        intensity (float, optional): Filter length. Defaults to 10.0.
        angle (float, optional): Filter angle. Defaults to 0.0.

    Returns:
        Tensor: Directional blurred image.
    """
    num_group, num_row, num_col = img_in.shape[1], img_in.shape[2], img_in.shape[3]

    # No blur effect when intensity is very small
    intensity, intensity_const = to_tensor_and_const(intensity * num_row / 256)
    if intensity_const <= 0.5:
        return img_in.clone()

    # Wrap the angle within [0, pi/4]
    angle, angle_const = to_tensor_and_const(angle % 0.5)
    vertical = angle_const >= 0.125 and angle_const < 0.375
    invert_offset = angle_const > 0.25

    angle = angle - 0.5 if angle_const >= 0.375 else angle
    angle = 0.25 - angle if vertical else angle
    angle = th.abs(angle) * (np.pi * 2.0)

    # Compute horizontal kernel weights
    cos = th.cos(angle)
    intensity_x, intensity_x_const = to_tensor_and_const((intensity - 0.5) * cos)
    kernel_len = int(math.ceil(intensity_x_const) * 2 + 1)
    kernel_rad = kernel_len >> 1
    kernel_idx = to_tensor([-abs(i) for i in range(-kernel_rad, kernel_rad + 1)])
    kernel_weights = th.clamp(kernel_idx + intensity_x + 1, 0.0, 1.0)

    # Special case: angle is 0 or 90 degrees
    if min(abs(angle_const - val) for val in (0.0, 0.25, 0.5)) < 1e-8:

        # Normalize kernel weights
        kernel_weights = kernel_weights / kernel_weights.sum()
        if vertical:
            kernel = kernel_weights.view(-1, 1).expand(num_group, 1, -1, -1)
        else:
            kernel = kernel_weights.expand(num_group, 1, 1, -1)

        # Perform convolution
        img_in = pad2d(img_in, (kernel_rad, 0) if vertical else (0, kernel_rad))
        img_out = conv2d(img_in, kernel, groups=num_group)
        img_out = th.clamp(img_out, 0.0, 1.0)

    # Compute directional motion blur in different algorithms
    # Special condition (3x3 kernel) when intensity is small
    elif intensity_x_const <= 1.0:

        # Construct the kernel using trigonometrics
        tan = th.tan(angle)
        kernel_2d = th.zeros(9)
        kernel_2d[[0, 8]] = tan
        kernel_2d[[3, 5]] = 1 - tan
        kernel_2d[4] = 1.0

        # Apply kernel weights and normalize the kernel
        kernel_2d = kernel_2d.view(3, 3) * kernel_weights
        kernel_2d = kernel_2d / kernel_2d.sum()

        # Account for the other angle ranges
        kernel_2d = th.flipud(kernel_2d) if invert_offset else kernel_2d
        kernel_2d = kernel_2d.T if vertical else kernel_2d
        kernel = kernel_2d.expand(num_group, 1, -1, -1)

        # Perform 3x3 convolution
        img_in = pad2d(img_in, 1)
        img_out = conv2d(img_in, kernel, groups=num_group)
        img_out = th.clamp(img_out, 0.0, 1.0)

    # The other cases require deformable convolution since the blur kernel is 'angled'
    else:

        # Compute a horizontal 3xN kernel from linear gradient interpolation
        tan = th.tan(angle)
        kernel_x = th.linspace(-kernel_rad + 0.5, kernel_rad + 0.5, kernel_len) * tan
        gradient_left = th.stack((tan, 2 - tan, th.zeros([]))).view(3, 1)
        gradient_right = gradient_left.roll(1, 0)
        kernel_2d = th.lerp(gradient_left, gradient_right, kernel_x % 1)
        kernel_2d = kernel_2d / (tan * (tan - 2) + 2)

        # Apply kernel weights and normalize the kernel
        kernel_2d = kernel_2d * kernel_weights
        kernel_2d = kernel_2d / kernel_2d.sum()

        # Account for other angle ranges
        kernel_2d = th.flipud(kernel_2d) if invert_offset else kernel_2d
        kernel_2d = kernel_2d.T if vertical else kernel_2d
        kernel = kernel_2d.expand(num_group, 1, -1, -1)

        # Compute offset for pixel coordinates (for deformable convolution)
        offset_row = -th.floor(kernel_x) if invert_offset else th.floor(kernel_x)
        offset_col = th.zeros_like(offset_row)
        if vertical:
            offset = th.stack((offset_col, offset_row), dim=1).expand(3, -1, -1).transpose(0, 1)
        else:
            offset = th.stack((offset_row, offset_col), dim=1).expand(3, -1, -1)

        # Expand the offset matrix to the target image size
        batch_size = img_in.shape[0]
        out_size_row = img_in.shape[2] - kernel.shape[2] + kernel_rad * 2 + 3
        out_size_col = img_in.shape[3] - kernel.shape[3] + kernel_rad * 2 + 3
        offset = offset.reshape(-1, 1, 1).expand(batch_size, -1, out_size_row, out_size_col)

        # Run deformable convolution and crop the output image back to original size
        img_in = pad2d(img_in, kernel_rad + 1)
        img_out = deform_conv2d(img_in, offset, kernel)
        row_s, col_s = (out_size_row - num_row) >> 1, (out_size_col - num_col) >> 1
        img_out_crop = img_out.narrow(2, row_s, num_row).narrow(3, col_s, num_col)
        img_out = th.clamp(img_out_crop, 0.0, 1.0)

    return img_out


@input_check(2, channel_specs='.g')
def d_warp(img_in: th.Tensor, intensity_mask: th.Tensor, intensity: FloatValue = 10.0,
           angle: FloatValue = 0.0) -> th.Tensor:
    """Atomic node: Directional Warp

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        intensity_mask (tensor): Intensity mask for computing displacement (G only).
        intensity (float, optional): Intensity multiplier. Defaults to 10.0.
        angle (float, optional): Direction to shift (in turns), 0 degree points to the left.
            Defaults to 0.0.

    Returns:
        Tensor: Directionally warped image.
    """
    # Convert parameters to tensors
    angle = to_tensor(angle) * (math.pi * 2.0)

    # Compute shifted image sampling grid
    sample_grid = get_pos(*img_in.shape[2:])
    vec_shift = th.stack([th.cos(angle), th.sin(angle)]) * (intensity / 256)
    sample_grid = sample_grid + intensity_mask.movedim(1, 3) * vec_shift

    # Perform sampling
    img_out = grid_sample(img_in, sample_grid, sbs_format=True)

    return img_out


@input_check(2, channel_specs='g.')
def distance(img_mask: th.Tensor, img_source: Optional[th.Tensor] = None, mode: str = 'gray',
             combine: bool = True, use_alpha: bool = False, dist: FloatValue = 10.0) -> th.Tensor:
    """Atomic node: Distance

    Args:
        img_mask (tensor): A mask image to be binarized by a threshold of 0.5 (G only).
        img_source (tensor, optional): Input colors to be fetched using `img_mask` (G or RGB(A)).
            Defaults to None.
        mode (str, optional): 'gray' or 'color', determine the format of output when `img_source`
            is not provided. Defaults to 'gray'.
        combine (bool, optional): Blend output and source colors. Defaults to True.
        use_alpha (bool, optional): Enable the alpha channel in output. Defaults to False.
        dist (float, optional): Propagation distance (Euclidean distance). Defaults to 10.0.

    Returns:
        Tensor: Distance transformed image.
    """
    # Check input validity
    check_arg_choice(mode, ['gray', 'color'], arg_name='mode')

    # Rescale distance
    num_rows, num_cols = img_mask.shape[2], img_mask.shape[3]
    dist, dist_const = to_tensor_and_const(dist * num_rows / 256)

    # Alpha channel from the source image is required in 'combine' mode
    if combine and img_source is not None and img_source.shape[1] == 3:
        img_source = resize_image_color(img_source, 4)

    # Special cases for small distances (no distance transform is needed)
    num_channels = 1 if mode == 'gray' else 3 if not use_alpha else 4

    if dist_const <= 1.0:

        # Quantize the mask into binary values
        img_mask = (img_mask > 0.5).float() if dist_const > 0.0 else th.zeros_like(img_mask)

        if img_source is None:  # No source
            img_out = img_mask.expand(-1, num_channels, -1, -1)
        elif not combine:  # Source only
            img_out = img_source
        elif img_source.shape[1] == 1:  # Grayscale source
            img_out = img_source * img_mask
        else:  # Color source
            img_out = img_source.clone()
            img_out[:,3] *= img_mask.squeeze(1)

        return img_out

    # Initialize output image
    img_out = th.zeros(img_mask.shape[0], num_channels, num_rows, num_cols)

    # Quantize the mask into binary values (the mask is inverted for SciPy API)
    inv_binary_mask = img_mask <= 0.5

    # Pre-pad the binary mask to account for tiling
    pad_dist = int(np.ceil(dist_const)) + 1
    pr, pc = tuple(min(n // 2, pad_dist) for n in (num_rows, num_cols))
    inv_binary_mask = pad2d(inv_binary_mask, (pr, pc))

    pad_size = to_tensor([pc, pr])
    img_size = to_tensor([num_cols, num_rows])

    # Loop through mini-batch
    for i, mask in enumerate(inv_binary_mask.unbind()):

        # Calculate Euclidean distance transform using the binary mask
        binary_mask_np = mask.detach().squeeze(0).cpu().numpy()
        dist_arr, indices = \
            distance_transform_edt(binary_mask_np, return_distances=True, return_indices=True)

        # Remove padding
        dist_arr = dist_arr[pr:pr+num_rows, pc:pc+num_cols].astype(np.float32)
        indices = indices[::-1, pr:pr+num_rows, pc:pc+num_cols].astype(np.float32)

        # Convert SciPy distance output to image gradient
        dist_mat = to_tensor(dist_arr).expand(1, 1, -1, -1)
        dist_weights = th.clamp(1.0 - dist_mat / dist, 0.0, 1.0)

        # No source, apply the gradient directly
        if img_source is None:
            img_out[i] = dist_weights
            continue

        # Normalize SciPy indices output to screen coordinates
        sample_grid = to_tensor(indices).movedim(0, 2).unsqueeze(0)
        sample_grid = ((sample_grid - pad_size) % img_size + 0.5) / img_size * 2.0 - 1.0

        # Sample the source image using normalized coordinates
        img_edt = grid_sample_impl(
            img_source[i].unsqueeze(0), sample_grid, mode='nearest', align_corners=False)

        # Combine source and transformed images
        if not combine:
            img_edt = th.where(dist_mat >= dist, img_source, img_edt)
        elif img_source.shape[1] == 1:
            img_edt = img_edt * dist_weights
        else:
            img_edt[:,3] *= dist_weights.squeeze(1)

        img_out[i] = img_edt.clamp(0.0, 1.0)

    return img_out


@input_check(2, channel_specs='.g')
def emboss(img_in: th.Tensor, height_map: th.Tensor, intensity: FloatValue = 5.0,
           light_angle: FloatValue = 0.0, highlight_color: FloatVector = [1.0] * 4,
           shadow_color: FloatVector = [0.0] * 4) -> th.Tensor:
    """Atomic node: Emboss

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        height_map (tensor): Height map (G only).
        intensity (float, optional): Height map multiplier. Defaults to 5.0.
        light_angle (float, optional): Light angle (in turns). Defaults to 0.0.
        highlight_color (list, optional): Highlight color. Defaults to [1.0, 1.0, 1.0, 1.0].
        shadow_color (list, optional): Shadow color. Defaults to [0.0, 0.0, 0.0, 0.0].

    Returns:
        Tensor: Embossed image.
    """
    # Split the alpha channel from the input image
    img_in, img_in_alpha = img_in.split(3, dim=1) if img_in.shape[1] == 4 else (img_in, None)

    # Process input parameters
    num_channels, num_rows = img_in.shape[1], img_in.shape[2]
    intensity = to_tensor(intensity) * (num_rows / 256)
    light_angle = to_tensor(light_angle) * (math.pi * 2.0)

    highlight_color = to_tensor(highlight_color[:num_channels]).clamp(0.0, 1.0).view(-1, 1, 1)
    shadow_color = to_tensor(shadow_color[:num_channels]).clamp(0.0, 1.0).view(-1, 1, 1)

    # Compute emboss intensity vector map
    dx, dy = height_map - height_map.roll(1, 3), height_map - height_map.roll(1, 2)
    delta = th.stack((dx, dy), dim=1)
    vec_emboss = th.stack((th.cos(light_angle), th.sin(light_angle))).view(2,1,1,1)
    intensity = intensity * vec_emboss * th.abs(delta)

    # Apply light and shadow colors
    light_mask = (delta >= 0) == (intensity >= 0)
    color_offset = th.where(light_mask, highlight_color, shadow_color - 1)
    img_out = th.clamp(img_in + (th.abs(intensity) * color_offset).sum(dim=1), 0.0, 1.0)

    # Append the original alpha channel
    if img_in_alpha is not None:
        img_out = th.cat([img_out, img_in_alpha], dim=1)

    return img_out


@input_check(1, channel_specs='g')
def gradient_map(img_in: th.Tensor, mode: str = 'color', linear_interp: bool = True,
                 use_alpha: bool = False, anchors: Optional[FloatArray] = None) -> th.Tensor:
    """Atomic node: Gradient Map

    Args:
        img_in (tensor): Input image (G only).
        mode (str, optional): 'color' or 'gray'. Defaults to 'color'.
        linear_interp (bool, optional): Use linear interpolation when set to True; use cubic
            interpolation with flat tangents when set to False. Defaults to True.
        use_alpha (bool, optional): Enable the alpha channel in output. Defaults to False.
        anchors (list or tensor, optional): Gradient anchors. Defaults to None.

    Returns:
        Tensor: Gradient map image.
    """
    # Check input validity
    check_arg_choice(mode, ['gray', 'color'], arg_name='mode')

    # When anchors are not provided, the node is simply used for grayscale-to-color conversion
    num_col = 2 if mode == 'gray' else 4 + use_alpha
    if anchors is None:
        return resize_image_color(img_in, num_col - 1)

    # Process anchor parameters by converting them to the correct number of channels and sorting
    # in ascending position order
    anchors = to_tensor(anchors)
    num_anchors = anchors.shape[0]
    anchors = resize_anchor_color(anchors, num_col - 1)
    anchors = anchors[th.argsort(anchors[:, 0])]

    # Compute anchor interval index
    anchor_idx = (img_in >= anchors[:, 0].view(-1, 1, 1)).sum(dim=1, keepdim=True)
    pre_mask = (anchor_idx == 0).expand(img_in.shape[0], num_col - 1, -1, -1)
    app_mask = (anchor_idx == num_anchors).expand(img_in.shape[0], num_col - 1, -1, -1)

    # Make sure every position gets a valid anchor index
    anchor_idx = anchor_idx.sub_(1).clamp_(0, num_anchors - 2)

    # Extract anchor pairs for each pixel
    anchor_idx_pairs = th.cat((anchor_idx, anchor_idx + 1), dim=1).unsqueeze(-1)
    grid_x = (anchor_idx_pairs + 0.5) / (num_anchors * 0.5) - 1
    grid = th.cat((grid_x, th.zeros(*grid_x.shape[:-1], 2)), dim=-1)
    grid = grid.expand(img_in.shape[0], *grid.shape[1:])

    img_anchors = anchors.T.view(1, -1, 1, 1, num_anchors)
    img_at_anchor_pairs = grid_sample_impl(img_anchors, grid, mode='nearest', align_corners=False)

    # Perform interpolation
    img_at_anchor, img_at_next = img_at_anchor_pairs.unbind(dim=2)
    img_at_anchor_pos, img_at_anchor_val = img_at_anchor.tensor_split([1], dim=1)
    img_at_next_pos, img_at_next_val = img_at_next.tensor_split([1], dim=1)

    a = (img_in - img_at_anchor_pos) / (img_at_next_pos - img_at_anchor_pos).clamp_min(1e-12)
    a = a if linear_interp else a ** 2 * (3 - a * 2)
    img_out = th.lerp(img_at_anchor_val, img_at_next_val, a)

    # Consider pixels that do not fall into any interpolation interval
    img_out = th.where(pre_mask, anchors[0, 1:].view(-1, 1, 1), img_out)
    img_out = th.where(app_mask, anchors[-1, 1:].view(-1, 1, 1), img_out)
    img_out = img_out.clamp(0.0, 1.0)

    return img_out


@input_check(2, channel_specs='g.')
def gradient_map_dyn(img_in: th.Tensor, img_gradient: th.Tensor, orientation: str = 'horizontal',
                     position: FloatValue = 0.0) -> th.Tensor:
    """Atomic node: Gradient Map (Dynamic)

    Args:
        img_in (tensor): Input image (G only).
        img_gradient (tensor): Gradient image (G or RGB(A)).
        orientation (str, optional): 'vertical' or 'horizontal', sampling direction.
            Defaults to 'horizontal'.
        position (float, optional): Normalized position to sample. Defaults to 0.0.

    Returns:
        Tensor: Gradient map image.
    """
    # Check input validity
    check_arg_choice(orientation, ['horizontal', 'vertical'], arg_name='orientation')

    # Convert parameters to tensors
    position = to_tensor(position).clamp(0.0, 1.0)

    # Construct sampling grid coordinates using the input image
    img_in_perm = img_in.movedim(1, 3)
    x_grid = (position * 2.0 - 1.0).expand_as(img_in_perm)
    y_grid = img_in_perm * 2.0 - 1.0
    x_grid, y_grid = (x_grid, y_grid) if orientation == 'vertical' else (y_grid, x_grid)
    sample_grid = th.cat([x_grid, y_grid], dim=3)

    # Get sampled image as output
    img_out = grid_sample_impl(img_gradient, sample_grid, align_corners=True)

    return img_out


@input_check(1, channel_specs='c')
def c2g(img_in: th.Tensor, flatten_alpha: bool = False,
        rgba_weights: FloatVector = [0.33, 0.33, 0.33, 0.0], bg: FloatValue = 1.0) -> th.Tensor:
    """Atomic function: Grayscale Conversion

    Args:
        img_in (tensor): Input image (RGB(A) only).
        flatten_alpha (bool, optional): Set the behaviour of alpha on the final grayscale image.
            Defaults to False (no effect).
        rgba_weights (list, optional): RGBA combination weights.
            Defaults to [0.33, 0.33, 0.33, 0.0].
        bg (float, optional): Uniform background color. Defaults to 1.0.

    Returns:
        Tensor: Grayscale converted image.
    """
    # Compute grayscale output by averaging input color channels
    num_channels = img_in.shape[1]
    rgba_weights = to_tensor(rgba_weights[:num_channels])
    img_out = (img_in * rgba_weights.view(num_channels, 1, 1)).sum(dim=1, keepdim=True)

    # Optionally use the alpha channel to blend into the background color
    if flatten_alpha and num_channels == 4:
        img_out = th.lerp(to_tensor(bg).clamp(0.0, 1.0), img_out, img_in[:,3:])

    # Clamp the output within [0, 1]
    img_out = img_out.clamp(0.0, 1.0)

    return img_out


@input_check(1, channel_specs='c')
def hsl(img_in: th.Tensor, hue: FloatValue = 0.5, saturation: FloatValue = 0.5,
        lightness: FloatValue = 0.5) -> th.Tensor:
    """Atomic node: HSL

    Args:
        img_in (tensor): Input image (RGB(A) only).
        hue (float, optional): Hue adjustment value. Defaults to 0.5.
        saturation (float, optional): Saturation adjustment value. Defaults to 0.5.
        lightness (float, optional): Lightness adjustment value. Defaults to 0.5.

    Returns:
        Tensor: HSL adjusted image.
    """
    # Split the alpha channel from the input image
    img_in, img_in_alpha = img_in.split(3, dim=1) if img_in.shape[1] == 4 else (img_in, None)

    # Convert input image from RGB to HSL
    ## Compute lightness
    max_vals, max_idx = th.max(img_in, 1, keepdim=True)
    min_vals, _ = th.min(img_in, 1, keepdim=True)
    delta, l = max_vals - min_vals, (max_vals + min_vals) * 0.5

    ## Compute saturation
    s = delta / (1.0 - th.abs(2 * l - 1) + 1e-8)

    ## Compute hue
    h_vol = (img_in.roll(-1, 1) - img_in.roll(1, 1)) / (delta + 1e-8)
    h_vol = (h_vol + th.linspace(0, 4, 3).view(-1, 1, 1)) % 6 / 6
    h = h_vol.take_along_dim(max_idx, 1)

    # Adjust HSL
    h = (h + hue * 2.0 - 1.0) % 1.0
    s = th.clamp(s + saturation * 2.0 - 1.0, 0.0, 1.0)
    l = th.clamp(l + lightness * 2.0 - 1.0, 0.0, 1.0)

    # Convert HSL back to RGB
    _t: Callable[[Any], th.Tensor] = lambda d: to_tensor(d).view(-1, 1, 1)
    c = (1.0 - th.abs(2.0 * l - 1.0)) * s
    w = (h - _t([0.5, 1/3, 2/3])).abs() * _t([6, -6, -6]) + _t([-1, 2, 2])
    img_out = (l + c * (w.clamp(0.0, 1.0) - 0.5)).clamp(0.0, 1.0)

    # Append original alpha channel
    if img_in_alpha is not None:
        img_out = th.cat([img_out, img_in_alpha], dim=1)

    return img_out


@input_check(1)
def levels(img_in: th.Tensor, in_low: FloatVector = 0.0, in_mid: FloatVector = 0.5,
           in_high: FloatVector = 1.0, out_low: FloatVector = 0.0,
           out_high: FloatVector = 1.0) -> th.Tensor:
    """Atomic node: Levels

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        in_low (float or list, optional): Low cutoff for input. Defaults to 0.0.
        in_mid (float or list, optional): Middle point for calculating gamma correction.
            Defaults to 0.5.
        in_high (float or list, optional): High cutoff for input. Defaults to 1.0.
        out_low (float or list, optional): Low cutoff for output. Defaults to 0.0.
        out_high (float or list, optional): High cutoff for output. Defaults to 1.0.

    Returns:
        Tensor: Level adjusted image.
    """
    # Resize parameters to fit the number of input channels
    num_channels = img_in.shape[1]

    def param_process(param_in: Union[float, FloatVector], default_val: float) -> th.Tensor:
        param_in = th.atleast_1d(to_tensor(param_in)).clamp(0.0, 1.0)
        return resize_color(param_in, num_channels, default_val=default_val).view(-1, 1, 1)

    in_low = param_process(in_low, 0.0)
    in_mid = param_process(in_mid, 0.5)
    in_high = param_process(in_high, 1.0)
    out_low = param_process(out_low, 0.0)
    out_high = param_process(out_high, 1.0)

    # Determine left, mid, right
    invert_mask = in_low > in_high
    img_in = th.where(invert_mask, 1.0 - img_in, img_in)
    left = th.where(invert_mask, 1.0 - in_low, in_low)
    right = th.where(invert_mask, 1.0 - in_high, in_high).clamp_min(left + 1e-4)
    mid = in_mid

    # Gamma correction
    gamma_corr = (th.abs(mid * 2.0 - 1.0) * 8.0 + 1.0).clamp_max(9.0)
    gamma_corr = th.where(mid < 0.5, 1.0 / gamma_corr, gamma_corr)
    img = (img_in.clamp(left, right) - left) / (right - left)
    img = (img + 1e-15) ** gamma_corr

    # Output linear mapping
    img_out = th.lerp(out_low, out_high, img.clamp(0.0, 1.0)).clamp(0.0, 1.0)

    return img_out


@input_check(2, channel_specs='.g')
def motion_blur(img_in: th.Tensor, img_grad: th.Tensor, intensity: FloatValue = 1.0,
                angle: FloatValue = 0.0) -> th.Tensor:
    """Atomic node: Motion Blur (obsolete)
    """
    # Scale intensity according to image size
    H, W = img_in.shape[2:]
    intensity = intensity * to_tensor([v / (256 * 3) for v in (W, H)])

    # Compute displacement vector field
    vec_shift = img_grad - th.cat((img_grad.roll(1, 3), img_grad.roll(1, 2)), dim=1)
    vec_shift = vec_shift.movedim(1, 3) * intensity

    # Rotate the vector field
    angle = to_tensor(angle * (math.pi * 2.0))
    cos_angle, sin_angle = th.cos(angle), th.sin(angle)
    R = th.stack([cos_angle, -sin_angle, sin_angle, cos_angle]).view(2, 2)
    vec_shift = (vec_shift.unsqueeze(-2) * R).sum(dim=-1)

    # Perform sampling to obtain the warped image
    sample_grid = get_pos(H, W) + vec_shift * th.linspace(-2.5, 2.5, 6).view(-1, 1, 1, 1)
    sample_grid = (sample_grid % 1 * 2 - 1) * to_tensor([W / (W + 2), H / (H + 2)])
    in_pad = pad2d(img_in, 1).expand(6, -1, -1, -1)
    img_out = grid_sample_impl(in_pad, sample_grid, align_corners=False).mean(dim=0, keepdim=True)

    return img_out.clamp(0.0, 1.0)


@input_check(1, channel_specs='g')
def normal(img_in: th.Tensor, mode: str = 'tangent_space', normal_format: str = 'dx',
           use_input_alpha: bool = False, use_alpha: bool = False, intensity: FloatValue = 1.0) \
            -> th.Tensor:
    """Atomic node: Normal

    Args:
        img_in (tensor): Input image (G only).
        mode (str, optional): 'tangent space' or 'object space'. Defaults to 'tangent_space'.
        normal_format (str, optional): 'gl' or 'dx'. Defaults to 'dx'.
        use_input_alpha (bool, optional): Use input image as alpha output. Defaults to False.
        use_alpha (bool, optional): Enable the alpha channel. Defaults to False.
        intensity (float, optional): Height map multiplier on dx, dy. Defaults to 1.0.

    Returns:
        Tensor: Normal image.
    """
    # Check input validity
    check_arg_choice(mode, ['tangent_space', 'object_space'], arg_name='mode')
    check_arg_choice(normal_format, ['dx', 'gl'], arg_name='normal_format')

    # Convert parameters to tensors
    intensity = to_tensor(intensity) * (img_in.shape[2] / 256)

    # Compute image gradient
    dx = th.roll(img_in, 1, 3) - img_in
    dy = th.roll(img_in, 1, 2) - img_in
    dy = dy if normal_format == 'dx' else -dy

    # Derive normal map from image gradient
    img_out = th.cat((intensity * dx, intensity * dy, th.ones_like(dx)), 1)
    img_out = img_out / img_out.norm(dim=1, keepdim=True)
    img_out = img_out / 2.0 + 0.5 if mode == 'tangent_space' else img_out

    # Attach an alpha channel to output if enabled
    if use_alpha == True:
        img_out_alpha = img_in if use_input_alpha else th.ones_like(img_in)
        img_out = th.cat([img_out, img_out_alpha], dim=1)

    return img_out


@input_check(1)
def sharpen(img_in: th.Tensor, intensity: FloatValue = 1.0) -> th.Tensor:
    """Atomic function: Sharpen (unsharp mask)

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        intensity (float, optional): Unsharp mask multiplier. Defaults to 1.0.

    Returns:
        Tensor: Sharpened image.
    """
    # Construct unsharp mask kernel
    num_group = img_in.shape[1]
    kernel = to_tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).expand(num_group, 1, -1, -1)
    unsharp_mask = conv2d(pad2d(img_in, 1), kernel, groups=num_group)

    # Adjust the sharpening effect on input
    img_out = th.clamp(img_in + unsharp_mask * intensity, 0.0, 1.0)

    return img_out


@input_check(1)
def transform_2d(img_in: th.Tensor, tiling: int = 3, sample_mode: str = 'bilinear',
                 mipmap_mode: str = 'auto', mipmap_level: int = 0,
                 matrix22: FloatVector = [1.0, 0.0, 0.0, 1.0], offset: FloatVector = [0.0, 0.0],
                 matte_color: FloatVector = [0.0, 0.0, 0.0, 1.0]) -> th.Tensor:
    """Atomic node: Transformation 2D

    Args:
        img_in (tensor): input image
        tiling (int, optional): tiling mode.
            0 = no tile,
            1 = horizontal tile,
            2 = vertical tile,
            3 = horizontal and vertical tile. Defaults to 3.
        sample_mode (str, optional): 'bilinear' or 'nearest'. Defaults to 'bilinear'.
        mipmap_mode (str, optional): 'auto' or 'manual'. Defaults to 'auto'.
        mipmap_level (int, optional): Manual mipmap level. Defaults to 0.
        matrix22: transformation matrix, default to [1.0, 0.0, 0.0, 1.0].
        offset: translation offset, default to [0.0, 0.0].
        matte_color (list, optional): background color. Defaults to [0.0, 0.0, 0.0, 1.0].

    Returns:
        Tensor: Transformed image.
    """
    # Check input validity
    check_arg_choice(sample_mode, ['bilinear', 'nearest'], arg_name='sample_mode')
    check_arg_choice(mipmap_mode, ['auto', 'manual'], arg_name='mipmap_mode')

    # Convert parameters to tensors
    mm_level = mipmap_level
    matrix22, (x1, x2, y1, y2) = to_tensor_and_const(matrix22)
    offset, (x_offset, y_offset) = to_tensor_and_const(offset)
    matte_color = th.atleast_1d(to_tensor(matte_color)).clamp(0.0, 1.0)
    matte_color = resize_color(matte_color, img_in.shape[1])

    # Offload automatic mipmap level computation to CPU for speed-up
    if mipmap_mode == 'auto':
        if abs(x1 * y2 - x2 * y1) < 1e-6:
            logger.warn('Singular transformation matrix may lead to unexpected behaviors')

        # Deduce mipmap level from transformation matrix
        inv_h1 = math.sqrt(x2 * x2 + y2 * y2)
        inv_h2 = math.sqrt(x1 * x1 + y1 * y1)
        max_compress_ratio = max(inv_h1, inv_h2)
        mm_level = sum(max_compress_ratio + 1e-8 >= 2 ** (i + 0.5) for i in range(12))

        # Special cases (scaling only, no rotation or shear)
        if abs(x1) == abs(y2) and x2 == 0 and y1 == 0 and math.log2(abs(x1)).is_integer() or \
           abs(x2) == abs(y1) and x1 == 0 and y2 == 0 and math.log2(abs(x2)).is_integer():
            scale = max(abs(x1), abs(x2))
            if (x_offset * scale).is_integer() and (y_offset * scale).is_integer():
                mm_level = max(0, mm_level - 1)

    # Mipmapping (optional)
    if mm_level > 0:
        mm_level = min(mm_level, int(math.log2(img_in.shape[2])))
        img_mm = automatic_resize(img_in, -mm_level)
    else:
        img_mm = img_in

    # Subtract background color from the image if tiling is not full
    if tiling < 3:
        img_mm = img_mm - matte_color.view(-1, 1, 1)

    # Compute 2D transformation
    theta = th.cat((matrix22.view(2, 2), offset.view(1, -1) * 2.0), dim=0).T
    theta = theta.expand(img_in.shape[0], 2, 3)
    sample_grid = affine_grid(theta, img_in.shape, align_corners=False)
    img_out = grid_sample(img_mm, sample_grid, mode=sample_mode, tiling=tiling)

    # Add the background color back after sampling
    if tiling < 3:
        img_out = (img_out + matte_color.view(-1, 1, 1)).clamp(0.0, 1.0)

    return img_out


def uniform_color(mode: str = 'color', num_imgs: int = 1, res_h: int = 512, res_w: int = 512,
                  use_alpha: bool = False, rgba: FloatVector = [0.0, 0.0, 0.0, 1.0]) -> th.Tensor:
    """Atomic node: Uniform Color

    Args:
        mode (str, optional): Output image type ('gray' or 'color'). Defaults to 'color'.
        num_imgs (int, optional): Number of images, i.e., batch size. Defaults to 1.
        res_h (int, optional): Image height. Defaults to 512.
        res_w (int, optional): Image width. Defaults to 512.
        use_alpha (bool, optional): Enable the alpha channel in output. Defaults to False.
        rgba (list, optional): RGBA or grayscale color. Defaults to [0.0, 0.0, 0.0, 1.0].

    Returns:
        Tensor: Uniform image.
    """
    # Check input validity
    check_arg_choice(mode, ['color', 'gray'], arg_name='mode')

    # Convert parameters to tensors
    # Resize RGBA color to match the number of channels in output
    num_channels = 1 if mode == 'gray' else 4 if use_alpha else 3
    rgba = resize_color(th.atleast_1d(to_tensor(rgba)).clamp(0.0, 1.0), num_channels)

    # Construct uniform color output
    img_out = rgba.view(-1, 1, 1).expand(num_imgs, num_channels, res_h, res_w).contiguous()

    return img_out


@input_check(2, channel_specs='.g')
def warp(img_in: th.Tensor, intensity_mask: th.Tensor, intensity: FloatValue = 1.0) -> th.Tensor:
    """Atomic node: Warp

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        intensity_mask (tensor): Intensity mask for computing displacement (G only).
        intensity (float, optional): Intensity mask multiplier. Defaults to 1.0.

    Returns:
        Tensor: Warped image.
    """
    # Convert parameters to tensors
    intensity = to_tensor(intensity / 256)

    # Compute displacement vector field
    num_row, num_col = img_in.shape[2], img_in.shape[3]
    vec_shift = th.cat((intensity_mask - th.roll(intensity_mask, 1, 3),
                        intensity_mask - th.roll(intensity_mask, 1, 2)), 1)
    vec_shift = vec_shift.movedim(1, 3) * (intensity * to_tensor([num_col, num_row]))

    # Perform sampling to obtain the warped image
    sample_grid = get_pos(num_row, num_col) + vec_shift
    img_out = grid_sample(img_in, sample_grid, sbs_format=True)

    return img_out


# ---------------------------------------- #
#          Pass-through functions          #
# ---------------------------------------- #


@input_check(1)
def passthrough(img_in: th.Tensor) -> th.Tensor:
    """Helper node: Dot (pass-through)

    Args:
        img_in (tensor): Input image (G or RGB(A)).

    Returns:
        Tensor: The same image.
    """
    return img_in


def passthrough_template(num_outputs: int = 1) -> NodeFunction:
    """A pass-through function template that generates pass-through functions of various numbers
    of output slots. Mainly used for dummy (ablated) nodes.
    """
    # Define the pass-through function to instantiate
    @input_check_all_positional()
    def func(*img_list: Optional[th.Tensor]) -> Union[th.Tensor, Tuple[th.Tensor, ...]]:

        # The input image list must not be empty
        if not img_list or all([img is None for img in img_list]):
            raise ValueError('No input image is provided')

        # Fetch the first non-empty input as output
        img_out = next(img for img in img_list if img is not None)
        output_imgs = [img_out]

        # Fill other output slots with empty images
        if num_outputs > 1:
            img_empty = th.zeros_like(img_out)
            output_imgs.extend([img_empty for _ in range(num_outputs - 1)])

        return img_out if num_outputs <= 1 else tuple(output_imgs)

    return func


# -------------------------------------- #
#          Non-atomic functions          #
# -------------------------------------- #

@input_check(1)
def linear_to_srgb(img_in: th.Tensor) -> th.Tensor:
    """Non-atomic node: Linear RGB to sRGB (Color and Grayscale)

    Args:
        img_in (tensor): Input image (G or RGB(A)).

    Returns:
        Tensor: Gamma corrected image.
    """
    # Adjust gamma
    in_mid = [0.425] if img_in.shape[1] == 1 else [0.425, 0.425, 0.425, 0.5]
    img_out = levels(img_in, in_mid=in_mid)

    return img_out


@input_check(1)
def srgb_to_linear(img_in: th.Tensor) -> th.Tensor:
    """Non-atomic node: sRGB to Linear RGB (Color and Grayscale)

    Args:
        img_in (tensor): Input image (G or RGB(A)).

    Returns:
        tensor: Gamma corrected image.
    """
    # Adjust gamma
    in_mid = [0.575] if img_in.shape[1] == 1 else [0.575, 0.575, 0.575, 0.5]
    img_out = levels(img_in, in_mid=in_mid)

    return img_out


@input_check(1, channel_specs='c')
def curvature(normal: th.Tensor, normal_format: str = 'dx',
              emboss_intensity: FloatValue = 1.0) -> th.Tensor:
    """Non-atomic node: Curvature

    Args:
        normal (tensor): Input normal image (RGB(A) only).
        normal_format (str, optional): Normal format ('dx' or 'gl'). Defaults to 'dx'.
        emboss_intensity (float, optional): Normalized intensity multiplier. Defaults to 1.0.

    Returns:
        tensor: Curvature image.
    """
    # Check input validity
    check_arg_choice(normal_format, ['dx', 'gl'], arg_name='normal_format')

    # Compute normal gradient
    normal_shift_x = normal[:,:1].roll(-1, 3)
    normal_shift_y = normal[:,1:2].roll(-1, 2)

    # Compute curvature contribution from X and Y using emboss filters
    gray = th.full_like(normal_shift_x, 0.5)
    pixel_size = 2048 / normal_shift_x.shape[2] * 0.1
    angle = 0.25 if normal_format == 'dx' else 0.75
    emboss_x = emboss(gray, normal_shift_x, emboss_intensity * pixel_size)
    emboss_y = emboss(gray, normal_shift_y, emboss_intensity * pixel_size, light_angle=angle)

    # Obtain the curvature image
    img_out = blend(emboss_x, emboss_y, blending_mode='add_sub', opacity=0.5)

    return img_out


@input_check(1)
def invert(img_in: th.Tensor, invert_switch: bool = True) -> th.Tensor:
    """Non-atomic node: Invert (Color and Grayscale)

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        invert_switch (bool, optional): Invert switch. Defaults to True.

    Returns:
        Tensor: Inverted image.
    """
    # No inversion
    if not invert_switch:
        img_out = img_in

    # Invert grayscale
    elif img_in.shape[1] == 1:
        img_out = th.clamp(1.0 - img_in, 0.0, 1.0)

    # Invert color (ignore the alpha channel)
    else:
        img_out = img_in.clone()
        img_out[:,:3] = th.clamp(1.0 - img_in[:,:3], 0.0, 1.0)

    return img_out


@input_check(1, channel_specs='g')
def histogram_scan(img_in: th.Tensor, invert_position: bool = False, position: FloatValue = 0.0,
                   contrast: FloatValue = 0.0) -> th.Tensor:
    """Non-atomic node: Histogram Scan

    Args:
        img_in (tensor): Input image (G only).
        invert_position (bool, optional): Invert position. Defaults to False.
        position (float, optional): Used to shift the middle point. Defaults to 0.0.
        contrast (float, optional): Used to adjust the contrast of the input. Defaults to 0.0.

    Returns:
        Tensor: Histogram scan image.
    """
    # Convert parameters to tensors
    position, contrast = to_tensor(position), to_tensor(contrast)
    position = position if invert_position else 1.0 - position

    # Compute histogram scan range
    start_low = (position.clamp_min(0.5) - 0.5) * 2.0
    end_low = (position * 2.0).clamp_max(1.0)
    weight_low = (contrast * 0.5).clamp(0.0, 1.0)
    in_low = th.lerp(start_low, end_low, weight_low)
    in_high = th.lerp(end_low, start_low, weight_low)

    # Perform histogram adjustment
    img_out = levels(img_in, in_low=in_low, in_high=in_high)

    return img_out


@input_check(1, channel_specs='g')
def histogram_range(img_in: th.Tensor, ranges: FloatValue = 0.5,
                    position: FloatValue = 0.5) -> th.Tensor:
    """Non-atomic node: Histogram Range

    Args:
        img_in (tensor): Input image (G only).
        ranges (float, optional): How much to reduce the range down from. This is similar to moving
            both Levels min and max sliders inwards. Defaults to 0.5.
        position (float, optional): Offset for the range reduction, setting a different midpoint
            for the range reduction. Defaults to 0.5.

    Returns:
        Tensor: Histogram range image.
    """
    # Convert parameters to tensors
    ranges, position = to_tensor(ranges), to_tensor(position)

    # Compute histogram mapping range
    out_low  = 1.0 - th.min(ranges * 0.5 + (1.0 - position), (1.0 - position) * 2.0)
    out_high = th.min(ranges * 0.5 + position, position * 2.0)

    # Perform histogram adjustment
    img_out = levels(img_in, out_low=out_low, out_high=out_high)

    return img_out


@input_check(1, channel_specs='g')
def histogram_select(img_in: th.Tensor, position: FloatValue = 0.5, ranges: FloatValue = 0.25,
                     contrast: FloatValue = 0.0) -> th.Tensor:
    """Non-atomic node: Histogram Select

    Args:
        img_in (tensor): Input image (G only).
        position (float, optional): Sets the middle position where the range selection happens.
            Defaults to 0.5.
        ranges (float, optional): Sets width of the selection range. Defaults to 0.25.
        contrast (float, optional): Adjusts the contrast/falloff of the result. Defaults to 0.0.

    Returns:
        Tensor: Histogram select image.
    """
    # Convert parameters to tensors
    position, contrast = to_tensor(position), to_tensor(contrast)
    ranges, ranges_const = to_tensor_and_const(ranges)

    # Output full-white image when ranages is zero
    if ranges_const == 0.0:
        img_out = th.ones_like(img_in)

    # Perform histogram adjustment
    else:
        img = (1.0 - th.abs(img_in - position) / ranges).clamp(0.0, 1.0)
        img_out = levels(img, in_low = contrast * 0.5, in_high = 1.0 - contrast * 0.5)

    return img_out


@input_check(1, channel_specs='g')
def edge_detect(img_in: th.Tensor, invert_flag: bool = False, edge_width: FloatValue = 2.0,
                edge_roundness: FloatValue = 4.0, tolerance: FloatValue = 0.0) -> th.Tensor:
    """Non-atomic node: Edge Detect

    Args:
        img_in (tensor): Input image (G only).
        invert_flag (bool, optional): Invert the result. Defaults to False.
        edge_width (float, optional): Normalized width of the detected areas around the edges.
            Defaults to 2.0.
        edge_roundness (float, optional): Normalized rounds, blurs and smooths together the
            generated mask. Defaults to 4.0.
        tolerance (float, optional): Tolerance threshold factor for where edges should appear.
            Defaults to 0.0.

    Returns:
        Tensor: Detected edge image.
    """
    # Convert parameters to tensors
    edge_width, edge_roundness = to_tensor(edge_width), to_tensor(edge_roundness)
    tolerance = to_tensor(tolerance)

    # Edge detect through symmetric difference
    img_scale = 256.0 / min(img_in.shape[2], img_in.shape[3])
    in_blur = blur(img_in, img_scale)
    blend_sub_1 = blend(in_blur, img_in, blending_mode='subtract')
    blend_sub_2 = blend(img_in, in_blur, blending_mode='subtract')
    img_out = blend(blend_sub_1, blend_sub_2, blending_mode='add')

    # Adjust edge tolerance
    img_out = levels(img_out, in_high=0.05)
    img_out = levels(img_out, in_low = 0.002, in_high = (tolerance + 0.2) * 0.01)

    # Apply edge width
    dist, dist_const = to_tensor_and_const((edge_width - 1.0).clamp_min(0.0))
    img_out = distance(img_out, img_out, combine=False, dist=dist) if dist_const else img_out

    # Apply edge roundness
    dist, dist_const = to_tensor_and_const(th.ceil(edge_roundness).clamp_min(0.0))
    img_out = distance(img_out, img_out, combine=False, dist=dist) if dist_const else img_out
    img_out = 1.0 - img_out
    img_out = distance(img_out, img_out, combine=False, dist=dist) if dist_const else img_out

    # Optional invert the final output
    img_out = 1.0 - img_out if invert_flag else img_out

    return img_out


@input_check(1)
def safe_transform(img_in: th.Tensor, tile: int = 1, tile_safe_rot: bool = True,
                   symmetry: str = 'none', tiling: int = 3, mipmap_mode: str = 'auto',
                   mipmap_level: int = 0, offset_mode: str = 'manual',
                   offset: FloatVector = [0.0, 0.0], angle: FloatValue = 0.0) -> th.Tensor:
    """Non-atomic node: Safe Transform (Color and Grayscale)

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        tile (int, optional): Scales the input down by tiling it. Defaults to 1.
        tile_safe_rot (bool, optional): Determines the behaviors of the rotation, whether it
            should snap to safe values that don't blur any pixels. Defaults to True.
        symmetry (str, optional): 'X' | 'Y' | 'X+Y' | 'none', performs symmetric transformation
            on the input. Defaults to 'none'.
        tiling (int, optional): see 'tiling' in Transformation 2D. Defaults to 3.
        mipmap_mode (str, optional): 'auto' or 'manual'. Defaults to 'auto'.
        mipmap_level (int, optional): Mipmap level. Defaults to 0.
        offset_mode (str, optional): Offset mode, 'manual' or 'random'. Defaults to 'manual'.
        offset (list, optional): Translates input by offset. Defaults to [0.0, 0.0].
        angle (float, optional): Rotates input along angle (in turns). Defaults to 0.0.

    Returns:
        Tensor: Safe transformed image.
    """
    # Check input validity
    check_arg_choice(symmetry, ['none', 'X', 'Y', 'X+Y'], arg_name='symmetry')
    check_arg_choice(offset_mode, ['manual', 'random'], arg_name='offset_mode')

    # Symmetry transform
    if symmetry == 'none':
        img_out = img_in
    else:
        flip_dims = {'X': [2], 'Y': [3], 'X+Y': [2, 3]}[symmetry]
        img_out = th.flip(img_in, dims=flip_dims)

    # Prepare transform parameters: rotation, scaling, and offset
    angle = to_tensor(angle)
    offset_tile = (tile + 1) % 2 * 0.5

    ## Consider tiling safety by truncating to 45*k degrees rotation
    if tile_safe_rot:
        angle, angle_const = to_tensor_and_const(th.floor(angle * 8.0) / 8.0)
        angle_res = abs(angle_const) % 0.25 * (np.pi * 2.0)
        tile = tile * (math.cos(angle_res) + math.sin(angle_res))

    ## Translation offset
    num_row, num_col = img_in.shape[2], img_in.shape[3]
    img_scale = to_tensor([num_col, num_row])
    offset = to_tensor(offset) if offset_mode == 'manual' else th.rand(2)
    offset = th.floor(offset * img_scale) / img_scale + offset_tile

    # Affine transformation
    angle = angle * (math.pi * 2.0)
    cos_angle, sin_angle = th.cos(angle), th.sin(angle)
    rotation_matrix = th.stack((cos_angle, -sin_angle, sin_angle, cos_angle))
    img_out = transform_2d(
        img_out, tiling=tiling, mipmap_mode=mipmap_mode, mipmap_level=mipmap_level,
        matrix22=rotation_matrix*tile, offset=offset)

    return img_out


@input_check(1)
def blur_hq(img_in: th.Tensor, high_quality: bool = False,
            intensity: FloatValue = 10.0) -> th.Tensor:
    """Non-atomic node: Blur HQ (Color and Grayscale)

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        high_quality (bool, optional): Increases internal sampling amount for even higher quality,
            at reduced computation speed. Defaults to False.
        intensity (tensor, optional): Normalized strength (radius) of the blur. The higher this
            value, the further the blur will reach. Defaults to 10.0.

    Returns:
        Tensor: High quality blurred image.
    """
    # Convert parameters to tensors
    intensity = to_tensor(intensity) * 0.66

    # Basic quality blur - 4 directions
    blur_1 = img_in
    for angle in (0.0, 0.125, 0.25, 0.875):
        blur_1 = d_blur(blur_1, intensity=intensity, angle=angle)
    img_out = blur_1

    # High quality blur - 8 directions
    if high_quality:
        blur_2 = img_in
        for angle in (0.0625, 0.4375, 0.1875, 0.3125):
            blur_2 = d_blur(blur_2, intensity=intensity, angle=angle)
        img_out = blend(img_out, blur_2, opacity=0.5)

    return img_out


@input_check(2, channel_specs='.g')
def non_uniform_blur(img_in: th.Tensor, img_mask: th.Tensor, samples: int = 4, blades: int = 5,
                     intensity: FloatValue = 10.0, anisotropy: FloatValue = 0.0,
                     asymmetry: FloatValue = 0.0, angle: FloatValue = 0.0) -> th.Tensor:
    """Non-atomic node: Non-uniform Blur (Color and Grayscale)

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        img_mask (tensor): Blur map (G only).
        samples (int, optional): Amount of samples, determines quality. Multiplied by amount of
            Blades. Defaults to 4.
        blades (int, optional): Amount of sampling sectors, determines quality. Multiplied by
            amount of Samples. Defaults to 5.
        intensity (float, optional): Intensity of blur. Defaults to 10.0.
        anisotropy (float, optional): Optionally adds directionality to the blur effect.
            Driven by the Angle parameter. Defaults to 0.0.
        asymmetry (float, optional): Optionally adds a bias to the sampling. Driven by the Angle
            parameter. Defaults to 0.0.
        angle (float, optional): Angle to set directionality and sampling bias. Defaults to 0.0.

    Returns:
        Tensor: Non-uniform blurred image.
    """
    # Check input validity
    check_arg_choice(samples, range(1, 17), arg_name='samples')
    check_arg_choice(blades, range(1, 10), arg_name='blades')

    # Convert parameters to tensors
    intensity, intensity_const = to_tensor_and_const(intensity)
    anisotropy, asymmetry = to_tensor(anisotropy), to_tensor(asymmetry)
    angle = to_tensor(angle * (math.pi * -2.0))

    # Precompute regular sampling grid
    num_row, num_col = img_in.shape[2], img_in.shape[3]
    sample_grid = get_pos(num_row, num_col)
    scales = to_tensor([num_col / (num_col + 2), num_row / (num_row + 2)])

    # Inline and pre-compute for ellipse
    cos_angle, sin_angle = th.cos(angle), th.sin(angle)
    vec_1, vec_2 = th.stack((sin_angle, cos_angle)), th.stack((cos_angle, -sin_angle))
    ellipse_factor_inv = 1.0 - anisotropy
    center_x = th.clamp_min(asymmetry * 0.5, 0.0)

    # Elliptic sampling function
    def ellipse(samples: int, sample_number: int, radius: th.Tensor,
                inner_rotation: float) -> th.Tensor:

        angle_1 = (sample_number / samples + inner_rotation) * (np.pi * 2.0)
        cos_1, sin_1 = math.cos(angle_1), math.sin(angle_1)
        factor_1 = ellipse_factor_inv * sin_1
        factor_2 = center_x * (abs(cos_1) - cos_1) + cos_1

        return radius * (vec_1 * factor_1 + vec_2 * factor_2)

    # Compute progressive warping results based on 'samples'
    def non_uniform_blur_sample(img_in: th.Tensor, img_mask: th.Tensor, intensity: th.Tensor,
                                inner_rotation: float) -> th.Tensor:

        # Pre-pad the input image
        img_pad = pad2d(img_in, 1)
        img_out = img_in

        # Progressive warping towards multiple blades
        for i in range(1, blades + 1):

            # Inline d-warp and blend
            e_vec = ellipse(blades, i, intensity, inner_rotation)
            sample_grid_ = (sample_grid + img_mask.movedim(1, 3) * (e_vec / 256)) % 1 * 2 - 1
            sample_grid_ = sample_grid_ * scales
            img_warp = grid_sample_impl(img_pad, sample_grid_, align_corners=False)
            img_out = th.lerp(img_out, img_warp, 1.0 / (i + 1))

        return img_out

    # Compute progressive blurring based on 'samples' and 'intensity'
    samples_level = min(samples, int(math.ceil(intensity_const * math.pi)))
    img_out = non_uniform_blur_sample(img_in, img_mask, intensity, 1 / samples)

    for i in range(1, samples_level):
        intensity_scale = math.exp(-i * math.sqrt(math.log(1e3) / math.e) / samples_level) ** 2
        blur_intensity = intensity * intensity_scale
        inner_rotation = 1 / (samples * (i + 1))
        img_out = non_uniform_blur_sample(img_out, img_mask, blur_intensity, inner_rotation)

    return img_out


@input_check(1, channel_specs='g')
def bevel(img_in: th.Tensor, non_uniform_blur_flag: bool = True, use_alpha: bool = False,
          dist: FloatValue = 0.5, smoothing: FloatValue = 0.0,
          normal_intensity: FloatValue = 10.0) -> Tuple[th.Tensor, th.Tensor]:
    """Non-atomic node: Bevel

    Args:
        img_in (tensor): Input image (G only).
        non_uniform_blur_flag (bool, optional): Whether smoothing should be done non-uniformly.
            Defaults to True.
        use_alpha (bool, optional): Enable the alpha channel in output. Defaults to False.
        dist (float, optional): How far the bevel effect should reach. Defaults to 0.5.
        smoothing (float, optional): How much additional smoothing (blurring) to perform after
            the bevel. Defaults to 0.0.
        normal_intensity (float, optional): Normalized intensity of the generated normal map.
            Defaults to 10.0.

    Returns:
        Tensor: Bevel image (G).
        Tensor: Normal map (RGB(A)).
    """
    # Convert parameters to tensors
    dist, dist_const = to_tensor_and_const(dist)
    smoothing, smoothing_const = to_tensor_and_const(smoothing)
    normal_intensity = to_tensor(normal_intensity)

    # Compute beveled height map
    if dist_const > 0:
        height = distance(img_in, combine=True, dist=dist * 128)
    elif dist_const < 0:
        height = 1.0 - distance(1.0 - img_in, combine=True, dist=-dist * 128)
    else:
        height = img_in

    # Height map smoothing after beveling
    if smoothing_const > 0:
        if non_uniform_blur_flag:
            img_blur = blur(height, intensity=0.5)
            img_blur = levels(img_blur, in_high=0.0214)
            height = non_uniform_blur(height, img_blur, samples=6, blades=5, intensity=smoothing)
        else:
            height = blur_hq(height, intensity=smoothing)

    # Compute normal map from height map
    normal_1 = normal(height.rot90(2, (2, 3)), use_alpha=use_alpha, intensity=normal_intensity)
    normal_1 = normal_1.rot90(2, (2, 3))
    normal_1[:, :2] = 1.0 - normal_1.narrow(1, 0, 2)

    normal_2 = normal(height, use_alpha=use_alpha, intensity=normal_intensity)
    normal_out = blend(normal_1, normal_2, opacity=0.5)

    return height, normal_out


@input_check(2, channel_specs='.g')
def slope_blur(img_in: th.Tensor, img_mask: th.Tensor, samples: int = 8, mode: str = 'blur',
               intensity: FloatValue = 10.0) -> th.Tensor:
    """Non-atomic node: Slope Blur (Color and Grayscale)

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        img_mask (tensor): Mask image (G only).
        samples (int, optional): Amount of samples, affects the quality at the expense of speed.
            Defaults to 1.
        mode (str, optional): Blending mode for consequent blur passes. "Blur" behaves more like
            a standard Anisotropic Blur, while Min will "eat away" existing areas and Max will
            "smear out" white areas. Defaults to 'blur'.
        intensity (tensor, optional): Normalized blur amount or strength. Defaults to 10.0.

    Returns:
        Tensor: Slope blurred image.
    """
    # Check input validity
    check_arg_choice(samples, range(1, 33), arg_name='samples')
    check_arg_choice(mode, ['blur', 'min', 'max'], arg_name='mode')

    # Convert parameters to tensors
    intensity, intensity_const = to_tensor_and_const(intensity)

    # Special case - no action needed
    if intensity_const == 0:
        return img_in

    # Compute displacement vector field and the sampling grid for warping
    num_row, num_col = img_in.shape[2], img_in.shape[3]
    scales = to_tensor([num_col / 256, num_row / 256])
    vec_shift = th.cat((img_mask - th.roll(img_mask, 1, 3),
                        img_mask - th.roll(img_mask, 1, 2)), dim=1)
    vec_shift = vec_shift.movedim(1, 3) * (scales * (intensity / samples))

    sample_grid = (get_pos(num_row, num_col) + vec_shift) % 1 * 2 - 1
    sample_grid = sample_grid * to_tensor([num_col / (num_col + 2), num_row / (num_row + 2)])

    # Apply slope blur effect via progressive warping and blending
    blending_mode = 'switch' if mode == 'blur' else mode
    img_warp = grid_sample_impl(pad2d(img_in, 1), sample_grid, align_corners=False)
    img_out = img_warp
    for i in range(2, samples + 1):
        img_warp = grid_sample_impl(pad2d(img_warp, 1), sample_grid, align_corners=False)
        img_out = blend(img_warp, img_out, blending_mode=blending_mode, opacity=1 / i)

    return img_out


@input_check(2, channel_specs='.g')
def mosaic(img_in: th.Tensor, img_mask: th.Tensor, samples: int = 1,
           intensity: FloatValue = 0.5) -> th.Tensor:
    """Non-atomic node: Mosaic (Color and Grayscale)

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        img_mask (tensor): Mask image (G only).
        samples (int, optional): Determines multi-sample quality. Defaults to 1.
        intensity (float, optional): Strength of the effect. Defaults to 0.5.

    Returns:
        Tensor: Mosaic image.
    """
    # Check input validity
    check_arg_choice(samples, range(1, 17), arg_name='samples')

    # Convert parameters to tensors
    intensity = to_tensor(intensity)
    if intensity == 0:
        return img_in

    # Compute displacement vector field and the sampling grid for warping
    num_row, num_col = img_in.shape[2], img_in.shape[3]
    scales = to_tensor([num_col / 256, num_row / 256])
    vec_shift = th.cat((img_mask - th.roll(img_mask, 1, 3),
                        img_mask - th.roll(img_mask, 1, 2)), 1)
    vec_shift = vec_shift.movedim(1, 3) * (scales * (intensity / samples))

    sample_grid = (get_pos(num_row, num_col) + vec_shift) % 1 * 2 - 1
    sample_grid = sample_grid * to_tensor([num_col / (num_col + 2), num_row / (num_row + 2)])

    # Apply mosaic effect via progressive warping
    img_out = img_in
    for _ in range(samples):
        img_out = grid_sample_impl(pad2d(img_out, 1), sample_grid, align_corners=False)

    return img_out


@input_check(1, channel_specs='g')
def auto_levels(img_in: th.Tensor) -> th.Tensor:
    """Non-atomic node: Auto Levels

    Args:
        img_in (tensor): Input image (G only).

    Returns:
        Tensor: Auto leveled image.
    """
    # Get input pixel value range
    max_val, min_val = th.max(img_in), th.min(img_in)
    delta, delta_const = to_tensor_and_const(max_val - min_val)

    # When input is a uniform image, and the pixel value is smaller (or greater) than 0.5,
    # output a white (or black) image
    if delta_const == 0:
        img_out = th.ones_like(img_in) if to_const(max_val) < 0.5 else th.zeros_like(img_in)
    else:
        img_out = (img_in - min_val) / (delta + 1e-15)

    return img_out


@input_check(1, channel_specs='g')
def ambient_occlusion(img_in: th.Tensor, spreading: FloatValue = 0.15,
                      equalizer: FloatVector = [0.0, 0.0, 0.0],
                      levels_param: FloatVector = [0.0, 0.5, 1.0]) -> th.Tensor:
    """Non-atomic node: Ambient Occlusion (deprecated)

    Args:
        img_in (tensor): Input image (G only).
        spreading (float, optional): Area of the ambient occlusion effect. Defaults to 0.15.
        equalizer (list, optional): Frequency equalizer. Defaults to [0.0, 0.0, 0.0].
        levels_param (list, optional): Controls final levels mapping. Defaults to [0.0, 0.5, 1.0].

    Returns:
        Tensor: ambient occlusion image
    """
    # Process parameters
    spreading, equalizer = to_tensor(spreading), to_tensor(equalizer)
    levels_param = to_tensor(levels_param)

    # Calculate an initial ambient occlusion map
    img_blur = blur_hq(1.0 - img_in, intensity = spreading * 128.0)
    img_ao = levels((img_blur + img_in).clamp_max(1.0), in_low=0.5)
    img_normal_z = normal(img_in, intensity=16.0)[:, 2:3]
    img_ao = img_ao * (img_ao + (1.0 - img_normal_z)).clamp_max(1.0)

    # Frequency split
    img_ao_blur = blur_hq(manual_resize(img_ao, -1), intensity=2.2)
    img_ao_blur_2 = blur_hq(manual_resize(img_ao_blur, -1), intensity=3.3)
    img_blend = blend(manual_resize(1.0 - img_ao_blur, 1), img_ao,
                      blending_mode='add_sub', opacity=0.5)
    img_blend_1 = blend(manual_resize(1.0 - img_ao_blur_2, 1), img_ao_blur,
                        blending_mode='add_sub', opacity=0.5)

    # Frequency equalization and composition
    img_ao_blur_2 = levels(img_ao_blur_2, in_mid = (equalizer[0] + 1) * 0.5)
    img_blend_1 = blend(img_blend_1, manual_resize(img_ao_blur_2, 1), blending_mode = 'add_sub',
                        opacity = equalizer[1] + 0.5)
    img_blend = blend(img_blend, manual_resize(img_blend_1, 1), blending_mode = 'add_sub',
                      opacity = equalizer[2] + 0.5)

    # Final gamma correction
    img_out = levels(img_blend, in_low=levels_param[0], in_mid=levels_param[1],
                     in_high=levels_param[2])

    return img_out


@input_check(1, channel_specs='g')
def hbao(img_in: th.Tensor, quality: int = 4, depth: FloatValue = 0.1, world_units: bool = False,
         surface_size: FloatValue = 300.0, height_cm: FloatValue = 30.0,
         radius: FloatValue = 1.0) -> th.Tensor:
    """Non-atomic node: Ambient Occlusion (HBAO)

    Args:
        img_in (tensor): Input image (G only).
        quality (int, optional): Amount of samples used for calculation. Defaults to 4.
        depth (float, optional): Height depth. Defaults to 0.1.
        radius (float, optional): The spread of the AO. Defaults to 1.0.

    Raises:
        NotImplementedError: Input image batch size is greater than 1.

    Returns:
        Tensor: ambient occlusion image.
    """
    # Check input validity
    check_arg_choice(quality, [4, 8, 16], arg_name='quality')
    if img_in.shape[0] > 1:
        raise NotImplementedError('Batched HBAO operation is currently not supported')

    # Convert parameters to tensors
    num_row, num_col = img_in.shape[2], img_in.shape[3]
    pixel_size = 1.0 / max(num_row, num_col)
    depth = to_tensor(height_cm / surface_size if world_units else depth) * min(num_row, num_col)
    radius = to_tensor(radius)

    # Create mipmap stack
    in_low, in_high = levels(img_in, out_high=0.5), levels(img_in, out_low=0.5)
    mipmaps_level = 11
    mipmaps = create_mipmaps(in_high, mipmaps_level, keep_size=True)

    # Precompute weights
    min_size_log2 = int(math.log2(min(num_row, num_col)))
    weights = radius * to_tensor([2 ** (min_size_log2 - i - 1) for i in range(mipmaps_level)]) - 1
    weights = weights.clamp(0.0, 1.0)

    # HBAO cone sampling
    ## Initialize the initial sampling grid and angle vectors in all sampling cones
    sample_grid_init = get_pos(num_row, num_col)
    angle_vecs = [(math.cos(i * math.pi * 2.0 / quality),
                   math.sin(i * math.pi * 2.0 / quality)) for i in range(quality)]
    angle_vecs = to_tensor(angle_vecs)

    ## Initialize cone sampling result
    img_sample = th.zeros_like(img_in)

    ## Run parallel cone sampling on each mipmap level
    for mm_idx, img_mm in enumerate(mipmaps):

        # Sample all cones in parallel
        mm_scale = (1 << mm_idx + 1)
        sample_grid = sample_grid_init + mm_scale * pixel_size * angle_vecs.view(quality, 1, 1, 2)
        img_mm_gs = grid_sample(img_mm.expand(quality, -1, -1, -1), sample_grid, sbs_format=True)

        # Add contribution from the current mipmap level to cone sampling result
        img_diff = (img_mm_gs - in_low - 0.5) / mm_scale
        img_max = th.max(img_max, img_diff) if mm_idx else img_diff
        img_sample = th.lerp(img_sample, img_max, weights[mm_idx])

    # Aggregate result from cones into the final image
    img_sample = img_sample * (depth * 2.0)
    img_sample = img_sample / th.sqrt(img_sample ** 2 + 1.0)
    img_out = th.mean(img_sample, 0, keepdim=True).view_as(img_in)

    # Final output
    img_out = th.clamp(1.0 - img_out, 0.0, 1.0)

    return img_out


@input_check(1)
def highpass(img_in: th.Tensor, radius: FloatValue = 6.0) -> th.Tensor:
    """Non-atomic node: Highpass (Color or Grayscale)

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        radius (float, optional): A small radius removes small differences, a bigger radius removes
            large areas. Defaults to 6.0.

    Returns:
        Tensor: Highpass filtered image.
    """
    # Highpass filter
    img_out = blur(img_in, radius)
    if img_out.shape[1] >= 4:
        rgb, a = img_out.split((3, 1), dim=1)
        img_out = th.cat((1 - rgb, a), dim=1)
    else:
        img_out = 1 - img_out

    img_out = blend(img_out, img_in, blending_mode='add_sub', opacity=0.5)

    return img_out


@input_check(1, channel_specs='c')
def normal_normalize(normal: th.Tensor) -> th.Tensor:
    """Non-atomic function: Normal Normalize

    Args:
        normal (tensor): Normal image (RGB(A) only).

    Returns:
        tensor: Normal normalized image.
    """
    # Split the alpha channel from input
    use_alpha = normal.shape[1] == 4
    normal_rgb, normal_alpha = normal.split(3, dim=1) if use_alpha else (normal, None)

    # Normalize normal map
    normal_rgb = normal_rgb * 2.0 - 1.0
    normal_rgb = normal_rgb / th.norm(normal_rgb, dim=1, keepdim=True) * 0.5 + 0.5

    # Append the original alpha channel
    normal = th.cat((normal_rgb, normal_alpha), dim=1) if use_alpha else normal_rgb

    return normal


@input_check(1, channel_specs='c')
def channel_mixer(img_in: th.Tensor, monochrome: bool = False,
                  red: FloatVector = [100.0, 0.0, 0.0, 0.0],
                  green: FloatVector = [0.0, 100.0, 0.0, 0.0],
                  blue: FloatVector = [0.0, 0.0, 100.0, 0.0]) -> th.Tensor:
    """Non-atomic node: Channel Mixer

    Args:
        img_in (tensor): Input image (RGB(A) only).
        monochrome (bool, optional): Output monochrome image. Defaults to False.
        red (list, optional): Mixing weights for output red channel.
            Defaults to [100.0, 0.0, 0.0, 0.0].
        green (list, optional): Mixing weights for output green channel.
            Defaults to [0.0, 100.0, 0.0, 0.0].
        blue (list, optional): Mixing weights for output blue channel.
            Defaults to [0.0, 0.0, 100.0, 0.0].

    Returns:
        Tensor: Channel mixed image.
    """
    # Convert parameters to tensors
    red, green, blue = tuple(to_tensor(color) * 0.01 for color in (red, green, blue))

    # Split the alpha channel from input
    img_in, img_in_alpha = img_in.split(3, dim=1) if img_in.shape[1] == 4 else (img_in, None)

    # Mix channels using provided coefficients
    if monochrome:
        img_out = (img_in * red[:3].view(-1, 1, 1)).sum(1, keepdim=True) + red[3]
        img_out = img_out.expand(-1, 3, -1, -1)
    else:
        weights = th.stack((red, green, blue))
        img_out = th.matmul(weights[:,:3], img_in.movedim(1, 3).unsqueeze(4))
        img_out = img_out.squeeze(4).movedim(3, 1) + weights[:,3:].unsqueeze(2)

    # Append the original alpha channel
    img_out = img_out.clamp(0.0, 1.0)
    img_out = th.cat((img_out, img_in_alpha), dim=1) if img_in_alpha is not None else img_out

    return img_out


@input_check(2, channel_specs='cc')
def normal_combine(img_normal1: th.Tensor, img_normal2: th.Tensor,
                   mode: str = 'whiteout') -> th.tensor:
    """Non-atomic node: Normal Combine

    Args:
        normal_one (tensor): First normal image (RGB(A) only).
        normal_two (tensor): Second normal image (RGB(A) only).
        mode (str, optional): 'whiteout' | 'channel_mixer' | 'detail_oriented'.
            Defaults to 'whiteout'.

    Returns:
        Tensor: Normal combined image.
    """
    # Check input validity
    check_arg_choice(mode, ['whiteout', 'channel_mixer', 'detail_oriented'], arg_name='mode')

    # Split input normal maps into individual channels
    n1r, n1g, n1b = img_normal1[:,:3].split(1, 1)
    n2r, n2g, n2b = img_normal2[:,:3].split(1, 1)

    # White-out mode
    if mode == 'whiteout':

        # Add two sources together
        img_out_rgb = th.cat((img_normal1[:,:2] + img_normal2[:,:2] - 0.5,
                              img_normal1[:,2:3] * img_normal2[:,2:3]), dim=1)
        img_out = normal_normalize(img_out_rgb)

        # Attach an opaque alpha channel
        if img_normal1.shape[1] == 4:
            img_out = th.cat([img_out, th.ones_like(n1r)], dim=1)

    # Channel mixer mode
    elif mode == 'channel_mixer':

        # Positive components
        n2_pos = img_normal2.clone()
        n2_pos[:,:2] = img_normal2[:,:2].clamp_min(0.5) - 0.5
        if img_normal2.shape[1] == 4:
            n2_pos[:,3] = 1.0

        # Negative components
        n2_neg = img_normal2.clone()
        n2_neg[:,:2] = 0.5 - img_normal2[:,:2].clamp_max(0.5)
        n2_neg[:,2] = 1.0 - img_normal2[:,2]
        if img_normal2.shape[1] == 4:
            n2_neg[:,3] = 1.0

        # Blend normals by deducting negative components and including positive components
        img_out = blend(n2_neg, img_normal1, blending_mode='subtract')
        img_out = blend(n2_pos, img_out, blending_mode='add')
        img_out[:,2] = th.min(n2b, n1b)

    # Detail oriented mode
    else:

        # Implement pixel processor ggb_rgb_temp
        n1x = n1r * 2.0 - 1.0
        n1y = n1g * 2.0 - 1.0
        inv_n1z = 1.0 / (n1b + 1.0)
        n1_xy_invz = (-n1x * n1y) * inv_n1z
        n1_xx_invz = 1.0 - n1x ** 2 * inv_n1z
        n1_yy_invz = 1.0 - n1y ** 2 * inv_n1z

        n1b_mask = n1b < -0.9999
        neg_x, neg_y = tuple(th.zeros_like(img_normal1[:,:3]) for _ in range(2))
        neg_x[:,0,:,:] = -1.0
        neg_y[:,1,:,:] = -1.0

        n1x_out = th.cat([n1_xx_invz, n1_xy_invz, -n1x], dim=1)
        n1x_out = th.where(n1b_mask, neg_y, n1x_out)
        n1y_out = th.cat([n1_xy_invz, n1_yy_invz, -n1y], dim=1)
        n1y_out = th.where(n1b_mask, neg_x, n1y_out)

        n1x_out = n1x_out * (n2r * 2.0 - 1.0)
        n1y_out = n1y_out * (n2g * 2.0 - 1.0)
        n1z_out = (img_normal1[:,:3] * 2.0 - 1.0) * (n2b * 2.0 - 1.0)
        img_out = (n1x_out + n1y_out + n1z_out) * 0.5 + 0.5

        if img_normal1.shape[1] == 4:
            img_out = th.cat((img_out, th.ones_like(n1r)), dim=1)

    # Clamp final output
    img_out = th.clamp(img_out, 0.0, 1.0)

    return img_out


@input_check(1, channel_specs='g')
def height_to_normal_world_units(img_in: th.Tensor, normal_format: str = 'dx',
                                 sampling_mode: str = 'standard', use_alpha: bool = False,
                                 surface_size: FloatValue = 300.0,
                                 height_depth: FloatValue = 16.0) -> th.Tensor:
    """Non-atomic node: Height to Normal World Units

    Args:
        img_in (tensor): Input image (G only).
        normal_format (str, optional): 'gl' or 'dx'. Defaults to 'gl'.
        sampling_mode (str, optional): 'standard' or 'sobel', switches between two sampling modes
            determining accuracy. Defaults to 'standard'.
        use_alpha (bool, optional): Enable the alpha channel. Defaults to False.
        surface_size (float, optional): Normalized dimensions of the input height map.
            Defaults to 300.0.
        height_depth (float, optional): Normalized depth of height map details. Defaults to 16.0.

    Returns:
        Tensor: Normal image.
    """
    # Check input validity
    check_arg_choice(normal_format, ['dx', 'gl'], arg_name='normal_format')
    check_arg_choice(sampling_mode, ['standard', 'sobel'], arg_name='sampling_mode')

    # Convert parameters to tensors
    surface_size, surface_size_const = to_tensor_and_const(surface_size)
    height_depth = to_tensor(height_depth)

    aspect_ratio = height_depth / surface_size if surface_size_const > 0 else 0.0
    res_x, inv_res_x = img_in.shape[2], 1.0 / img_in.shape[2]
    res_y, inv_res_y = img_in.shape[3], 1.0 / img_in.shape[3]

    # Standard normal conversion
    if sampling_mode == 'standard':
        img_out = normal(img_in, normal_format = normal_format, use_alpha = use_alpha,
                         intensity = aspect_ratio * 256.0)

    # Sobel sampling
    else:

        # Convolution
        db_x = d_blur(img_in, intensity = inv_res_x * 256.0)
        db_y = d_blur(img_in, intensity = inv_res_y * 256.0, angle = 0.25)
        sample_x = db_y.roll(1, 3) - db_y.roll(-1, 3)
        sample_y = db_x.roll(-1, 2) - db_x.roll(1, 2)

        # Multiplier
        mult_x = aspect_ratio * (res_x * 0.5)
        mult_y = aspect_ratio * ((-1.0 if normal_format == 'dx' else 1.0) * res_y * 0.5)
        sample_x = sample_x * (mult_x * (min(res_x, res_y) / res_x))
        sample_y = sample_y * (mult_y * (min(res_x, res_y) / res_y))

        # Output
        scale = 0.5 * th.rsqrt(sample_x ** 2 + sample_y ** 2 + 1)
        img_out = th.cat([sample_x, sample_y, th.ones_like(img_in)], dim=1) * scale + 0.5
        img_out = th.clamp(img_out, 0.0, 1.0)

        # Add opaque alpha channel
        img_out = th.cat([img_out, th.ones_like(img_in)], dim=1) if use_alpha else img_out

    return img_out


@input_check(1, channel_specs='c')
def normal_to_height(img_in: th.Tensor, normal_format: str = 'dx',
                     relief_balance: FloatVector = [0.5, 0.5, 0.5],
                     opacity: FloatValue = 0.36) -> th.Tensor:
    """Non-atomic node: Normal to Height

    Args:
        img_in (tensor): Input image (RGB(A) only).
        normal_format (str, optional): 'gl' or 'dx'. Defaults to 'dx'.
        relief_balance (list, optional): Adjust the extent to which the different frequencies
            influence the final result.
        This is largely dependent on the input map and requires a fair bit of tweaking.
            Defaults to [0.5, 0.5, 0.5].
        opacity (float, optional): Global opacity of the effect. Defaults to 0.36.

    Raises:
        NotImplementedError: Input image size is smaller than 16.

    Returns:
        Tensor: Height image.
    """
    # Check input validity
    check_arg_choice(normal_format, ['dx', 'gl'], arg_name='normal_format')
    if int(math.log2(img_in.shape[2])) < 4:
        raise NotImplementedError('Image sizes smaller than 16 are not supported')

    # Convert parameters to tensors
    relief_balance = to_tensor(relief_balance) * opacity
    low_freq, mid_freq, high_freq = relief_balance.unbind()

    # Frequency transform for R and G channels
    img_freqs = frequency_transform(img_in[:,:2], normal_format=normal_format)
    img_out = None

    # Low frequencies (for 16x16 images only)
    for i in range(4):
        coeff = 0.0625 * 2 * (8 >> i) * 100
        blend_opacity = th.clamp(coeff * low_freq, 0.0, 1.0)
        img_out = img_freqs[i] if img_out is None else \
            blend(img_freqs[i], img_out, blending_mode='add_sub', opacity=blend_opacity)

    # Mid frequencies
    for i in range(min(2, len(img_freqs) - 4)):
        coeff = 0.0156 * 2 * (2 >> i) * 100
        blend_opacity = th.clamp(coeff * mid_freq, 0.0, 1.0)
        img_out = blend(img_freqs[i + 4], manual_resize(img_out, 1), blending_mode='add_sub',
                        opacity=blend_opacity)

    # High frequencies
    for i in range(min(6, len(img_freqs) - 6)):
        coeff = 0.0078 * 0.0625 * (32 >> i) * 100 if i < 5 else 0.0078 * 0.0612 * 100
        blend_opacity = th.clamp(coeff * high_freq, 0.0, 1.0)
        img_out = blend(img_freqs[i + 6], manual_resize(img_out, 1), blending_mode='add_sub',
                        opacity=blend_opacity)

    # Combine both channels
    img_out = blend(*img_out.split(1, dim=1), blending_mode='add_sub', opacity=0.5)

    return img_out


@input_check(1, channel_specs='c')
def curvature_smooth(img_in: th.Tensor, normal_format: str = 'dx') -> th.Tensor:
    """Non-atomic node: Curvature Smooth

    Args:
        img_in (tensor): Input normal image (RGB(A) only).
        normal_format (str, optional): 'dx' or 'gl'. Defaults to 'dx'.

    Raises:
        NotImplementedError: Input image size is smaller than 16.

    Returns:
        Tensor: Curvature smooth image.
    """
    # Check input validity
    check_arg_choice(normal_format, ['dx', 'gl'], arg_name='normal_format')
    if int(math.log2(img_in.shape[2])) < 4:
        raise NotImplementedError('Image sizes smaller than 16 are not supported')

    # Frequency transform for R and G channels
    img_freqs = frequency_transform(img_in[:,:2], normal_format)
    img_out = img_freqs[0]

    # Low frequencies (for 16x16 images only)
    for i in range(1, 4):
        img_out = blend(img_freqs[i], img_out, blending_mode='add_sub', opacity=0.25)

    # Mid and high frequencies
    for i in range(4, len(img_freqs)):
        img_out = blend(img_freqs[i], manual_resize(img_out, 1), blending_mode='add_sub',
                        opacity=1.0 / (i + 1))

    # Combine both channels
    img_out = blend(*img_out.split(1, dim=1), blending_mode='add_sub', opacity=0.5)

    return img_out


@input_check_all_positional(channel_specs='-')
def multi_switch(*img_list: th.Tensor, input_number: int = 2, input_selection: int = 1):
    """Non-atomic node: Multi Switch (Color and Grayscale)

    Args:
        img_list (list): A list of input images (G or RGB(A), must be identical across inputs).
        input_number (int, optional): Amount of inputs to expose. Defaults to 2.
        input_selection (int, optional): Which input to return as the result. Defaults to 1.

    Raises:
        ValueError: No input image is provided.

    Returns:
        Tensor: The selected input image.
    """
    # Check input validity
    if not img_list:
        raise ValueError('Input image list is empty')

    check_arg_choice(input_number, range(1, len(img_list) + 1), arg_name='input_number')
    check_arg_choice(input_selection, range(1, input_number + 1), arg_name='input_selection')

    # Select the output image
    img_out = img_list[input_selection - 1]

    return img_out


@input_check(1, channel_specs='c')
def rgba_split(rgba: th.Tensor) -> Tuple[th.Tensor, ...]:
    """Non-atomic node: RGBA Split

    Args:
        rgba (tensor): RGBA input image (RGB(A) only).

    Returns:
        Tuple of tensors: 4 single-channel images.
    """
    # Extract R, G, B, and A channels
    r, g, b = rgba[:,:3].split(1, dim=1)
    a = rgba[:,3:] if rgba.shape[1] == 4 else th.ones_like(r)

    return r, g, b, a


@input_check(4, channel_specs='gggg')
def rgba_merge(r: th.Tensor, g: th.Tensor, b: th.Tensor, a: Optional[th.Tensor] = None,
               use_alpha: bool = False) -> th.Tensor:
    """Non-atomic node: RGBA Merge

    Args:
        r (tensor): Red channel (G only).
        g (tensor): Green channel (G only).
        b (tensor): Blue channel (G only).
        a (tensor, optional): Alpha channel (G only). Defaults to None.
        use_alpha (bool, optional): Enable the alpha channel in output. Defaults to False.

    Returns:
        Tensor: RGBA image merged from the input 4 single-channel images.
    """
    # Collected all used channels
    active_channels = [r, g, b]
    if use_alpha:
        active_channels.append(a if a is not None else th.zeros_like(r))

    # Merge channels
    img_out = th.cat(active_channels, dim=1)

    return img_out


@input_check(3, channel_specs='.gg')
def pbr_converter(base_color: th.Tensor, roughness: th.Tensor, metallic: th.Tensor,
                  use_alpha: bool = False) -> Tuple[th.Tensor, ...]:
    """Non-atomic node: BaseColor / Metallic / Roughness converter

    Args:
        base_color (tensor): Base color map (G or RGB(A)).
        roughness (tensor): Roughness map (G only).
        metallic (tensor): Metallic map (G only).
        use_alpha (bool, optional): Enable the alpha channel in output. Defaults to False.

    Returns:
        Tuple of tensors: Diffuse, specular and glossiness maps.
    """
    # Initialize an opaque, black image
    black = th.zeros_like(base_color)
    if use_alpha and base_color.shape[1] == 4:
        black[:,3] = 1.0

    # Compute diffuse map
    invert_metallic = 1.0 - metallic
    invert_metallic_sRGB = 1.0 - linear_to_srgb(invert_metallic)
    diffuse = blend(black, base_color, invert_metallic_sRGB)

    # Compute specular map
    base_color_linear = srgb_to_linear(base_color)
    specular_blend = blend(black, base_color_linear, invert_metallic)
    specular_levels = th.clamp(invert_metallic * 0.04, 0.0, 1.0).expand(-1, 3, -1, -1)
    if use_alpha:
        specular_levels = th.cat((specular_levels, th.ones_like(invert_metallic)), dim=1)

    specular_blend_2 = blend(specular_levels, specular_blend)
    specular = linear_to_srgb(specular_blend_2)

    # Compute glossiness map
    glossiness = 1.0 - roughness

    return diffuse, specular, glossiness


@input_check(1, channel_specs='c')
def alpha_split(rgba: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    """Non-atomic node: Alpha Split

    Args:
        rgba (tensor): RGBA input image (RGB(A) only).

    Returns:
        Tuple of tensors: RGB and alpha images.
    """
    # Split the alpha channel from input
    rgb, a = rgba.split(3, dim=1) if rgba.shape[1] == 4 else (rgb, None)
    a = th.ones_like(rgba[:,:1]) if a is None else a

    # Append opaque alpha to RGB output
    rgb = th.cat((rgb, th.ones_like(a)), dim=1) if rgba.shape[1] == 4 else rgb

    return rgb, a


@input_check(2, channel_specs='cg')
def alpha_merge(rgb: th.Tensor, a: Optional[th.Tensor] = None) -> th.Tensor:
    """Non-atomic node: Alpha Merge

    Args:
        rgb (tensor): RGB input image (RGB(A) only).
        a (tensor): Alpha input image (G only).

    Returns:
        Tensor: RGBA input image.
    """
    # Merge the source alpha channel into RGB
    if rgb.shape[1] == 4:
        a = a if a is not None else th.zeros_like(rgb[:,:1])
        img_out = th.cat((rgb[:,:3], a), dim=1)
    else:
        img_out = rgb

    return img_out


@input_check(2, channel_specs='--', reduction='any', reduction_range=2)
def switch(img_1: Optional[th.Tensor] = None, img_2: Optional[th.Tensor] = None,
           flag: bool = True) -> th.Tensor:
    """Non-atomic node: Switch (Color and Grayscale)

    Args:
        img_1 (tensor, optional): First input image (G or RGB(A)).
        img_2 (tensor, optional): Second input image (G or RGB(A)).
        flag (bool, optional): Output the first image if True. Defaults to True.

    Returns:
        Tensor: Either input image.
    """
    # Select input image and deduce from empty connections
    img_out = img_1 if flag else img_2
    img_out = img_out if img_out is not None else th.zeros_like(img_2 if img_1 is None else img_1)

    return img_out


@input_check(3, channel_specs='ccg')
def normal_blend(normal_fg: th.Tensor, normal_bg: th.Tensor, mask: Optional[th.Tensor] = None,
                 use_mask: bool = True, opacity: FloatValue = 1.0) -> th.Tensor:
    """Non-atomic node: Normal Blend

    Args:
        normal_fg (tensor): Foreground normal (RGB(A) only).
        normal_bg (tensor): Background normal (RGB(A) only).
        mask (tensor, optional): Mask slot used for masking the node's effects (G only).
            Defaults to None.
        use_mask (bool, optional): Use mask if True. Defaults to True.
        opacity (float, optional): Blending opacity between foreground and background.
            Defaults to 1.0.

    Returns:
        Tensor: Blended normal image.
    """
    # Blend RGB channels
    mask = mask if use_mask else None
    img_out = blend(normal_fg[:,:3], normal_bg[:,:3], mask, opacity=opacity)

    # Blend alpha channels
    if normal_fg.shape[1] == 4:
        img_out_alpha = blend(normal_fg[:,3:], normal_bg[:,3:], mask, opacity=opacity)
        img_out = th.cat([img_out, img_out_alpha], dim=1)

    # Normalize the blended normal map
    img_out = normal_normalize(img_out)

    return img_out


@input_check(1)
def mirror(img_in: th.Tensor, mirror_axis: str = 'x', corner_type: str = 'tl',
           invert_x: bool = False, invert_y: bool = False, offset_x: FloatValue = 0.5,
           offset_y: FloatValue = 0.5) -> th.Tensor:
    """Non-atomic node: Mirror (Color and Grayscale)

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        mirror_axis (int, optional): 'x' | 'y' | 'corner'. Defaults to 'x'.
        corner_type (int, optional): 'tl' | 'tr' | 'bl' | 'br'. Defaults to 'tl'.
        invert_x (bool, optional): Whether flip along x-axis. Defaults to False.
        invert_y (bool, optional): Whether flip along y-axis. Defaults to False.
        offset_x (float, optional): Where the axis locates. Defaults to 0.5.
        offset_y (float, optional): Where the axis locates. Defaults to 0.5.

    Returns:
        Tensor: Mirrored image.
    """
    # Check input validity
    check_arg_choice(mirror_axis, ['x', 'y', 'corner'], arg_name='mirror_axis')
    check_arg_choice(corner_type, ['tl', 'tr', 'bl', 'br'], arg_name='corner_type')

    # Input image dimensions
    res_h, res_w = img_in.shape[2], img_in.shape[3]

    # Helper function that calculates copy slices
    def get_copy_slice(offset: int, dim_size: int) -> Tuple[int, int, int]:
        dst_left = max(offset, 0)
        length = min(offset, 0) + dim_size - dst_left
        src_left = dim_size - length if dst_left == 0 else 0
        return src_left, dst_left, length

    # Horizontal/vertical mirror
    if mirror_axis in 'xy':

        # Determine tensor dimension, offset, and invert flag given the mirror axis option
        dim = 3 if mirror_axis == 'x' else 2
        dim_size = (res_h, res_w)[dim - 2]
        offset = (offset_x, 1.0 - offset_y)[3 - dim]
        invert = (invert_x, invert_y)[3 - dim]

        # Offset the mirrored half of the image
        img_flip = th.flip(img_in, [dim])
        img_fg = th.zeros_like(img_flip)
        offset, offset_const = to_tensor_and_const(offset)
        offset_flip = (offset * 2 - 1) * dim_size
        offset_flip_const = (offset_const * 2 - 1) * dim_size
        i_src, i_dst, i_length = get_copy_slice(int(math.floor(offset_flip_const)), dim_size)
        j_src, j_dst, j_length = get_copy_slice(int(math.ceil(offset_flip_const)), dim_size)

        if i_length == j_length:
            img_fg.narrow(dim, i_dst, i_length).copy_(img_flip.narrow(dim, i_src, i_length))
        else:
            weight = offset_flip % 1.0
            img_fg.narrow(dim, i_dst, i_length).copy_(
                img_flip.narrow(dim, i_src, i_length) * (1.0 - weight))
            img_fg.narrow(dim, j_dst, j_length).add_(
                img_flip.narrow(dim, j_src, j_length) * weight)

        # Blend the original image and the mirrored one
        offset_blend = offset * dim_size
        offset_blend_const = offset_const * dim_size
        weights_blend = th.arange(dim_size) + (1.0 - offset_blend % 1.0) - int(offset_blend_const)
        weights_blend = th.atleast_2d(th.clamp(weights_blend, 0.0, 1.0))
        weights_blend = 1.0 - weights_blend if invert else weights_blend
        img_out = th.lerp(img_in, img_fg, weights_blend.T if dim == 2 else weights_blend)

    # Center mirror
    elif mirror_axis == 'corner':
        img_out = img_in.clone()

        # Vertical mirror first
        res_h //= 2
        top_half, bottom_half = img_out.narrow(2, 0, res_h), img_out.narrow(2, res_h, res_h)
        if corner_type.startswith('t'):
            bottom_half.copy_(th.flip(top_half, [2]))
        else:
            top_half.copy_(th.flip(bottom_half, [2]))

        # Horizontal mirror next
        res_w //= 2
        left_half, right_half = img_out.narrow(3, 0, res_w), img_out.narrow(3, res_w, res_w)
        if corner_type.endswith('l'):
            right_half.copy_(th.flip(left_half, [3]))
        else:
            left_half.copy_(th.flip(right_half, [3]))

    return img_out


@input_check(1)
def make_it_tile_patch(img_in: th.Tensor, octave: int = 3, mask_size: FloatValue = 1.0,
                       mask_precision: FloatValue = 0.5, mask_warping: FloatValue = 0.0,
                       pattern_width: FloatValue = 200.0, pattern_height: FloatValue = 200.0,
                       disorder: FloatValue = 0.0, size_variation: FloatValue = 0.0,
                       rotation: FloatValue = 0.0, rotation_variation: FloatValue = 0.0,
                       background_color: FloatVector = [0.0, 0.0, 0.0, 1.0],
                       color_variation: FloatValue = 0.0) -> th.Tensor:
    """Non-atomic node: Make it Tile Patch (Color and Grayscale)

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        octave (int, optional): Logarithm of the tiling factor (by 2). Defaults to 3.
        use_alpha (bool, optional): Enable the alpha channel. Defaults to False.
        mask_size (float, optional): Size of the round mask used when stamping the patch.
            Defaults to 1.0.
        mask_precision (float, optional): Falloff/smoothness precision of the mask.
            Defaults to 0.5.
        mask_warping (float, optional): Warping intensity at mask edges. Defaults to 0.0.
        pattern_width (float, optional): Width of the patch. Defaults to 200.0.
        pattern_height (float, optional): Height of the patch. Defaults to 200.0.
        disorder (float, optional): Translational randomness. Defaults to 0.0.
        size_variation (float, optional): Size variation for the mask. Defaults to 0.0.
        rotation (float, optional): Rotation angle of the patch (in turning number).
            Defaults to 0.0.
        rotation_variation (float, optional): Randomness in rotation for every patch stamp.
            Defaults to 0.0.
        background_color (list, optional): Background color for areas where no patch appears.
            Defaults to [0.0, 0.0, 0.0, 1.0].
        color_variation (float, optional): Color (or luminosity) variation per patch.
            Defaults to 0.0.

    Returns:
        Tensor: Image with stamped patches of the input.
    """
    _, num_channels, num_rows, num_cols = img_in.shape

    # Process input parameters
    mask_size = to_tensor(mask_size)
    mask_precision = to_tensor(mask_precision)
    mask_warping, mask_warping_const = to_tensor_and_const(mask_warping)
    pattern_w = to_tensor(pattern_width * 0.01)
    pattern_h = to_tensor(pattern_height * 0.01)
    disorder, disorder_const = to_tensor_and_const(disorder)
    size_var, size_var_const = to_tensor_and_const(size_variation * 0.01)
    rotation = to_tensor(rotation * 0.0028)
    rotation_var, rotation_var_const = to_tensor_and_const(rotation_variation * 0.0028)
    background_color = resize_color(th.atleast_1d(to_tensor(background_color)), num_channels)
    color_var, color_var_const = to_tensor_and_const(color_variation)

    grid_size = 1 << octave
    num_patches = grid_size * grid_size * 2

    # Mode switch
    mode_color = num_channels > 1

    ## For Make It Tile Patch Grayscale nodes, background color is replaced by color variation
    ## This is a bug in Substance Designer that still hasn't been fixed as of today
    if not mode_color:
        background_color = resize_color(color_var.clamp(0.0, 1.0).unsqueeze(0), num_channels)

    # Gaussian pattern (44.8 accounts for the 1.4x pattern size)
    x = th.linspace(-31 / 44.8, 31 / 44.8, 32).expand(32, 32)
    x = x ** 2 + x.T ** 2
    img_gs = th.exp(x / -0.09).expand(1, 1, 32, 32)
    img_gs = automatic_resize(img_gs, int(math.log2(img_in.shape[2])) - 5)
    img_gs = levels(img_gs, 1.0 - mask_size, 0.5, 1 - mask_precision * mask_size)

    # Add alpha channel
    if mask_warping_const != 0.0:
        img_in_gc = c2g(img_in, rgba_weights=[0.3, 0.59, 0.11, 0.0]) if mode_color else img_in
        img_a = d_blur(img_in_gc, intensity=1.6)
        img_a = d_blur(img_a, intensity=1.6, angle=0.125)
        img_a = d_blur(img_a, intensity=1.6, angle=0.25)
        img_a = d_blur(img_a, intensity=1.6, angle=0.875)
        img_a = warp(img_gs, img_a, mask_warping * 0.05)
    else:
        img_a = img_gs

    img_patch = img_in.narrow(1, 0, 3) if mode_color else img_in.expand(-1, 3, -1, -1)
    img_patch = th.cat([img_patch, img_a], dim=1)

    # Sample random color, scale, translation, and rotation variations among patches
    colors = th.ones(1, 3)
    offsets = th.zeros(1, 2)
    sizes = th.stack((pattern_w, pattern_h)).unsqueeze(0)
    rotations = rotation.view(1, 1)
    color_rands, offset_rands, size_rands_1, size_rands_2, rotation_rands = \
        th.rand(num_patches, 7).split((3, 1, 1, 1, 1), dim=1)

    if color_var_const:
        colors = th.clamp(colors - color_rands * color_var, 0.0, 1.0)
    if disorder_const:
        angles = offset_rands * 6.28
        offsets = disorder * th.cat((th.cos(angles), th.sin(angles)), dim=1)
    if size_var_const:
        sizes = (size_rands_1 - size_rands_2) * size_var + sizes
    if rotation_var_const:
        rotations = rotation_rands * rotation_var + rotations

    # Compute pattern center positions in rendering order
    from diffmat.core.fxmap.util import get_pattern_pos
    pos = get_pattern_pos(octave)
    offsets = (pos.repeat_interleave(2, dim=0) + offsets)

    # Generate the background image
    img_bg = uniform_color(res_h=num_rows, res_w=num_cols, use_alpha=True, rgba=background_color)

    # Initiate an FX-map executor
    from diffmat.core.fxmap import FXMapExecutorV2 as FXE
    executor = FXE(int(math.log2(num_rows)), device=img_in.device)
    executor.reset(img_bg, img_patch, mode='color')

    # Assemble the FX-map job array
    job_arr: FXMapJobArray = {
        'color': th.cat((colors, th.ones(colors.shape[0], 1)), dim=1).expand(num_patches, -1),
        'offset': offsets.expand(num_patches, -1),
        'size': sizes.expand(num_patches, -1),
        'rotation': rotations.expand(num_patches, -1),
        'variation': [0.0],
        'depth': [octave],
        'blending': [FXE.BLEND_MAX_COPY],
        'filtering': [FXE.FILTER_BILINEAR_MIPMAPS],
        'image_index': [0],
    }

    # Execute the FX-map
    blending_opacity = np.ones(octave + 1, dtype=np.float32)
    fx_map = executor.evaluate(blending_opacity, batched_jobs={'image': job_arr})

    # Output channel conversion (if needed)
    img_out = resize_image_color(fx_map, num_channels) if mode_color else c2g(fx_map)

    return img_out


@input_check(1)
def make_it_tile_photo(img_in: th.Tensor, mask_warping_x: FloatValue = 0.0,
                       mask_warping_y: FloatValue = 0.0, mask_size_x: FloatValue = 0.1,
                       mask_size_y: FloatValue = 0.1, mask_precision_x: FloatValue = 0.5,
                       mask_precision_y: FloatValue = 0.5) -> th.Tensor:
    """Non-atomic node: Make it Tile Photo (Color and Grayscale)

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        mask_warping_x (float, optional): Warping intensity on the X-axis. Defaults to 0.0.
        mask_warping_y (float, optional): Warping intensity on the Y-axis. Defaults to 0.0.
        mask_size_x (float, optional): Width of the transition edge. Defaults to 0.1.
        mask_size_y (float, optional): Height of the transition edge. Defaults to 0.1.
        mask_precision_x (float, optional): Smoothness of the horizontal transition.
            Defaults to 0.5.
        mask_precision_y (float, optional): Smoothness of the vertical transition.
            Defaults to 0.5.

    Returns:
        Tensor: Image with fixed tiling behavior.
    """
    # Convert parameters to tensors
    mask_warping_x, mask_warping_x_const = to_tensor_and_const(mask_warping_x)
    mask_warping_y, mask_warping_y_const = to_tensor_and_const(mask_warping_y)
    mask_size_x = to_tensor(mask_size_x)
    mask_size_y = to_tensor(mask_size_y)
    mask_precision_x = to_tensor(mask_precision_x)
    mask_precision_y = to_tensor(mask_precision_y)

    # Split channels
    num_channels = img_in.shape[1]
    img_gs = c2g(img_in.narrow(1, 0, 3)) if num_channels >= 3 else img_in

    # Create pyramid pattern
    res = img_in.shape[2]
    vec_grad = th.linspace(1 / res - 1, 1 - 1 / res, res).abs()
    img_grad_x = vec_grad.unsqueeze(1)
    img_grad_y = vec_grad.unsqueeze(0)
    img_pyramid = th.clamp_max((1.0 - th.max(img_grad_x, img_grad_y)) * 7.39, 1.0)
    img_pyramid = img_pyramid.expand(1, 1, -1, -1)

    # Create cross mask
    img_grad = 1.0 - vec_grad ** 2
    img_grad_x = img_grad.unsqueeze(1).expand(1, 1, res, res)
    img_grad_y = img_grad.unsqueeze(0).expand(1, 1, res, res)
    img_grad_x = levels(img_grad_x, 1.0 - mask_size_x, 0.5, 1.0 - mask_size_x * mask_precision_x)
    img_grad_y = levels(img_grad_y, 1.0 - mask_size_y, 0.5, 1.0 - mask_size_y * mask_precision_y)

    img_gs = blur_hq(img_gs.view(1, 1, res, res), intensity=2.75)
    if mask_warping_x_const != 0:
        img_grad_x = d_warp(img_grad_x, img_gs, intensity=mask_warping_x, angle=0.25)
        img_grad_x = d_warp(img_grad_x, img_gs, intensity=mask_warping_x, angle=-0.25)
    if mask_warping_y_const != 0:
        img_grad_y = d_warp(img_grad_y, img_gs, intensity=mask_warping_y)
        img_grad_y = d_warp(img_grad_y, img_gs, intensity=mask_warping_y, angle=0.5)

    img_cross = (img_pyramid * th.max(img_grad_x, img_grad_y)).clamp(0.0, 1.0)

    # Create sphere mask
    img_grad = vec_grad ** 2 * 16
    img_grad = th.clamp_min(1.0 - img_grad.unsqueeze(0) - img_grad.unsqueeze(1), 0.0)
    img_sphere = (img_grad.roll(res >> 1, 0) + img_grad.roll(res >> 1, 1)).expand(1, 1, -1, -1)
    img_sphere = warp(img_sphere, img_gs, 0.24)

    # Fix tiling for an image
    img = th.lerp(img_in.roll((res >> 1, res >> 1), dims=(2, 3)), img_in, img_cross)
    img_bg = img.roll((res >> 1, res >> 1), dims=(2, 3))
    img_fg = img.roll((-(res >> 2), -(res >> 2)), dims=(2, 3)) if res >= 4 else \
             transform_2d(img, offset=[0.25, 0.25])
    img_out = th.lerp(img_bg, img_fg, img_sphere)

    return img_out


@input_check(1, channel_specs='c')
def replace_color(img_in: th.Tensor, source_color: FloatVector = [0.5, 0.5, 0.5],
                  target_color: FloatVector = [0.5, 0.5, 0.5]) -> th.Tensor:
    """Non-atomic node: Replace Color

    Args:
        img_in (tensor): Input image (RGB(A) only).
        source_color (list, optional): Color to start hue shifting from.
            Defaults to [0.5, 0.5, 0.5].
        target_color (list, optional): Color where hue shifting ends.
            Defaults to [0.5, 0.5, 0.5].

    Returns:
        Tensor: Replaced color image.
    """
    # Convert source colors to HSL
    target_hsl = color_from_rgb(target_color)
    source_hsl = color_from_rgb(source_color)

    # Apply HSL difference to the input image
    diff_hsl = (target_hsl - source_hsl) * 0.5 + 0.5
    dh, ds, dl = diff_hsl.unbind()
    img_out = hsl(img_in, hue=dh, saturation=ds, lightness=dl)

    return img_out


def normal_color(normal_format: str = 'dx', num_imgs: int = 1, res_h: int = 512, res_w: int = 512,
                 use_alpha: bool = False, direction: FloatValue = 0.0,
                 slope_angle: FloatValue = 0.0) -> th.Tensor:
    """Non-atomic node: Normal Color

    Args:
        normal_format (str, optional): Normal format ('dx' | 'gl'). Defaults to 'dx'.
        num_imgs (int, optional): Batch size. Defaults to 1.
        res_h (int, optional): Resolution in height. Defaults to 512.
        res_w (int, optional): Resolution in width. Defaults to 512.
        use_alpha (bool, optional): Enable the alpha channel. Defaults to False.
        direction (float, optional): Normal direction (in turning number). Defaults to 0.0.
        slope_angle (float, optional): Normal slope angle (in turning number). Defaults to 0.0.

    Returns:
        Tensor: Uniform normal color image.
    """
    # Check input validity
    check_arg_choice(normal_format, ['dx', 'gl'], arg_name='normal_format')

    # Convert parameters to tensors (turns to radians)
    dir_angle = to_tensor(direction) * (math.pi * 2)
    slope_angle = to_tensor(slope_angle) * (math.pi * 2)

    # Calculate normal color and expand into an image
    cos_angle = -th.cos(dir_angle)
    sin_angle = th.sin(dir_angle if normal_format == 'gl' else -dir_angle)
    vec = th.stack([cos_angle, sin_angle]) * th.sin(slope_angle) * 0.5 + 0.5
    rgba = th.cat([vec, th.ones(2)])
    img_out = uniform_color(
        num_imgs=num_imgs, res_h=res_h, res_w=res_w, use_alpha=use_alpha, rgba=rgba)

    return img_out


@input_check(2, channel_specs='.c')
def vector_morph(img_in: th.Tensor, vector_field: Optional[th.Tensor] = None,
                 amount: FloatValue = 1.0) -> th.Tensor:
    """Non-atomic node: Vector Morph (Color and Grayscale)

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        vector_field (tensor, optional): Vector map that drives warping. Defaults to None.
        amount (float, optional): Normalized warping intensity as a multiplier for the vector map.
            Defaults to 1.0.

    Returns:
        Tensor: Warped image.
    """
    # Resize vector field input to only use the first two channels
    num_channels = img_in.shape[1]
    if vector_field is None:
        vector_field = img_in.expand(-1, 2, -1, -1) if num_channels == 1 else img_in[:,:2]
    else:
        vector_field = vector_field[:,:2]

    # Convert parameters to tensors
    amount, amount_const = to_tensor_and_const(amount)

    # Special case - no effect when the amount is zero
    if amount_const == 0.0:
        return img_in

    # Progressive vector field sampling
    res_h, res_w = img_in.shape[2], img_in.shape[3]
    sample_grid = get_pos(res_h, res_w).unsqueeze(0)

    vector_field_pad = pad2d(vector_field, 1)
    size_scales = to_tensor([res_w / (res_w + 2), res_h / (res_h + 2)])

    for i in range(16):
        if i == 0:
            vec = vector_field
        else:
            sample_grid_sp = (sample_grid * 2.0 - 1.0) * size_scales
            vec = grid_sample_impl(vector_field_pad, sample_grid_sp, align_corners=False)
        sample_grid = (sample_grid + (vec.movedim(1, 3) - 0.5) * (amount * 0.0625)) % 1

    # Final image sampling
    sample_grid = (sample_grid * 2.0 - 1.0) * size_scales
    img_out = grid_sample_impl(pad2d(img_in, 1), sample_grid, align_corners=False)

    return img_out


@input_check(2, channel_specs='.c')
def vector_warp(img_in: th.Tensor, vector_map: Optional[th.Tensor] = None,
                vector_format: str = 'dx', intensity: FloatValue = 1.0) -> th.Tensor:
    """Non-atomic node: Vector Warp (Color and Grayscale)

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        vector_map (tensor, optional): Distortion driver map (RGB(A) only). Defaults to None.
        vector_format (str, optional): Normal format of the vector map ('dx' | 'gl').
            Defaults to 'dx'.
        intensity (float, optional): Normalized intensity multiplier of the vector map.
            Defaults to 1.0.

    Returns:
        Tensor: Distorted image.
    """
    # Check input validity
    check_arg_choice(vector_format, ['dx', 'gl'], arg_name='vector_format')

    # Resize vector map input to only use the first two channels
    num_channels = img_in.shape[1]
    if vector_map is None:
        vector_map = img_in.expand(-1, 2, -1, -1) if num_channels == 1 else img_in[:,:2]
    else:
        vector_map = vector_map[:,:2]

    # Convert parameters to tensors
    intensity, intensity_const = to_tensor_and_const(intensity)

    # Special case - no effect when intensity is zero
    if intensity_const == 0.0:
        return img_in

    # Calculate displacement field
    vector_map = vector_map * 2.0 - 1.0
    if vector_format == 'gl':
        vector_map.select(1, 1).neg_()
    vector_map = vector_map * th.norm(vector_map, dim=1, keepdim=True) * intensity

    # Sample input image
    res_h, res_w = img_in.shape[2], img_in.shape[3]
    sample_grid = get_pos(res_h, res_w).unsqueeze(0) + vector_map.movedim(1, 3)
    img_out = grid_sample(img_in, sample_grid, sbs_format=True)

    return img_out


@input_check(1)
def contrast_luminosity(img_in: th.Tensor, contrast: FloatValue = 0.0,
                        luminosity: FloatValue = 0.0) -> th.Tensor:
    """Non-atomic node: Contrast/Luminosity (Color and Grayscale)

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        contrast (float, optional): Contrast of the result. Defaults to 0.0.
        luminosity (float, optional): Brightness of the result. Defaults to 0.0.

    Returns:
        Tensor: Adjusted image.
    """
    # Process input parameters
    contrast, luminosity = to_tensor(contrast), to_tensor(luminosity)

    # Perform histogram adjustment
    in_low = th.clamp(contrast * 0.5, 0.0, 0.5)
    in_high = th.clamp(1.0 - contrast * 0.5, 0.5, 1.0)

    contrast_half = th.abs(contrast.clamp_max(0.0)) * 0.5
    out_low = th.clamp(contrast_half + luminosity, 0.0, 1.0)
    out_high = th.clamp(luminosity + 1.0 - contrast_half, 0.0, 1.0)

    img_out = levels(img_in, in_low=in_low, in_high=in_high, out_low=out_low, out_high=out_high)

    return img_out


@input_check(1, channel_specs='c')
def p2s(img_in: th.Tensor) -> th.Tensor:
    """Non-atomic node: Pre-Multiplied to Straight

    Args:
        img_in (tensor): Image with pre-multiplied color (RGB(A) only).

    Returns:
        Tensor: Image with straight color.
    """
    # Split alpha from input
    rgb, a = img_in.split(3, dim=1) if img_in.shape[1] == 4 else (img_in, None)
    img_out = th.cat(((rgb / (a + 1e-15)).clamp(0.0, 1.0), a), dim=1) if a is not None else rgb

    return img_out


@input_check(1, channel_specs='c')
def s2p(img_in: th.Tensor) -> th.Tensor:
    """Non-atomic node: Straight to Pre-Multiplied

    Args:
        img_in (tensor): Image with straight color (RGB(A) only).

    Returns:
        Tensor: Image with pre-multiplied color.
    """
    # Split alpha from input
    rgb, a = img_in.split(3, dim=1) if img_in.shape[1] == 4 else (img_in, None)
    img_out = th.cat((rgb * a, a), dim=1) if a is not None else rgb

    return img_out


@input_check(1)
def clamp(img_in: th.Tensor, clamp_alpha: bool = True, low: FloatValue = 0.0,
          high: FloatValue = 1.0) -> th.Tensor:
    """Non-atomic node: Clamp (Color and Grayscale)

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        clamp_alpha (bool, optional): Clamp the alpha channel. Defaults to True.
        low (float, optional): Lower clamp limit. Defaults to 0.0.
        high (float, optional): Upper clamp limit. Defaults to 1.0.

    Returns:
        Tensor: Clamped image.
    """
    # Convert parameters to tensors
    low, high = to_tensor(low).clamp(0.0, 1.0), to_tensor(high).clamp(0.0, 1.0)

    # Split alpha from input if alpha isn't clamped
    if img_in.shape[1] == 4 and not clamp_alpha:
        img_rgb, img_a = img_in.split(3, dim=1)
        img_out = th.cat((img_rgb.clamp(low, high), img_a), dim=1)

    # Clamp all channels
    else:
        img_out = img_in.clamp(low, high)

    return img_out


@input_check(1)
def pow(img_in: th.Tensor, exponent: FloatValue = 4.0) -> th.Tensor:
    """Non-atomic node: Pow (Color and Grayscale)

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        exponent (float, optional): Normalized exponent of the power function. Defaults to 4.0.

    Returns:
        Tensor: Powered image.
    """
    # Convert parameters to tensors
    exponent, exponent_const = to_tensor_and_const(exponent)

    # Split alpha from input
    img_in, img_in_alpha = img_in.split(3, dim=1) if img_in.shape[1] == 4 else (img_in, None)

    # Gamma correction
    in_mid = (exponent - 1.0) / 16.0 + 0.5 if exponent_const >= 1.0 else \
             0.5625 if exponent_const == 0 else (1.0 / exponent - 9.0) / -16.0
    img_out = levels(img_in, in_mid=in_mid)

    # Attach the original alpha channel
    img_out = th.cat([img_out, img_in_alpha], dim=1) if img_in_alpha is not None else img_out

    return img_out


@input_check(1)
def quantize(img_in: th.Tensor, grayscale_flag: bool = False, quantize_gray: int = 3,
             quantize_r: int = 4, quantize_g: int = 4, quantize_b: int = 4,
             quantize_alpha: int = 4) -> th.Tensor:
    """Non-atomic node: Quantize (Color and Grayscale)

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        quantize_number (int or list, optional): Number of quantization steps (a list input
            controls each channel separately). Defaults to 3.

    Returns:
        Tensor: Quantized image.
    """
    # Distinguish between color mode and grayscale mode
    if grayscale_flag:
        quantize_number = quantize_gray
        grayscale_input_check(img_in, 'img_in')
    else:
        quantize_number = [quantize_r, quantize_g, quantize_b, quantize_alpha]
        color_input_check(img_in, 'img_in')

    # Per-channel quantization
    qn = (to_tensor(quantize_number) - 1) / 255.0
    qt_shift = 1.0 - 286.0 / 512.0
    img_in = levels(img_in, out_high=qn)
    img_qt = th.floor(img_in * 255.0 + qt_shift) / 255.0
    img_out = levels(img_qt, in_high=qn)

    return img_out


@input_check(1)
def anisotropic_blur(img_in: th.Tensor, high_quality: bool = False, intensity: FloatValue = 10.0,
                     anisotropy: FloatValue = 0.5, angle: FloatValue = 0.0) -> th.Tensor:
    """Non-atomic node: Anisotropic Blur (Color and Grayscale)

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        high_quality (bool, optional): Switch between a box blur (False) and an HQ blur (True)
            internally. Defaults to False.
        intensity (float, optional): Directional blur intensity. Defaults to 10.0.
        anisotropy (float, optional): Directionality of the blur. Defaults to 0.5.
        angle (float, optional): Angle of the blur direction (in turning number). Defaults to 0.0.

    Returns:
        Tensor: Anisotropically blurred image.
    """
    # Convert parameters to tensors
    intensity, anisotropy = to_tensor(intensity), to_tensor(anisotropy)
    angle = to_tensor(angle)

    # Two-pass directional blur
    quality_factor = 0.6 if high_quality else 1.0
    img_out = d_blur(img_in, intensity * quality_factor, angle)
    img_out = d_blur(img_out, intensity * (1.0 - anisotropy) * quality_factor, angle + 0.25)
    if high_quality:
        img_out = d_blur(img_out, intensity * quality_factor, angle)
        img_out = d_blur(img_out, intensity * (1.0 - anisotropy) * quality_factor, angle + 0.25)

    return img_out


@input_check(1)
def glow(img_in: th.Tensor, glow_amount: FloatValue = 0.5, clear_amount: FloatValue = 0.5,
         size: FloatValue = 10.0, color: FloatVector = [1.0, 1.0, 1.0, 1.0]) -> th.Tensor:
    """Non-atomic node: Glow (Color and Grayscale)

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        glow_amount (float, optional): Global opacity of the glow effect. Defaults to 0.5.
        clear_amount (float, optional): Cut-off threshold of the glow effect. Defaults to 0.5.
        size (float, optional): Scale of the glow effect. Defaults to 10.0.
        color (list, optional): Color of the glow effect. Defaults to [1.0, 1.0, 1.0, 1.0].

    Returns:
        Tensor: Image with the glow effect.
    """
    # Convert parameters to tensors
    glow_amount, clear_amount = to_tensor(glow_amount), to_tensor(clear_amount)
    size, color = to_tensor(size), to_tensor(color)

    # Calculate glow mask
    num_channels = img_in.shape[1]
    img_mask = (img_in[:,:3] * 0.33).sum(dim=1, keepdim=True) if num_channels > 1 else img_in
    img_mask = levels(img_mask, in_low = clear_amount - 0.01, in_high = clear_amount + 0.01)
    img_mask = blur_hq(img_mask, intensity=size)

    # Apply glow effect to input
    if num_channels > 1:
        img_mask = (img_mask * glow_amount).clamp(0.0, 1.0)
        img_out = blend(color[:num_channels].view(-1, 1, 1).expand_as(img_in), img_in, img_mask,
                        blending_mode='add')
    else:
        img_out = blend(img_mask, img_in, blending_mode='add', opacity=glow_amount)

    return img_out


@input_check(1)
def car2pol(img_in: th.Tensor) -> th.Tensor:
    """Non-atomic node: Cartesian to Polar (Color and Grayscale)

    Args:
        img_in (tensor): Input image in Cartesian coordinates (G or RGB(A)).

    Returns:
        Tensor: Image in polar coordinates.
    """
    # Generate an initial sampling grid (pixel centers)
    res_h, res_w = img_in.shape[2], img_in.shape[3]
    sample_grid = get_pos(res_h, res_w) - 0.5

    # Convert Cartesian to polar coordinates
    radii = th.norm(sample_grid, dim=-1, keepdim=True) * 2.0 % 1.0
    angles = -th.atan2(sample_grid[...,1:], sample_grid[...,:1]) / (math.pi * 2) % 1.0
    sample_grid = th.cat((angles, radii), dim=-1).unsqueeze(0)

    # Sample the input image
    img_out = grid_sample(img_in, sample_grid, sbs_format=True)

    return img_out


@input_check(1)
def pol2car(img_in: th.Tensor) -> th.Tensor:
    """Non-atomic node: Polar to Cartesian (Color and Grayscale)

    Args:
        img_in (tensor): Image in polar coordinates (G or RGB(A)).

    Returns:
        Tensor: Image in Cartesian coordinates.
    """
    # Generate an initial sampling grid (pixel centers)
    res_h, res_w = img_in.shape[2], img_in.shape[3]
    sample_grid = get_pos(res_h, res_w)

    # Convert polar coordinates to Cartesian
    angles, radii = sample_grid.unbind(dim=2)
    angles, radii = angles * (-math.pi * 2.0), radii * 0.5
    sample_grid = th.stack((th.cos(angles), th.sin(angles)), dim=2) * radii.unsqueeze(2) + 0.5

    # Sample the input image
    img_out = grid_sample(img_in, sample_grid.unsqueeze(0), sbs_format=True)

    return img_out


@input_check(1, channel_specs='g')
def normal_sobel(img_in: th.Tensor, normal_format: str = 'dx', use_alpha: bool = False,
                 intensity: FloatValue = 1.0) -> th.Tensor:
    """Non-atomic node: Normal Sobel

    Args:
        img_in (tensor): Input height map (G only).
        normal_format (str, optional): Input normal format. Defaults to 'dx'.
        use_alpha (bool, optional): Output alpha channel. Defaults to False.
        intensity (float, optional): Normal intensity. Defaults to 1.0.

    Returns:
        Tensor: Output normal map.
    """
    # Check input validity
    check_arg_choice(normal_format, ['dx', 'gl'], arg_name='normal_format')

    # Pre-compute scale multipliers
    intensity = to_tensor(intensity)
    res_h, res_w = img_in.shape[2], img_in.shape[3]
    mult_x = intensity * (res_w / 512.0)
    mult_y = intensity * (res_h / 512.0 * (-1 if normal_format == 'dx' else 1))

    # Pre-blur the input image
    img_blur_x = d_blur(img_in, intensity=256 / res_w)
    img_blur_y = d_blur(img_in, intensity=256 / res_h, angle=0.25)

    # Compute normal
    normal_x = (th.roll(img_blur_y, 1, 3) - th.roll(img_blur_y, -1, 3)) * mult_x
    normal_y = (th.roll(img_blur_x, -1, 2) - th.roll(img_blur_x, 1, 2)) * mult_y
    normal = th.cat((normal_x, normal_y, th.ones_like(normal_x)), dim=1)
    img_normal = th.clamp((normal * 0.5 / normal.norm(dim=1, keepdim=True)) + 0.5, 0.0, 1.0)

    # Add output alpha channel
    if use_alpha:
        img_normal = th.cat((img_normal, th.ones_like(normal_x)), dim=1)

    return img_normal


@input_check(2, channel_specs='cg')
def normal_vector_rotation(img_in: th.Tensor, img_map: Optional[th.Tensor] = None,
                           normal_format: str = 'dx', rotation: FloatValue = 0.0) -> th.Tensor:
    """Non-atomic node: Normal Vector Rotation

    Args:
        img_in (tensor): Input normal map (RGB(A) only).
        img_map (tensor, optional): Rotation map (G only). Defaults to 'None'.
        normal_format (str, optional): Input normal format ('dx' or 'gl'). Defaults to 'dx'.
        rotation (float, optional): Normal vector rotation angle. Defaults to 0.0.

    Returns:
        Tensor: Output normal map.
    """
    # Check input validity
    check_arg_choice(normal_format, ['dx', 'gl'], arg_name='normal_format')

    # Substitute empty input to rotation map by zeros
    if img_map is None:
        img_map = th.zeros(1, 1, img_in.shape[2], img_in.shape[3])

    # Rotate normal vector map
    nx, ny, nzw = img_in.tensor_split((1, 2), dim=1)
    nx = nx * 2 - 1
    ny = 1 - ny * 2 if normal_format == 'dx' else ny * 2 - 1

    rotation = to_tensor(rotation)
    angle_rad_map = (img_map + rotation) * (math.pi * 2.0)
    cos_angle, sin_angle = th.cos(angle_rad_map), th.sin(angle_rad_map)

    nx_rot = nx * cos_angle + ny * sin_angle
    ny_rot = ny * cos_angle - nx * sin_angle
    nx_rot = nx_rot * 0.5 + 0.5
    ny_rot = 0.5 - ny_rot * 0.5 if normal_format == 'dx' else ny_rot * 0.5 + 0.5

    # Merge rotated normal components
    img_out = th.cat((nx_rot, ny_rot, nzw), dim=1)

    return img_out


@input_check(1)
def non_square_transform(img_in: th.Tensor, tiling: int = 3, tile_mode: str = 'automatic',
                         tile: Tuple[int, int] = [1,1], tile_safe_rotation: bool = True,
                         offset: FloatVector = [0.0, 0.0], rotation: FloatValue = 0.0,
                         background_color: FloatVector = [0.0, 0.0, 0.0, 1.0]) -> th.Tensor:
    """Non-atomic node: Non-Square Transform (Color and Grayscale)

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        tiling (int, optional): Output tiling (see function 'transform_2d'). Defaults to 3.
        tile_mode (str, optional): Tiling mode control ('automatic' or 'manual').
            Defaults to 'automatic'.
        tile (int, optional): Tiling in [X, Y] direction (if tile_mode is 'manual').
            Defaults to [1, 1].
        tile_safe_rotation (bool, optional): Snaps to safe values to maintain sharpness of pixels.
            Defaults to True.
        offset (float, optional): [X, Y] translation offset. Defaults to 0.0, 0.0.
        rotation (float, optional): Image rotation angle. Defaults to 0.0.
        background_color (list, optional): Background color when tiling is disabled. Defaults to
            [0.0, 0.0, 0.0, 1.0].

    Returns:
        Tensor: Transformed image.
    """
    # Check input validity
    check_arg_choice(tile_mode, ['automatic', 'manual'], arg_name='tile_mode')

    # Convert parameters to tensors
    offset, rotation = to_tensor(offset), to_tensor(rotation)
    background_color = to_tensor(background_color).view(-1)
    background_color = resize_color(background_color, img_in.shape[1])

    # Compute rotation angle
    angle_trunc = th.floor(rotation * 4) * 0.25
    angle = angle_trunc if tile_safe_rotation else rotation
    angle_rad = angle * (math.pi * 2.0)

    # Compute scaling factors
    res_h, res_w = img_in.shape[2], img_in.shape[3]
    angle_remainder_rad = angle_trunc.abs() % 0.25 * (math.pi * 2.0)

    if tile_mode == 'manual':
        rotation_scale = th.cos(angle_remainder_rad) + th.sin(angle_remainder_rad) \
                         if tile_safe_rotation else 1.0
        x_scale = tile[0] * min(res_w / res_h, 1.0) * rotation_scale
        y_scale = tile[1] * min(res_h / res_w, 1.0) * rotation_scale
    else:
        x_scale = max(res_w / res_h, 1.0)
        y_scale = max(res_h / res_w, 1.0)

    # Compute transform matrix
    x1, x2 = x_scale * th.cos(angle_rad), x_scale * -th.sin(angle_rad)
    y1, y2 = y_scale * th.sin(angle_rad), y_scale * th.cos(angle_rad)
    matrix22 = th.stack((x1, x2, y1, y2))

    # Compute translation offset
    img_res = to_tensor([res_w, res_h])
    offset = th.floor((offset - 0.5) * img_res) / img_res + 0.5

    # Initiate 2D transformation
    img_out = transform_2d(img_in, tiling=tiling, mipmap_mode='manual', matrix22=matrix22,
                           offset=offset, matte_color=background_color)

    return img_out


@input_check(1)
def quad_transform(img_in: th.Tensor, culling: str = 'f-b', enable_tiling: bool = False,
                   sampling: str = 'bilinear', p00: FloatVector = [0.0, 0.0],
                   p01: FloatVector = [0.0, 1.0], p10: FloatVector = [1.0, 0.0],
                   p11: FloatVector = [1.0, 1.0],
                   background_color: FloatVector = [0.0, 0.0, 0.0, 1.0]) -> th.Tensor:
    """Non-atomic node: Quad Transform (Color and Grayscale)

    Args:
        img_in (tensor): Input image.
        culling (str, optional): Set culling/hiding of shape when points cross over each other.
            [Options]
                - 'f': front only.
                - 'b': back only.
                - 'f-b': front over back.
                - 'b-f': back over front.
            Defaults to 'f-b'.
        enable_tiling (bool, optional): Enable tiling. Defaults to False.
        sampling (str, optional): Set sampling quality ('bilinear' or 'nearest').
            Defaults to 'bilinear'.
        p00 (list, optional): Top left point. Defaults to [0.0, 0.0].
        p01 (list, optional): Bottom left point. Defaults to [0.0, 1.0].
        p10 (list, optional): Top right point. Defaults to [1.0, 0.0].
        p11 (list, optional): Bottom right point. Defaults to [1.0, 1.0].
        background_color (list, optional): Solid background color if tiling is off.
            Defaults to [0.0, 0.0, 0.0, 1.0].

    Returns:
        Tensor: Transformed image.
    """
    # Check input validity
    check_arg_choice(culling, ['f', 'b', 'f-b', 'b-f'], arg_name='culling')
    check_arg_choice(sampling, ['bilinear', 'nearest'], arg_name='sampling')

    # Convert parameters to tensors
    p00, p01, p10, p11 = to_tensor(p00), to_tensor(p01), to_tensor(p10), to_tensor(p11)
    background_color = to_tensor(background_color).view(-1)
    background_color = resize_color(background_color, img_in.shape[1])

    # Compute a few derived (or renamed) values
    b, c, a = p01 - p00, p10 - p00, p11 - p01
    d = a - c
    x2_1, x2_2 = cross_2d(c, d), cross_2d(b, d)
    x1_a, x1_b = cross_2d(b, c), d
    p0, x0_1, x0_2 = p00, b, c
    enable_2sided = len(culling) > 1 and not enable_tiling
    front_first = culling.startswith('f')

    # Solve quadratic equations
    res_h, res_w = img_in.shape[2], img_in.shape[3]
    pos_offset = p0 - get_pos(res_h, res_w)
    x1_b = cross_2d(pos_offset, x1_b)
    x0_1 = cross_2d(pos_offset, x0_1)
    x0_2 = cross_2d(pos_offset, x0_2)
    qx1, qy1, error1 = solve_poly_2d(x2_1, x1_b - x1_a, x0_1)
    qx2, qy2, error2 = solve_poly_2d(x2_2, x1_b + x1_a, x0_2)

    # Compute sampling positions
    sample_pos_ff = th.stack((qx1, qy2), dim=2)
    sample_pos_bf = th.stack((qy1, qx2), dim=2)
    in_01_ff = th.all((sample_pos_ff >= 0) & (sample_pos_ff <= 1), dim=2)
    in_01_bf = th.all((sample_pos_bf >= 0) & (sample_pos_bf <= 1), dim=2)

    # Determine which face is being considered
    cond_face = th.as_tensor((in_01_ff if front_first else ~in_01_bf) \
                             if enable_2sided else front_first)
    in_01 = th.where(cond_face, in_01_ff, in_01_bf)
    sample_pos = th.where(cond_face.unsqueeze(-1), sample_pos_ff, sample_pos_bf)

    # Perform sampling
    sample_pos = (sample_pos % 1.0).expand(img_in.shape[0], res_h, res_w, 2)
    img_sample = grid_sample(img_in, sample_pos, sbs_format=True)

    # Apply sampling result on background color
    img_bg = background_color.view(-1, 1, 1)
    cond = error1 | error2 | ~(in_01 | enable_tiling)
    img_out = th.where(cond, img_bg, img_sample)

    return img_out


@input_check(1, channel_specs='c')
def chrominance_extract(img_in: th.Tensor) -> th.Tensor:
    """Non-atomic node: Chrominance Extract

    Args:
        img_in (tensor): Input image (RGB(A) only).

    Returns:
        Tensor: Chrominance of the image.
    """
    # Calculate chrominance
    lum = c2g(img_in, rgba_weights=[0.3, 0.59, 0.11, 0.0])
    blend_fg = resize_image_color(1 - lum, img_in.shape[1])
    img_out = blend(blend_fg, img_in, blending_mode="add_sub", opacity=0.5)

    return img_out


@input_check(1, channel_specs='g')
def histogram_shift(img_in: th.Tensor, position: FloatValue = 0.5) -> th.Tensor:
    """Non-atomic node: Histogram Shift

    Args:
        img_in (tensor): Input image (G only)
        position (float, optional): How much to shift the input by. Defaults to 0.5.

    Returns:
        Tensor: Histogram shifted image.
    """
    # Convert parameters to tensors
    position = to_tensor(position)

    # Perform histogram adjustment
    levels_1 = levels(img_in, in_low=position, out_high=1.0-position)
    levels_2 = levels(img_in, in_high=position, out_low=1.0-position)
    levels_3 = levels(img_in, in_low=position, in_high=position)
    img_out = blend(levels_1, levels_2, levels_3)

    return img_out


@input_check(1, channel_specs='g')
def height_map_frequencies_mapper(img_in: th.Tensor, relief: FloatValue = 16.0) -> th.Tensor:
    """Non-atomic node: Height Map Frequencies Mapper

    Args:
        img_in (tensor): Input image (G only).
        relief (float, optional): Controls the displacement output's detail size. Defaults to 16.0.

    Returns:
        Tensor: Blurred displacement map.
        Tensor: Relief parallax map.
    """
    # Convert parameters to tensors
    relief = to_tensor(relief)

    # Compute displacement map and relief parallax map
    blend_fg = th.full_like(img_in, 0.498)
    blend_bg = blur_hq(img_in, intensity=relief)
    displacement = th.lerp(blend_bg, blend_fg, (relief / 32.0).clamp(0.0, 1.0))
    relief_parallax = blend(1 - displacement, img_in, blending_mode='add_sub', opacity=0.5)

    return displacement, relief_parallax


@input_check(1, channel_specs='c')
def luminance_highpass(img_in: th.Tensor, radius: FloatValue = 6.0) -> th.Tensor:
    """Non-atomic node: Luminance Highpass

    Args:
        img_in (tensor): Input image (RGB(A) only).
        radius (float, optional): Radius of the highpass effect. Defaults to 6.0.

    Returns:
        Tensor: Luminance highpassed image.
    """
    # Convert parameters to tensors
    radius = to_tensor(radius)

    # Highpass filtering
    grayscale = c2g(img_in, rgba_weights = [0.3, 0.59, 0.11, 0.0])
    highpassed = highpass(grayscale, radius=radius)
    transformed = transform_2d(grayscale, mipmap_level=12, mipmap_mode='manual')
    blend_fg = blend(highpassed, transformed, blending_mode='add_sub', opacity=0.5)

    # Apply the filtering result to input image
    blend_bg = blend(resize_image_color(1 - grayscale, img_in.shape[1]), img_in,
                     blending_mode='add_sub', opacity=0.5)
    img_out = blend(resize_image_color(blend_fg, img_in.shape[1]), blend_bg,
                    blending_mode='add_sub', opacity=0.5)

    return img_out


@input_check(1, channel_specs='c')
def replace_color_range(img_in: th.Tensor, source_color: FloatVector = [0.501961] * 3,
                        target_color: FloatVector = [0.501961] * 3,
                        source_range: FloatValue = 0.5, threshold: FloatValue = 1.0) -> th.Tensor:
    """Non-atomic node: Replace Color Range

    Args:
        img_in (tensor): Input image (RGB(A) only).
        source_color (list, optional): Color to replace. Defaults to [0.501961] * 3.
        target_color (list, optional): Color to replace with. Defaults to [0.501961] * 3.
        source_range (float, optional): Range or tolerance of the picked Source. Can be increased
            so further neighbouring colours are also hue-shifted. Defaults to 0.5.
        threshold (float, optional): Falloff/contrast for range. Set low to replace only Source
            color, set higher to replace colors blending into Source as well. Defaults to 1.0.

    Returns:
        Tensor: Color replaced image.
    """
    # Convert parameters to tensors
    source_range = to_tensor(source_range)

    # Split alpha from input
    rgb, alpha = img_in.split(3, dim=1) if img_in.shape[1] == 4 else (img_in, None)

    # Compute pixel-wise squared distance from the source color
    source_color_img = uniform_color(
        'color', res_h=rgb.shape[2], res_w=rgb.shape[3], rgba=source_color)
    blend_1 = blend(source_color_img, rgb, blending_mode="subtract")
    blend_2 = blend(rgb, source_color_img, blending_mode="subtract")
    blend_3 = blend(blend_1, blend_2, blending_mode="max")
    blend_4 = blend(blend_3, blend_3, blending_mode="multiply")

    # Determine blending weights from negative distance
    grayscale = 1.0 - blend_4.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    blend_mask = levels(grayscale, in_low = 1 - threshold, out_low = (source_range - 0.5) * 2,
                        out_high = source_range * 2)

    # Blend replaced colors into the input
    blend_fg = replace_color(rgb, source_color, target_color)
    blend_fg = resize_image_color(blend_fg, img_in.shape[1]) if alpha is not None else blend_fg
    img_out = blend(blend_fg, img_in, blend_mask)

    return img_out


@input_check(3, channel_specs='ccg', reduction='any', reduction_range=2)
def dissolve(img_fg: Optional[th.Tensor] = None, img_bg: Optional[th.Tensor] = None,
             mask: Optional[th.Tensor] = None, alpha_blending: bool = True,
             opacity: FloatValue = 1.0) -> th.Tensor:
    """Non-atomic node: Dissolve

    Args:
        img_fg (tensor): Foreground image (RGB(A) only).
        img_bg (tensor): Background image (RGB(A) only).
        mask (tensor, optional): Mask slot used for masking the node's effects (G only).
            Defaults to None.
        alpha_blending (bool, optional): Toggles blending of the Foreground and Background alpha
            channels. If set to False, the alpha channel of the foreground is ignored.
            Defaults to True.
        opacity (float, optional): Blending Opacity between Foreground and Background.
            Defaults to 1.0.

    Returns:
        Tensor: Dissolved image.
    """
    # Generate white noise as blending mask
    white_noise = th.rand(1, 1, img_fg.shape[2], img_fg.shape[3])

    # Blend foreground into the background using the generated mask
    blend_2 = white_noise * mask if mask is not None else white_noise
    blend_3_mask = levels(blend_2, in_low = 1.0 - opacity, in_high = 1.0 - opacity)
    img_out = blend(img_fg, img_bg, blend_3_mask, alpha_blend=alpha_blending)

    return img_out


@input_check(3, channel_specs='ccg', reduction='any', reduction_range=2)
def color_blend(img_fg: Optional[th.Tensor] = None, img_bg: Optional[th.Tensor] = None,
                mask: Optional[th.Tensor] = None, alpha_blending: bool = True,
                opacity: FloatValue = 1.0) -> th.Tensor:
    """Non-atomic node: Color (Blend)

    Args:
        img_fg (tensor): Foreground image (RGB(A) only).
        img_bg (tensor): Background image (RGB(A) only).
        mask (tensor, optional): Mask slot used for masking the node's effects (G only).
            Defaults to None.
        alpha_blending (bool, optional): Toggles blending of the Foreground and Background alpha
            channels. If set to False, the alpha channel of the foreground is ignored.
            Defaults to True.
        opacity (float, optional): Blending Opacity between Foreground and Background.
            Defaults to 1.0.

    Returns:
        Tensor: Color blended image.
    """
    # Convert input images to luminance
    diff_fg, lum_bg = None, None

    if img_fg is not None:
        lum_fg = c2g(img_fg, rgba_weights=[0.3, 0.59, 0.11, 0.0])
        lum_fg = resize_image_color(lum_fg, img_fg.shape[1])
        diff_fg = blend(lum_fg, img_fg, blending_mode='subtract', alpha_blend=False)

    if img_bg is not None:
        use_alpha = img_bg.shape[1] == 4
        grayscale_bg = c2g(img_bg.narrow(1, 0, 3), rgba_weights=[0.3, 0.59, 0.11, 0.0])
        lum_bg = grayscale_bg.expand(-1, 3, -1, -1)
        if use_alpha:
            lum_bg = th.cat((lum_bg, img_bg[:,3:]), dim=1)

    # Combine the luminance maps of both inputs
    blend_2 = blend(diff_fg, lum_bg, blending_mode='add', alpha_blend=alpha_blending)

    if img_bg is not None:
        lum_bg_gm = gradient_map(grayscale_bg, linear_interp = False, use_alpha = use_alpha,
                                 anchors = [[0.0] * 4 + [1.0], [0.101] + [249 / 255] * 3 + [1.0]])
    else:
        lum_bg_gm = resize_image_color(th.zeros_like(img_fg[:,:1]), img_fg.shape[1])

    blend_3 = blend(lum_bg_gm, blend_2, blending_mode='multiply', alpha_blend=False)
    blend_4 = blend(blend_3, img_bg, mask)
    img_out = blend(blend_4, img_bg, opacity=opacity)

    return img_out


@input_check(3, channel_specs='ccg', reduction='any', reduction_range=2)
def color_burn(img_fg: Optional[th.Tensor] = None, img_bg: Optional[th.Tensor] = None,
               mask: Optional[th.Tensor] = None, alpha_blending: bool = True,
               opacity: FloatValue = 1.0) -> th.Tensor:
    """Non-atomic node: Color Burn

    Args:
        img_fg (tensor): Foreground image (RGB(A) only).
        img_bg (tensor): Background image (RGB(A) only).
        mask (tensor, optional): Mask slot used for masking the node's effects (G only).
            Defaults to None.
        alpha_blending (bool, optional): Toggles blending of the Foreground and Background alpha
            channels. If set to False, the alpha channel of the foreground is ignored.
            Defaults to True.
        opacity (float, optional): Blending Opacity between Foreground and Background.
            Defaults to 1.0.

    Returns:
        Tensor: Color burn image.
    """
    # Split alpha from input images and replace empty inputs by zero
    fg_rgb = img_fg[:,:3] if img_fg is not None else th.zeros_like(img_bg[:,:3])
    bg_rgb = img_bg[:,:3] if img_bg is not None else th.zeros_like(img_fg[:,:3])
    fg_alpha = img_fg[:,3:] if img_fg is not None and img_fg.shape[1] == 4 else None
    bg_alpha = img_bg[:,3:] if img_bg is not None and img_bg.shape[1] == 4 else None

    # Color burn blending
    blend_1 = 1 - blend(fg_rgb, 1 - bg_rgb, blending_mode='divide')

    if fg_alpha is not None or bg_alpha is not None:
        blend_alpha = blend(img_fg, img_bg, opacity=opacity)
        blend_1 = th.cat((blend_1, blend_alpha.narrow(1, 3, 1)), dim=1)

    blend_2 = blend(blend_1, img_bg, mask)
    img_out = blend(blend_2, img_bg, alpha_blend=alpha_blending, opacity=opacity)

    return img_out


@input_check(3, channel_specs='ccg', reduction='any', reduction_range=2)
def color_dodge(img_fg: Optional[th.Tensor] = None, img_bg: Optional[th.Tensor] = None,
                mask: Optional[th.Tensor] = None, alpha_blending: bool = True,
                opacity: FloatValue = 1.0) -> th.Tensor:
    """Non-atomic node: Color Dodge

    Args:
        img_fg (tensor): Foreground image (RGB(A) only).
        img_bg (tensor): Background image (RGB(A) only).
        mask (tensor, optional): Mask slot used for masking the node's effects (G only).
            Defaults to None.
        alpha_blending (bool, optional): Toggles blending of the Foreground and Background alpha
            channels. If set to False, the alpha channel of the foreground is ignored.
            Defaults to True.
        opacity (float, optional): Blending Opacity between Foreground and Background.
            Defaults to 1.0.

    Returns:
        Tensor: Color dodge image.
    """
    # Blend RGB channels from input
    fg_rgb = 1 - img_fg[:,:3] if img_fg is not None else th.ones_like(img_bg[:,:3])
    bg_rgb = img_bg[:,:3] if img_bg is not None else th.zeros_like(img_fg[:,:3])
    blend_1 = blend(fg_rgb, bg_rgb, blending_mode='divide')

    # Resize RGB blending result to match input image format
    if any(img is not None and img.shape[1] == 4 for img in (img_fg, img_bg)):
        blend_1 = resize_image_color(blend_1, 4)

    # Apply blending result to background image
    blend_2 = blend(blend_1, img_bg, mask, blending_mode='switch')
    img_out = blend(blend_2, img_bg, blending_mode='switch', alpha_blend=alpha_blending,
                    opacity=opacity)

    return img_out


@input_check(3, channel_specs='ccg', reduction='any', reduction_range=2)
def difference(img_bg: Optional[th.Tensor] = None, img_fg: Optional[th.Tensor] = None,
               mask: Optional[th.Tensor] = None, alpha_blending: bool = True,
               opacity: FloatValue = 1.0) -> th.Tensor:
    """Non-atomic node: Difference

    Args:
        img_fg (tensor): Foreground image (RGB(A) only).
        img_bg (tensor): Background image (RGB(A) only).
        mask (tensor, optional): Mask slot used for masking the node's effects (G only).
            Defaults to None.
        alpha_blending (bool, optional): Toggles blending of the Foreground and Background alpha
            channels. If set to False, the alpha channel of the foreground is ignored.
            Defaults to True.
        opacity (float, optional): Blending Opacity between Foreground and Background.
            Defaults to 1.0.

    Returns:
        Tensor: Difference image.
    """
    # Split alpha from input images and replace empty inputs by zero
    fg_rgb = img_fg[:,:3] if img_fg is not None else th.zeros_like(img_bg[:,:3])
    bg_rgb = img_bg[:,:3] if img_bg is not None else th.zeros_like(img_fg[:,:3])
    fg_alpha = img_fg[:,3:] if img_fg is not None and img_fg.shape[1] == 4 else None
    bg_alpha = img_bg[:,3:] if img_bg is not None and img_bg.shape[1] == 4 else None

    # Blend input RGBs
    blend_1 = (th.max(fg_rgb, bg_rgb) - th.min(fg_rgb, bg_rgb)).clamp(0.0, 1.0)

    # Blend input alpha
    if fg_alpha is not None or bg_alpha is not None:
        blend_alpha = blend(img_bg, img_fg, alpha_blend=alpha_blending, opacity=opacity)
        blend_1 = th.cat((blend_1, blend_alpha[:,3:]), dim=1)

    # Apply blending result to foreground image
    blend_5 = blend(blend_1, img_fg, mask, blending_mode='switch')
    img_out = blend(blend_5, img_fg, blending_mode='switch', opacity=opacity)

    return img_out


@input_check(3, channel_specs='ccg', reduction='any', reduction_range=2)
def linear_burn(img_fg: Optional[th.Tensor] = None, img_bg: Optional[th.Tensor] = None,
                mask: Optional[th.Tensor] = None, alpha_blending: bool = True,
                opacity: FloatValue = 1.0) -> th.Tensor:
    """Non-atomic node: Linear Burn

    Args:
        img_fg (tensor): Foreground image (RGB(A) only).
        img_bg (tensor): Background image (RGB(A) only).
        mask (tensor, optional): Mask slot used for masking the node's effects (G only).
            Defaults to None.
        alpha_blending (bool, optional): Toggles blending of the Foreground and Background alpha
            channels. If set to False, the alpha channel of the foreground is ignored.
            Defaults to True.
        opacity (float, optional): Blending Opacity between Foreground and Background.
            Defaults to 1.0.

    Returns:
        Tensor: Linear burn image.
    """
    # Split alpha from input images and replace empty inputs by zero
    fg_rgb = img_fg[:,:3] if img_fg is not None else th.zeros_like(img_bg[:,:3])
    bg_rgb = img_bg[:,:3] if img_bg is not None else th.zeros_like(img_fg[:,:3])
    fg_alpha = img_fg[:,3:] if img_fg is not None and img_fg.shape[1] == 4 else None
    bg_alpha = img_bg[:,3:] if img_bg is not None and img_bg.shape[1] == 4 else None

    # Blend input RGBs
    blend_1 = (((fg_rgb + bg_rgb) * 0.5 - 0.5).clamp_min(0.0) * 2).clamp_max(1.0)

    # Blend input alpha
    if fg_alpha is not None or bg_alpha is not None:
        blend_alpha = blend(img_fg, img_bg, opacity=opacity)
        blend_1 = th.cat((blend_1, blend_alpha[:,3:]), dim=1)

    # Apply blending result to background image
    blend_5 = blend(blend_1, img_bg, fg_alpha, alpha_blend=alpha_blending) \
              if fg_alpha is not None else img_bg
    blend_6 = blend(blend_5, img_bg, mask, blending_mode='switch')
    img_out = blend(blend_6, img_bg, blending_mode='switch', opacity=opacity)

    return img_out


@input_check(3, channel_specs='ccg', reduction='any', reduction_range=2)
def luminosity(img_fg: Optional[th.Tensor] = None, img_bg: Optional[th.Tensor] = None,
               mask: Optional[th.Tensor] = None, alpha_blending: bool = True,
               opacity: FloatValue = 1.0) -> th.Tensor:
    """Non-atomic node: Luminosity

    Args:
        img_fg (tensor): Foreground image (RGB(A) only).
        img_bg (tensor): Background image (RGB(A) only).
        mask (tensor, optional): Mask slot used for masking the node's effects (G only).
            Defaults to None.
        alpha_blending (bool, optional): Toggles blending of the Foreground and Background alpha
            channels. If set to False, the alpha channel of the foreground is ignored.
            Defaults to True.
        opacity (float, optional): Blending Opacity between Foreground and Background.
            Defaults to 1.0.

    Returns:
        Tensor: Luminosity image.
    """
    # Compute input luminance
    lum_fg = c2g(img_fg) if img_fg is not None else th.zeros_like(img_bg[:,:1])
    lum_bg = 1 - c2g(img_bg) if img_bg is not None else th.ones_like(img_fg[:,:1])

    # Resize luminance maps to match the number of input channels
    num_channels = (img_fg if img_fg is not None else img_bg).shape[1]
    lum_fg = resize_image_color(lum_fg, num_channels)
    lum_bg = resize_image_color(lum_bg, num_channels)

    # Blend luminance into inputs
    blend_1 = blend(lum_bg, img_bg, blending_mode='add_sub', opacity=0.5)
    blend_2 = blend(lum_fg, blend_1, blending_mode='add_sub', opacity=0.5)

    # Apply blending result to background image
    blend_3 = blend(blend_2, img_bg, mask)
    img_out = blend(blend_3, img_bg, alpha_blend=alpha_blending, opacity=opacity)

    return img_out


@input_check(2, channel_specs='.g')
def multi_dir_warp(img_in: th.Tensor, intensity_mask: th.Tensor, mode: str = 'average',
                   directions: int = 4, intensity: FloatValue = 10.0,
                   angle: FloatValue = 0.0) -> th.Tensor:
    """Non-atomic node: Multi Directional Warp (Color and Grayscale)

    Args:
        img_in (tensor): Base map to which the warping will be applied (G or RGB(A)).
        intensity_mask (tensor): Mandatory mask map that drives the intensity of the warping
            effect, must be grayscale.
        mode (str, optional): Sets the Blend mode for consecutive passes. Only has effect if
            Directions is 2 or 4. Defaults to 'average'.
        directions (int, optional): Sets in how many Axes the warp works.
            - 1: Moves in the direction of the Angle, and the opposite of that direction
            - 2: The axis of the angle, plus the perpendicular axis
            - 4: The previous axes, plus 45 degree increments.
            Defaults to 4.
        intensity (float, optional): Sets the intensity of the warp effect, how far to push
            pixels out. Defaults to 10.0.
        angle (float, optional): Sets the Angle or direction in which to apply the Warp effect.
            Defaults to 0.0.

    Returns:
        Tensor: Multi-directional warped image.
    """
    # Check input validity
    check_arg_choice(mode, ['average', 'max', 'min', 'chain'], arg_name='mode')
    check_arg_choice(directions, [1, 2, 4], arg_name='directions')

    # Precompute sampling grids for directional warp
    num_row, num_col = img_in.shape[2], img_in.shape[3]
    sample_grid = get_pos(num_row, num_col)

    angles_list = [0.875, 0.375, 0.625, 0.125, 0.75, 0.25, 0.5, 0.0]
    start_index = {4: 0, 2: 4, 1: 6}[directions]
    end_index = 4 if mode in ('max', 'min') and directions == 4 else len(angles_list)
    angles_list = angles_list[start_index:end_index]

    angles = ((to_tensor(angles_list) + angle) % 1).view(-1, 1, 1, 1, 1)
    angles_rad = angles * (math.pi * 2.0)

    vec_shift = intensity * th.cat((th.cos(angles_rad), th.sin(angles_rad)), dim=-1) / 256
    sample_grids = (sample_grid + intensity_mask.movedim(1, 3) * vec_shift) % 1 * 2 - 1
    sample_grids = sample_grids * to_tensor([num_col / (num_col + 2), num_row / (num_row + 2)])

    # Chain mode warps the input along axes sequentially
    if mode == 'chain':
        img_out = img_in
        for i in range(sample_grids.shape[0]):
            img_pad = pad2d(img_out, 1)
            img_out = grid_sample_impl(img_pad, sample_grids[i], align_corners=False)

    # Other modes warps the input along axes individually and combine the results
    else:

        # Gather warped images in different directions
        imgs_warped = []
        for i in range(0, sample_grids.shape[0], 2):
            img_pad = pad2d(img_in, 1)
            img_out = grid_sample_impl(img_pad, sample_grids[i], align_corners=False)
            img_pad = pad2d(img_out, 1)
            img_out = grid_sample_impl(img_pad, sample_grids[i + 1], align_corners=False)
            imgs_warped.append(img_out)

        # Reduce the axially warped images
        img_warped = th.cat(imgs_warped, dim=1)
        if mode == 'average':
            img_out = img_warped.mean(dim=1, keepdim=True)
        elif mode == 'max':
            img_out = img_warped.max(dim=1, keepdim=True)[0]
        else:
            img_out = img_warped.min(dim=1, keepdim=True)[0]

    return img_out


@input_check(1)
def shape_drop_shadow(img_in: th.Tensor, input_is_pre_multiplied: bool = True,
                      pre_multiplied_output: bool = False, use_alpha: bool = False,
                      angle: FloatValue = 0.25, dist: FloatValue = 0.02, size: FloatValue = 0.15,
                      spread: FloatValue = 0.0, opacity: FloatValue = 0.5,
                      mask_color: FloatVector = [1.0, 1.0, 1.0],
                      shadow_color: FloatVector = [0.0, 0.0, 0.0]) -> Tuple[th.Tensor, th.Tensor]:
    """Non-atomic node: Shape Drop Shadow (Color and Grayscale)

    Args:
        img_in (tensor): Input image.
        input_is_pre_multiplied (bool, optional): Whether the input should be assumed as
            pre-multiplied (color version only). Defaults to True.
        pre_multiplied_output (bool, optional): Whether the output should be pre-multiplied.
            Defaults to False.
        angle (float, optional): Incidence Angle of the (fake) light. Defaults to 0.25.
        dist (float, optional): Distance the shadow drop down to/moves away from the shape.
            Defaults to 0.02.
        size (float, optional): Controls blurring/fuzzines of the shadow. Defaults to 0.15.
        spread (float, optional): Cutoff/treshold for the blurring effect, makes the shadow
            spread away further. Defaults to 0.0.
        opacity (float, optional): Blending Opacity for the shadow effect. Defaults to 0.5.
        mask_color (list, optional): Solid color to be used for the transparency mapped output.
            Defaults to [1.0,1.0,1.0].
        shadow_color (list, optional): Color tint to be applied to the shadow.
            Defaults to [0.0,0.0,0.0].

    Raises:
        ValueError: Input image has an invalid number of channels (not 1, 3, or 4).

    Returns:
        Tensor: Shape drop shadow image.
        Tensor: Shadow mask.
    """
    num_channels, num_row, num_col = img_in.shape[1], img_in.shape[2], img_in.shape[3]

    # Separate alpha from input
    # For grayscale input, treat it as the alpha channel of a uniform color mask
    if num_channels == 1:
        rgb, alpha = uniform_color(res_h=num_row, res_w=num_col, rgba=mask_color), img_in.clone()
        img_in = th.cat((rgb, alpha), dim=1)
    elif num_channels == 3:
        alpha = th.ones_like(img_in[:,:1])
        img_in = th.cat((img_in, alpha), dim=1)
    elif num_channels == 4:
        alpha = img_in[:,3:]
    else:
        raise ValueError(f'Input image has an invalid number of channels: {num_channels}')

    # Convert premultiplied RGB to straight RGB for input
    if input_is_pre_multiplied:
        rgb_straight = blend(alpha.expand(-1, 3, -1, -1), img_in[:,:3], blending_mode='divide')
        alpha_merge_1 = th.cat((rgb_straight, alpha), dim=1)
    else:
        alpha_merge_1 = img_in

    # Compute shadow mask
    angle_rad = (angle - 0.5) * (math.pi * 2.0)
    offset = dist * th.stack((th.cos(angle_rad), th.sin(angle_rad))) * 0.5 + 1.0
    transform_2d_1 = transform_2d(alpha, offset=offset)

    blur_hq_1 = blur_hq(transform_2d_1, intensity = size ** 2 * 64)
    levels_1 = levels(blur_hq_1, in_high = 1.0 - spread, out_high = opacity)
    img_mask = ((1 - alpha) * levels_1).clamp(0.0, 1.0)

    # Create colored shadow mask
    uniform_color_1 = uniform_color(res_h=num_row, res_w=num_col, rgba=shadow_color)
    alpha_merge_2 = th.cat((uniform_color_1, levels_1), dim=1)

    # Helper function for straight alpha blending
    def straight_blend(fg: th.Tensor, bg: th.Tensor) -> th.Tensor:
        (fg_rgb, fg_alpha), (bg_rgb, bg_alpha) = fg.split(3, dim=1), bg.split(3, dim=1)
        bg_alpha = bg_alpha * (1 - fg_alpha)
        out_alpha = fg_alpha + bg_alpha
        out_rgb = (bg_rgb * bg_alpha + fg_rgb * fg_alpha) / out_alpha.clamp_min(1e-15)
        return th.cat((out_rgb, out_alpha), dim=1).clamp(0.0, 1.0)

    # Blend input and colored shadow map
    blend_3 = straight_blend(alpha_merge_1, alpha_merge_2)

    # Convert straight RGB to premultiplied RGB for output
    if pre_multiplied_output:
        out_rgb, out_alpha = blend_3.split(3, dim=1)
        out_rgb = out_rgb * out_alpha
        img_out = th.cat((out_rgb, out_alpha), dim=1) if use_alpha else out_rgb
    else:
        img_out = blend_3 if use_alpha else blend_3[:,:3]

    return img_out, img_mask


@input_check(1)
def shape_glow(img_in: th.Tensor, input_is_pre_multiplied: bool = True,
               pre_multiplied_output: bool = False, use_alpha: bool = False, mode: str = 'soft',
               width: str = 0.25, spread: FloatValue = 0.0, opacity: FloatValue = 0.5,
               mask_color: FloatVector = [1.0, 1.0, 1.0],
               glow_color: FloatVector = [1.0, 1.0, 1.0]) -> Tuple[th.Tensor, th.Tensor]:
    """Non-atomic node: Shape Glow (Color and Grayscale)

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        input_is_pre_multiplied (bool, optional): Whether the input should be assumed as
            pre-multiplied. Defaults to True.
        pre_multiplied_output (bool, optional): Whether the output should be pre-multiplied.
            Defaults to False.
        mode (str, optional): Switches between two accuracy modes. Defaults to 'soft'.
        width (float, optional): Controls how far the glow reaches. Defaults to 0.25.
        spread (float, optional): Cut-off / treshold for the blurring effect, makes the glow appear
            solid close to the shape. Defaults to 0.0.
        opacity (float, optional): Blending Opacity for the glow effect. Defaults to 1.0.
        mask_color (list, optional): Solid color to be used for the transparency mapped output.
            Defaults to [1.0, 1.0, 1.0].
        glow_color (list, optional): Color tint to be applied to the glow.
            Defaults to [1.0, 1.0, 1.0].

    Raises:
        ValueError: Input image has an invalid number of channels (not 1, 2, or 4)

    Returns:
        Tensor: Shape glow image.
        Tensor: Glow mask.
    """
    num_channels, num_row, num_col = img_in.shape[1:]

    # Convert parameters to tensors
    width, width_const = to_tensor_and_const(width)

    # Separate alpha from input
    # For grayscale input, treat it as the alpha channel of a uniform color mask
    if num_channels == 1:
        rgb, alpha = uniform_color(res_h=num_row, res_w=num_col, rgba=mask_color), img_in.clone()
        img_in = th.cat((rgb, alpha), dim=1)
    elif num_channels == 3:
        alpha = th.ones_like(img_in[:,:1])
        img_in = th.cat((img_in, alpha), dim=1)
    elif num_channels == 4:
        alpha = img_in[:,3:]
    else:
        raise ValueError(f'Input image has an invalid number of channels: {num_channels}')

    # Convert premultiplied RGB to straight RGB for input
    if input_is_pre_multiplied and num_channels > 1:
        rgb_straight = blend(alpha.expand(-1, 3, -1, -1), img_in[:,:3], blending_mode='divide')
        alpha_merge_1 = th.cat((rgb_straight, alpha), dim=1)
    else:
        rgb_straight = img_in[:,:3]
        alpha_merge_1 = img_in

    # Compute glow mask
    invert_alpha = 1 - alpha if width_const < 0 else alpha
    alpha_contrast = levels(invert_alpha, in_high=0.03146853)

    if mode == 'soft':
        glow_mask = distance(alpha_contrast, dist = th.abs(width) * 8)
        glow_mask = blur_hq(glow_mask, intensity = width * width * 64)
        glow_mask = linear_to_srgb(glow_mask)
    else:
        glow_mask = distance(alpha_contrast, dist = 128 * width * width)
        glow_mask = blur_hq(glow_mask, intensity = (1 - th.abs(width)) * 2)

    glow_mask = levels(glow_mask, in_high = 1 - spread)
    img_mask = ((1 - invert_alpha) * glow_mask).clamp(0.0, 1.0)

    # Helper function for straight alpha blending
    def straight_blend(fg, bg, mask=None):

        # Opaque inputs
        if fg.shape[1] == 3:
            return th.lerp(bg, fg, mask) if mask is not None else fg

        (fg_rgb, fg_alpha), (bg_rgb, bg_alpha) = fg.split(3, dim=1), bg.split(3, dim=1)
        fg_alpha = fg_alpha * mask if mask is not None else fg_alpha
        bg_alpha = bg_alpha * (1 - fg_alpha)
        out_alpha = fg_alpha + bg_alpha
        out_rgb = (bg_rgb * bg_alpha + fg_rgb * fg_alpha) / out_alpha.clamp_min(1e-15)

        return th.cat((out_rgb, out_alpha), dim=1).clamp(0.0, 1.0)

    # Apply glow color to input image using the glow mask
    img_glow_color = uniform_color(res_h=num_row, res_w=num_col, rgba=glow_color)

    if width_const >= 0:
        img_glow_color = resize_image_color(img_glow_color, 4)
        out_rgb = straight_blend(alpha_merge_1, img_glow_color).narrow(1, 0, 3)
        out_alpha = blend(glow_mask, alpha, blending_mode='max', opacity=opacity)
    else:
        out_alpha = levels(glow_mask, out_high=opacity)
        if input_is_pre_multiplied and num_channels > 1:
            out_rgb = straight_blend(img_glow_color, rgb_straight, out_alpha)
        else:
            img_glow_color = resize_image_color(img_glow_color, 4)
            out_rgb = straight_blend(img_glow_color, img_in, out_alpha).narrow(1, 0, 3)

    # Convert straight RGB to premultiplied RGB for output
    out_rgb = out_rgb * out_alpha if pre_multiplied_output else out_rgb
    img_out = th.cat((out_rgb, out_alpha if width_const >= 0 else alpha), dim=1) \
              if use_alpha else out_rgb

    return img_out, img_mask


@input_check(1)
def swirl(img_in: th.Tensor, tiling: int = 3, amount: FloatValue = 8.0,
          offset: FloatVector = [0.0, 0.0], matrix22 = [1.0, 0.0, 0.0, 1.0]) -> th.Tensor:
    """Non-atomic node: Swirl (Color and Grayscale)

    Args:
        img_in (tensor): Input image.
        tiling (int, optional): Tile mode. Defaults to 3 (horizontal and vertical).
        amount (float, optional): Strength of the swirling effect. Defaults to 8.0.
        offset (float, optional): Translation, default to [0.0, 0.0]
        matrix22 (float, optional): Transformation matrix, default to [1.0, 0.0, 0.0, 1.0].

    Returns:
        Tensor: Swirl image.
    """
    # Convert parameters to tensors
    amount = to_tensor(amount)
    matrix22, offset = to_tensor(matrix22).reshape(2, 2).T, to_tensor(offset)
    matrix22_inv = th.inverse(matrix22)

    # Helper function for position transform
    def transform_position(pos, offset):
        return th.matmul(matrix22, (pos - 0.5).unsqueeze(-1)).squeeze(-1) + 0.5 + offset

    # Helper function for inverse position transform
    def inverse_transform_position(pos, offset):
        return th.matmul(matrix22_inv, (pos - offset - 0.5).unsqueeze(-1)).squeeze(-1) + 0.5

    # Transform initial pixel center positions to swirl space
    num_row, num_col = img_in.shape[2], img_in.shape[3]
    pos = get_pos(num_row, num_col)

    inv_center = inverse_transform_position(to_tensor([0.5, 0.5]), offset)
    pos_inv_offset = pos - (inv_center + 0.5)
    pos_inv = pos_inv_offset.floor() + inv_center + 1

    # Discard out-of-range positions in swirl space
    out_of_bound = (pos_inv_offset > 0) | (pos_inv_offset < -1)
    out_of_bound *= th.as_tensor([not tiling % 2, tiling < 2])
    pos_active = th.where(out_of_bound, inv_center, pos_inv)

    # Construct sampling grid for swirl effect
    pos_trans_1 = -transform_position(pos_active, to_tensor([-0.5, -0.5]))
    pos_trans_2 = transform_position(pos, pos_trans_1)

    dists = th.norm(pos_trans_2 - 0.5, dim=-1, keepdim=True)
    angles = (0.5 - dists).clamp_min(0.0) ** 2 * (math.pi * 2.0 * amount)
    cos_angles, sin_angles = th.cos(angles), th.sin(angles)
    rot_matrices = th.cat((cos_angles, sin_angles, -sin_angles, cos_angles), dim=-1)
    rot_matrices = rot_matrices.unflatten(-1, (2, 2))

    pos_rotated = th.matmul(rot_matrices, (pos_trans_2 - 0.5).unsqueeze(-1)).squeeze(-1) + 0.5
    sample_grid = inverse_transform_position(pos_rotated, pos_trans_1).unsqueeze(0)

    # Perform final image sampling
    img_out = grid_sample(img_in, sample_grid, tiling=tiling, sbs_format=True)

    return img_out


@input_check(1, channel_specs='c')
def curvature_sobel(img_in: th.Tensor, normal_format: str = 'dx',
                    intensity: FloatValue = 0.5) -> th.Tensor:
    """Non-atomic node: Curvature Sobel

    Args:
        img_in (tensor): Input image (RGB(A) only).
        normal_format (str, optional): Normal format. Defaults to 'dx'.
        intensity (float, optional): Intensity of the effect, adjusts contrast. Defaults to 0.5.

    Returns:
        Tensor: Curvature image.
    """
    # Check input validity
    check_arg_choice(normal_format, ['dx', 'gl'], arg_name='normal_format')

    # Invert normal for OpenGL format
    img_in = normal_invert(img_in) if normal_format == 'gl' else img_in

    # Sobel filter
    kernel = to_tensor([[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]).unsqueeze(0)
    sobel = conv2d(pad2d(img_in[:,:2], 1), kernel)

    # Adjust filter intensity
    img_out = (sobel * to_tensor(intensity) * 0.5 + 0.5).clamp(0.0, 1.0)

    return img_out


@input_check(2, channel_specs='cg')
def emboss_with_gloss(img_in: th.Tensor, height: th.Tensor, intensity: FloatValue = 5.0,
                      light_angle: FloatValue = 0.0, gloss: FloatValue = 0.25,
                      highlight_color: FloatVector = [1.0, 1.0, 1.0, 1.0],
                      shadow_color: FloatVector = [0.0, 0.0, 0.0, 1.0]) -> th.Tensor:
    """Non-atomic node: Emboss with Gloss

    Args:
        img_in (tensor): Input image (RGB(A) only).
        height (tensor): Height image (G only).
        intensity (float, optional): Normalized intensity of the highlight. Defaults to 5.
        light_angle (float, optional): Light angle. Defaults to 0.0.
        gloss (float, optional): Glossiness highlight size. Defaults to 0.25.
        highlight_color (list, optional): Highlight color. Defaults to [1.0, 1.0, 1.0, 1.0].
        shadow_color (list, optional): Shadow color. Defaults to [0.0, 0.0, 0.0, 1.0].

    Returns:
        Tensor: Embossed image with gloss.
    """
    num_channels, num_row, num_col = img_in.shape[1], img_in.shape[2], img_in.shape[3]

    # Apply emboss filter to input
    emboss_input = uniform_color(res_h = num_row, res_w = num_col, rgba = [127 / 255] * 3)
    emboss_1 = emboss(emboss_input, height, intensity=intensity, light_angle=light_angle,
                      highlight_color=highlight_color, shadow_color=shadow_color)

    # Add gloss effect
    levels_1 = levels(emboss_1, in_low=0.503831)
    levels_2 = levels(emboss_1, in_high=0.484674, out_low=1.0, out_high=0.0)
    blur_hq_1 = blur_hq(c2g(levels_1), intensity=1.5)
    warp_1 = warp(levels_1, blur_hq_1, intensity=-gloss)

    # Blend the emboss with gloss map into input
    blend_1 = blend(resize_image_color(warp_1, num_channels), img_in, blending_mode='add')
    img_out = blend(resize_image_color(levels_2, num_channels), blend_1, blending_mode='subtract')

    return img_out


@input_check(1, channel_specs='c')
def facing_normal(img_in: th.Tensor) -> th.Tensor:
    """Non-atomic node: Facing Normal

    Args:
        img_in (tensor): Input image (RGB(A) only).

    Returns:
        Tensor: Facing normal image.
    """
    # Drop alpha channel
    img_in = img_in[:,:3] if img_in.shape[1] == 4 else img_in

    # Compute the product of X and Y normal components
    img_pos, img_neg = levels(img_in, in_low=0.5), levels(img_in, in_high=0.5)
    img_diff = blend(img_pos, img_neg, blending_mode='subtract')
    img_out = img_diff[:,:1] * img_diff[:,1:2]

    return img_out


@input_check(2, channel_specs='gc')
def height_normal_blend(img_height: th.Tensor, img_normal: th.Tensor, normal_format: str = 'dx',
                        normal_intensity: FloatValue = 0.0) -> th.Tensor:
    """Non-atomic node: Height Normal Blender

    Args:
        img_height (tensor): Grayscale Heightmap to blend with (G only).
        img_normal (tensor): Base Normalmap to blend onto (RGB(A) only).
        normal_format (str, optional): Normal format. Defaults to 'dx'.
        normal_intensity (float, optional): normal intensity. Defaults to 0.0.

    Returns:
        Tensor: Height normal blender image.
    """
    # Check input validity
    check_arg_choice(normal_format, ['dx', 'gl'], arg_name='normal_format')

    # Convert height map to normal
    normal_1 = normal(img_height, normal_format=normal_format, intensity=normal_intensity)

    # Blend two normal maps
    blend_rgb = blend(normal_1, img_normal[:,:3], blending_mode='add_sub', opacity=0.5)
    blend_rgb[:,2:] = (normal_1[:,2:] * img_normal[:,2:3]).clamp(0.0, 1.0)
    img_blend = resize_image_color(blend_rgb, 4) if img_normal.shape[1] == 4 else blend_rgb

    # Normalize output normals
    img_out = normal_normalize(img_blend)

    return img_out


@input_check(1, channel_specs='c')
def normal_invert(img_in: th.Tensor, invert_red: bool = False, invert_green: bool = True,
                  invert_blue: bool = False, invert_alpha: bool = False) -> th.Tensor:
    """Non-atomic node: Normal Invert

    Args:
        img_in (tensor): Normal image (RGB(A) only).
        invert_red (bool, optional): invert red channel flag. Defaults to False.
        invert_green (bool, optional): invert green channel flag. Defaults to True.
        invert_blue (bool, optional): invert blue channel flag. Defaults to False.
        invert_alpha (bool, optional): invert alpha channel flag. Defaults to False.

    Returns:
        Tensor: Normal inverted image.
    """
    invert_mask = [invert_red, invert_green, invert_blue, invert_alpha][:img_in.shape[1]]
    img_out = th.where(th.as_tensor(invert_mask).view(-1, 1, 1), 1 - img_in, img_in)

    return img_out


@input_check(1)
def skew(img_in: th.Tensor, tiling: int = 3, axis: str = 'horizontal', align: str = 'top_left',
         amount: FloatValue = 0.0) -> th.Tensor:
    """Non-atomic node: Skew (Color and Grayscale)

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        axis (str, optional): Choose to skew vertically or horizontally. Defaults to 'horizontal'.
        align (str, optional): Sets the origin point of the Skew transformation.
            Defaults to 'top_left'.
        amount (int, optional): Amount of skew. Defaults to 0.

    Returns:
        Tensor: Skewed image.
    """
    # Check input validity
    check_arg_choice(axis, ['horizontal', 'vertical'], arg_name='axis')
    check_arg_choice(align, ['center', 'top_left', 'bottom_right'], arg_name='align')

    # Convert parameters to tensors
    amount = to_tensor(amount)

    # Transformation matrix
    matrix22 = th.eye(2).flatten()
    dim = ['horizontal', 'vertical'].index(axis)
    matrix22[2 - dim] = amount

    # Offset vector
    offset = th.zeros(2)
    if align == 'top_left':
        offset[dim] = 0.5 * amount
    elif align == 'bottom_right':
        offset[dim] = -0.5 * amount

    # Perform 2D transformation
    img_out = transform_2d(img_in, tiling=tiling, matrix22=matrix22, offset=offset)

    return img_out


@input_check(1)
def trapezoid_transform(img_in: th.Tensor, sampling: str = 'bilinear', tiling: int = 3,
                        top_stretch: FloatValue = 0.0, bottom_stretch: FloatValue = 0.0,
                        bg_color: FloatVector = [0.0, 0.0, 0.0, 1.0]) -> th.Tensor:
    """Non-atomic node: Trapezoid Transform (Color and Grayscale)

    Args:
        img_in (tensor): Input image.
        sampling (str, optional): Set sampling quality ('bilinear' or 'nearest').
            Defaults to 'bilinear'.
        tiling (int, optional): Tiling mode (see 'transform_2d'). Defaults to 3.
        top_stretch (float, optional): Set the amount of stretch or squash at the top.
            Defaults to 0.0.
        bottom_stretch (float, optional): Set the amount of stretch or squash at the botton.
            Defaults to 0.0.
        bg_color (list, optional): Set solid background color in case tiling is turned off.
            Defaults to [0.0, 0.0, 0.0, 1.0].

    Returns:
        Tensor: Trapezoid transformed image.
    """
    # Convert parameters to tensors
    top_stretch = to_tensor(top_stretch)
    bottom_stretch = to_tensor(bottom_stretch)
    bg_color = th.atleast_1d(to_tensor(bg_color))

    # Create a uniform color image as background
    num_channels, num_row, num_col = img_in.shape[1], img_in.shape[2], img_in.shape[3]
    bg_color = resize_color(bg_color, num_channels).view(-1, 1, 1).expand_as(img_in)

    # Compute trapezoid sampling grid
    x_grid, y_grid = get_pos(num_row, num_col).unbind(dim=2)
    slope = (x_grid - 0.5) / th.lerp(1 - top_stretch, 1 - bottom_stretch, y_grid).clamp_min(1e-15)
    sample_grid = th.stack((slope + 0.5, y_grid), dim=-1).unsqueeze(0)

    # Sample the input image and compose onto the background
    img_out = grid_sample(img_in, sample_grid, mode=sampling, tiling=tiling, sbs_format=True)
    img_out = th.where((slope.abs() > 0.5) & (not tiling % 2), bg_color, img_out)

    return img_out


@input_check(1, channel_specs='c')
def color_to_mask(img_in: th.Tensor, flatten_alpha: bool = False, keying_type: str = 'rgb',
                  rgb: FloatVector = [0.0, 1.0, 0.0], mask_range: FloatValue = 0.0,
                  mask_softness: FloatValue = 0.0) -> th.Tensor:
    """Non-atomic node: Color to Mask

    Args:
        img_in (tensor): Input image (RGB(A) only).
        flatten_alpha (bool, optional): Whether the alpha should be flattened for the result.
            Defaults to False.
        keying_type (str, optional): Keying type for isolating the color 
            ('rgb' | 'chrominance' | 'luminance'). Defaults to 'rgb'.
        rgb (list, optional): Which color to base the mask on. Defaults to [0.0, 1.0, 0.0].
        mask_range (float, optional): Width of the range that should be selected. Defaults to 0.0.
        mask_softness (float, optional): How hard the contrast/falloff of the mask should be.
            Defaults to 0.0.

    Returns:
        Tensor: Color to mask image.
    """
    # Compute luminance of the input image and the source color
    in_lum = c2g(img_in, flatten_alpha = flatten_alpha,
                 rgba_weights = [0.299, 0.587, 0.114, 0.0], bg = 0.0)
    img_color = to_tensor(rgb).view(-1, 1, 1).expand(1, -1, *img_in.shape[2:])
    color_lum = c2g(img_color, rgba_weights = [0.299, 0.587, 0.114, 0.0], bg = 1.0)

    # Prepare source and query map
    ## RGB mode
    if keying_type == 'rgb':
        img_in, img_key = img_in, resize_image_color(img_color, img_in.shape[1]).expand_as(img_in)

    ## Chrominance mode
    elif keying_type == 'chrominance':

        # Compute chrominance from luminance
        invert_lum = resize_image_color(1 - in_lum, img_in.shape[1])
        img_in = blend(invert_lum, img_in, blending_mode='add_sub', opacity=0.5)

        invert_color_lum = resize_image_color(1 - color_lum, 3)
        img_key = blend(invert_color_lum, img_color, blending_mode='add_sub', opacity=0.5)
        img_key = resize_image_color(img_key, img_in.shape[1])

    ## Luminance mode
    else:
        img_in, img_key = in_lum, color_lum

    # Obtain the color mask from source and query maps
    mask_pos = blend(img_key, img_in, blending_mode='subtract')
    mask_neg = blend(img_in, img_key, blending_mode='subtract')
    img_mask = blend(mask_pos, mask_neg, blending_mode='max')
    if img_mask.shape[1] > 1:
        img_mask = img_mask[:,:3].sum(dim=1, keepdim=True).clamp(0.0, 1.0)

    # Adjust the contrast of the color mask
    mask_range = to_tensor(mask_range).clamp_min(0.0005) * 0.25
    img_out = levels(img_mask, in_low = (1 - mask_softness) * mask_range, in_high = mask_range,
                     out_low = 1.0, out_high = 0.0)

    return img_out


@input_check(1, channel_specs='c')
def c2g_advanced(img_in: th.Tensor, grayscale_type: str = 'desaturation'):
    """Non-atomic node: Grayscale Conversion Advanced

    Args:
        img_in (tensor): Input image (RGB(A) only).
        grayscale_type (str, optional): Grayscale conversion type.
            ('desaturation' | 'luma' | 'average' | 'max' | 'min'). Defaults to 'desaturation'.

    Raises:
        ValueError: Unknown grayscale blending mode.

    Returns:
        Tensor: Grayscale image.
    """
    # Desaturation and average mode
    if grayscale_type in ('desaturation', 'average'):
        img_in = hsl(img_in, saturation=0.0) if grayscale_type.startswith('d') else img_in
        img_out = c2g(img_in)

    # Luma mode
    elif grayscale_type == 'luma':
        img_out = c2g(img_in, rgba_weights = [0.299, 0.587, 0.114, 0.0])

    # Max mode
    elif grayscale_type == 'max':
        img_out = img_in[:,:3].max(dim=1, keepdim=True)[0]

    # Min mode
    elif grayscale_type == 'min':
        img_out = img_in[:,:3].min(dim=1, keepdim=True)[0]

    # Unknown mode
    else:
        raise ValueError(f'Unknown grayscale conversion mode: {grayscale_type}')

    return img_out


@input_check(3, channel_specs='ggg')
def height_blend(img_fg: th.Tensor, img_bg: th.Tensor, img_mask: Optional[th.Tensor] = None,
                 mode: str = 'balanced', position: FloatValue = 0.5, contrast: FloatValue = 0.9,
                 opacity: FloatValue = 1.0) -> Tuple[th.Tensor, th.Tensor]:
    """Non-atomic node: Height Blend

    Args:
        img_fg (tensor): Top height map (G only).
        img_bg (tensor): Bottom height map (G only).
        img_mask (tensor, optional): Blending mask. Defaults to None.
        mode (str, optional): Blending mode specification ('balanced' or 'bottom').
            Defaults to 'balanced'.
        position (float, optional): Offsets height maps so the blend level is moved along the
            height axis. Defaults to 0.5.
        contrast (float, optional): Adjusts the contrast of the blending, makes transitions
            sharper. Defaults to 0.9.
        opacity (float, optional): Blending opacity of the top height map. Defaults to 1.0.

    Returns:
        Tensor: Blended height map.
        Tensor: Blending mask used to generate the output height map.
    """
    # Convert parameters to tensors
    position = to_tensor(position)

    # Adjust foreground and background by 'position' (priority)
    if mode == 'balanced':
        fg_levels = img_fg * (position * 2).clamp_max(1.0)
        bg_levels = img_bg * ((1 - position).clamp_max(0.5) * 2)
    else:
        low, high = (position * 2 - 1).clamp_min(0.0), (position * 2).clamp_max(1.0)
        fg_levels = th.lerp(low, high, img_fg)
        bg_levels = img_bg

    # Calculate height blending mask
    if img_mask is not None:
        height_fg = fg_levels * img_mask
        height_bg = th.lerp(img_bg, bg_levels, img_mask)
    else:
        height_bg, height_fg = bg_levels, fg_levels

    height_mask = th.max(height_fg, height_bg) - height_fg
    height_mask = opacity * (1 - height_mask / (1 - contrast).clamp_min(1e-4)).clamp_min(0.0)

    # Get blended height map
    blended_height = th.lerp(height_bg, height_fg, height_mask)

    return blended_height, height_mask


@input_check(1, channel_specs='g')
def noise_upscale_1(img_in: th.Tensor, offset_1x: FloatValue = 0.0, offset_1y: FloatValue = 0.0,
                    offset_2x: th.Tensor = 0.0, offset_2y: th.Tensor = 0.0) -> th.Tensor:
    """Non-atomic node: Noise Upscale 1
    """
    # Assemble offset parameters
    offset1 = th.stack((to_tensor(offset_1x), to_tensor(offset_1y)))
    offset2 = th.stack((to_tensor(offset_2x), to_tensor(offset_2y)))

    # Construct blending mask
    H_log2 = int(math.log2(img_in.shape[2]))
    y_grid_np, x_grid_np = np.ogrid[1/32:1.0:1/16, 1/32:1.0:1/16]
    mask_np = np.abs(x_grid_np - 0.5) > np.abs(y_grid_np - 0.5) - 1/32
    img_mask = automatic_resize(to_tensor(mask_np).expand(1, 1, -1, -1), scale_log2=H_log2-4)

    # Blend upscaled noises with different offsets
    img_fg = transform_2d(img_in, matrix22=[2.0, 0.0, 0.0, 2.0], offset=offset1)
    img_bg = transform_2d(img_in, matrix22=[-2.0, 0.0, 0.0, -2.0], offset=offset2)
    img_out = blend(img_fg, img_bg, blend_mask=img_mask)

    return img_out


@input_check(1, channel_specs='g')
def noise_upscale_2(img_in: th.Tensor, offset_1x: FloatValue = 0.0, offset_1y: FloatValue = 0.0,
                    offset_2x: th.Tensor = 0.0, offset_2y: th.Tensor = 0.0) -> th.Tensor:
    """Non-atomic node: Noise Upscale 2
    """
    img_upscale = noise_upscale_1(
        img_in, offset_1x=offset_1x, offset_1y=offset_1y, offset_2x=offset_2x, offset_2y=offset_2y)
    img_out = th.lerp(img_in, img_in.max(img_upscale), 0.5)
    img_out = th.lerp(img_out, img_out.min(img_upscale), 0.56)

    return img_out


@input_check(4, channel_specs='cccg')
def color_match(img_in: th.Tensor, img_src_color: Optional[th.Tensor] = None,
                img_target_color: Optional[th.Tensor] = None, img_mask: Optional[th.Tensor] = None,
                src_color_mode: str = 'average', src_color: FloatVector = [0.5, 0.5, 0.5],
                target_color_mode: str = 'parameter', target_color: FloatVector = [0.5, 0.5, 0.5],
                color_variation: bool = False, hue_variation: FloatValue = 0.0,
                chroma_variation: FloatValue = 1.0, luma_variation: FloatValue = 1.0,
                use_mask: bool = True, mask_mode: str = 'parameter',
                mask_hue_range: FloatValue = 30.0, mask_chroma_range: FloatValue = 0.5,
                mask_luma_range: FloatValue = 0.5, mask_blur: FloatValue = 0.0,
                mask_smoothness: FloatValue = 0.0) -> Tuple[th.Tensor, Optional[th.Tensor]]:
    """Non-atomic node: Color Match
    """
    # Separate RGB and alpha channels for input images
    img_rgb, img_alpha = img_in.split((3, 1), dim=1) if img_in.shape[1] == 4 else (img_in, None)

    # Determine what source and target colors to use
    if src_color_mode == 'average':
        img_src_color = img_rgb.mean(dim=(2, 3), keepdim=True)
    elif src_color_mode == 'parameter':
        img_src_color = to_tensor(src_color)
    elif src_color_mode == 'input':
        img_src_color = img_src_color[:,:3] if img_src_color is not None else th.zeros(3)
    else:
        raise ValueError(f'Unknown source color mode: {src_color_mode}')

    if target_color_mode == 'parameter':
        img_target_color = to_tensor(target_color)
    elif target_color_mode == 'input':
        img_target_color \
            = img_target_color[:,:3] if img_target_color is not None else th.zeros(3)
    else:
        raise ValueError(f'Unknown target color mode: {target_color_mode}')

    def to_hcl(img: th.Tensor) -> th.Tensor:
        """Helper function for converting an RGB image (or vector) to HCL.
        """
        return color_from_rgb(img, mode='hcl', dim=1) if img.ndim == 4 \
               else color_from_rgb(img, mode='hcl').view(1, -1, 1, 1)

    # Convert RGB to HCL
    in_hcl, src_hcl = to_hcl(img_rgb), to_hcl(img_src_color)
    target_hcl = to_hcl(img_target_color)
    diff_hcl = in_hcl - src_hcl

    # Apply custom color variation
    var_weights = (hue_variation / 360, chroma_variation, luma_variation)
    var_weights = th.stack([to_tensor(t) for t in var_weights]) if color_variation \
                  else to_tensor([0.0, 1.0, 1.0])
    out_hcl = diff_hcl * var_weights.view(-1, 1, 1) + target_hcl
    out_hcl = th.cat([out_hcl[:,:1] % 1, out_hcl[:,1:].clamp_min(0.0)], dim=1)

    # Convert HCL back to RGB
    img_out = color_to_rgb(out_hcl, mode='hcl', dim=1)
    img_out = th.cat((img_out, img_alpha), dim=1) if img_alpha is not None else img_out

    # Calculate HCL mask
    if mask_mode == 'parameter':
        dh_abs, dc_abs, dl_abs = diff_hcl.abs().split(1, dim=1)
        h_mask = th.min(dh_abs, 1 - dh_abs) <= mask_hue_range * 0.5 / 360
        c_mask = dc_abs <= mask_chroma_range * 0.5
        l_mask = dl_abs <= mask_luma_range * 0.5
        hcl_mask = (h_mask & c_mask & l_mask).float()

        # Blur the mask
        img_mask = blur_hq(hcl_mask, high_quality=True, intensity=mask_smoothness)
        img_mask = histogram_scan(img_mask, position=0.5, contrast=0.95)
        img_mask = blur_hq(img_mask, high_quality=True, intensity=mask_blur)

    # Use input mask
    elif mask_mode == 'input':
        img_mask = th.zeros_like(img_in[:,:1]) if img_mask is None else img_mask

    # Use a blending mask for the result
    if use_mask:
        img_out = blend(img_out, img_in, blend_mask=img_mask)

    return img_out, img_mask


@input_check(1, channel_specs='g')
def shadows(img_in: th.Tensor, dist: FloatValue = 0.3, light_angle: FloatValue = 0.13,
            edge_softness: FloatValue = 1.0, samples: int = 8) -> th.Tensor:
    """Non-atomic node: Shadows
    """
    img_blur = non_uniform_blur(
        img_in, 1 - img_in, samples=samples, blades=4, intensity=dist*10, anisotropy=edge_softness,
        asymmetry=edge_softness, angle=light_angle+0.5)
    img_out = 1 - (img_blur - img_in).clamp(0.0, 1.0)

    return img_out


# ---------------------------------------------------------------------------- #
#          Mathematical functions used in the implementation of nodes.         #
# ---------------------------------------------------------------------------- #

def cross_2d(v1: th.Tensor, v2: th.Tensor, dim: int = -1) -> th.Tensor:
    """2D cross product function.

    Args:
        v1 (tensor): the first vector (or an array of vectors)
        v2 (tensor): the second vector

    Raises:
        TypeError: Input v1 or v2 is not a torch tensor.
        ValueError: Input v1 or v2 does not represent 2D vectors.

    Returns:
        Tensor: cross product(s) of v1 and v2.
    """
    # Check input validity
    if not isinstance(v1, th.Tensor) or not isinstance(v2, th.Tensor):
        raise TypeError("Input 'v1' and 'v2' must be torch tensors")
    if v1.shape[dim] != 2 or v2.shape[dim] != 2:
        raise ValueError("Input 'v1' and 'v2' should both be 2D vectors")

    # Compute cross product
    (v1x, v1y), (v2x, v2y) = v1.unbind(dim=dim), v2.unbind(dim=dim)
    ret = v1x * v2y - v1y * v2x

    return ret


def solve_poly_2d(a: th.Tensor, b: th.Tensor, c: th.Tensor) -> Tuple[th.Tensor, ...]:
    """Solve quadratic equations (ax^2 + bx + c = 0).

    Args:
        a (tensor): 2D array of value a's (M x N)
        b (tensor): 2D array of value b's (M x N)
        c (tensor): 2D array of value c's (M x N)

    Returns:
        Tensor: the first solutions of the equations
        Tensor: the second solutions of the equations
        Tensor: error flag when equations are invalid
    """
    # Check input validity
    if any(not isinstance(x, th.Tensor) for x in (a, b, c)):
        raise TypeError('All inputs must be torch tensors')

    # Compute discriminant
    delta = b * b - 4 * a * c
    error = delta < 0

    # Return solutions
    sqrt_delta = th.sqrt(delta + 1e-16)
    x_quad_1, x_quad_2 = (sqrt_delta - b) * 0.5 / a, (-sqrt_delta - b) * 0.5 / a
    x_linear = -c / b
    cond = (a == 0) | error
    x1 = th.where(cond, x_linear, x_quad_1)
    x2 = th.where(cond, x_linear, x_quad_2)

    return x1, x2, error


def color_from_rgb(rgb: FloatVector, mode: str = 'hsl', dim: int = -1) -> th.Tensor:
    """RGB to HSL/HCL.

    Args:
        rgb (list or tensor): RGB value.

    Returns:
        Tensor: HSL value.
    """
    check_arg_choice(mode, ['hsl', 'hcl'], arg_name='mode')
    rgb = to_tensor(rgb).movedim(dim, -1)

    # Compute chroma, lightness, and saturation
    dk_ = {'dim': -1, 'keepdim': True}
    (max_vals, max_inds), (min_vals, _) = rgb.max(**dk_), rgb.min(**dk_)
    c = max_vals - min_vals

    if mode == 'hsl':
        l = (max_vals + min_vals) * 0.5
        s = c / (1.0 - th.abs(2*l - 1.0) + 1e-8)
    else:
        l = (rgb * to_tensor([0.299, 0.587, 0.114])).sum(**dk_)
        s = c   # Use chroma for saturation

    # Compute hue
    seq = th.linspace(0, 4, 3)
    h_all = ((rgb.roll(-1, dims=-1) - rgb.roll(1, dims=-1)) / (c + 1e-8) + seq) % 6 / 6
    h = h_all.take_along_dim(max_inds, dim=-1)

    return th.cat([t.movedim(-1, dim) for t in (h, s, l)], dim=dim)


def color_to_rgb(src: FloatVector, mode: str = 'hsl', dim: int = -1) -> th.Tensor:
    """HSL/HCL to RGB.
    """
    check_arg_choice(mode, ['hsl', 'hcl'], arg_name='mode')
    h, s, l = to_tensor(src).movedim(dim, -1).split(1, dim=-1)

    # Pre-compute chroma weights
    w = (h - to_tensor([0.5, 1/3, 2/3])).abs() * to_tensor([6, -6, -6]) + to_tensor([-1, 2, 2])
    w = w.clamp(0.0, 1.0)

    # HSL to RGB
    if mode == 'hsl':
        c = (1 - (2 * l - 1).abs()) * s
        rgb = (l + c * (w - 0.5)).clamp(0.0, 1.0)

    # HCL to RGB
    else:
        temp_rgb = s * w
        temp_l = (temp_rgb * to_tensor([0.299, 0.587, 0.114])).sum(dim=-1, keepdim=True)
        rgb = (temp_rgb + (l - temp_l)).clamp(0.0, 1.0)

    return rgb.movedim(-1, dim).contiguous()


# ---------------------------------------------------------------------------------- #
#          Image manipulation functions used in the implementation of nodes.         #
# ---------------------------------------------------------------------------------- #

def get_pos(res_h: int, res_w: int) -> th.Tensor:
    """Get the $pos matrix of an input image (the center coordinates of each pixel).

    Args:
        res_h (int): Output image height.
        res_w (int): Output image width.

    Returns:
        Tensor: $pos matrix (size: (H, W, 2))
    """
    row_coords = th.linspace(0.5 / res_h, 1 - 0.5 / res_h, res_h)
    col_coords = th.linspace(0.5 / res_w, 1 - 0.5 / res_w, res_w)
    pos_grid = th.stack(th.meshgrid(col_coords, row_coords, indexing='xy'), dim=2)

    return pos_grid


def frequency_transform(img_in: th.Tensor, normal_format: str = 'dx') -> List[th.Tensor]:
    """Calculate convolution at multiple frequency levels.

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        normal_format (str, optional): Switch for inverting the vertical 1-D convolution
            direction ('dx' | 'gl'). Defaults to 'dx'.

    Returns:
        List[List[Tensor]]: List of convoluted images (in X and Y direction respectively).
    """
    in_size = img_in.shape[2]
    in_size_log2 = int(math.log2(in_size))

    # Create mipmap levels for R and G channels
    img_in = img_in[:,:2]
    mm_list: List[th.Tensor] = [img_in]
    if in_size_log2 > 4:
        mm_list.extend(create_mipmaps(img_in, in_size_log2 - 4))

    # Convolution operator
    def conv(img: th.Tensor) -> th.Tensor:
        dr = -1 if normal_format == 'dx' else 1
        img_x, img_y = img.split(1, dim=1)
        img_bw = img - th.cat((th.roll(img_x, 1, 3), th.roll(img_y, -dr, 2)), dim=1)
        img_fw = th.cat((th.roll(img_x, -1, 3), th.roll(img_y, dr, 2)), dim=1) - img
        return (img_bw.clamp(-0.5, 0.5) + img_fw.clamp(-0.5, 0.5)) * 0.5 + 0.5

    # Init blended images
    img_freqs: List[th.Tensor] = []

    # Low frequencies (for 16x16 images only)
    img_4 = mm_list[-1]
    img_4_scale: List[Optional[th.Tensor]] = [None, None, None, img_4]
    for i in reversed(range(3)):
        img_4_scale[i] = pad2d(manual_resize(img_4_scale[i + 1], -1), 4)

    for i, scale in enumerate([8.0, 4.0, 2.0, 1.0]):
        img_4_c = conv(img_4_scale[i])
        if scale > 1.0:
            img_4_c = transform_2d(
                img_4_c, mipmap_mode = 'manual', matrix22 = [1.0 / scale, 0.0, 0.0, 1 / scale])
        img_freqs.append(img_4_c)

    # Other frequencies
    for i in range(len(mm_list) - 1):
        img_i_c = conv(mm_list[-2 - i])
        img_freqs.append(img_i_c)

    return img_freqs
