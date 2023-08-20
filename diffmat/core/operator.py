from typing import Union, List, Tuple

from torch.nn.functional import grid_sample as grid_sample_impl
import torch as th
import numpy as np

from diffmat.core.util import to_tensor, check_arg_choice


def resize_color(color: th.Tensor, num_channels: int, default_val: float = 1.0) -> th.Tensor:
    """Resize color to a specified number of channels.

    Args:
        color (tensor): Input color.
        num_channels (int): Target number of channels.
        default_val (float, optional): Default value for the alpha channel (if used).
            Defaults to 1.0.

    Raises:
        TypeError: Input color is not a 1-D tensor.
        ValueError: Input channel number is outside 1-4.
        RuntimeError: The input color cannot be resized to the target number of channels.

    Returns:
        Tensor: Resized color.
    """
    # Check input validity
    if not isinstance(color, th.Tensor) or color.ndim != 1:
        raise TypeError('Input color must be a 1-D tensor')
    if num_channels not in range(1, 5):
        raise ValueError(f'Number of channels must be 1-4, got {num_channels} instead')

    # Match background color with image channels
    C = len(color)
    if C > num_channels:
        color = color[:num_channels]
    elif C < num_channels:
        if C == 1 and num_channels > 1:
            color = color.expand(min(num_channels, 3))
        if len(color) == 3 and num_channels == 4:
            color = th.cat((color, th.atleast_1d(to_tensor(default_val))))

    # Report error if color adjustment failed
    if len(color) != num_channels:
        raise RuntimeError(
            f'Input color size [{C}] cannot be resized to {num_channels} channels')

    return color


def resize_image_color(img: th.Tensor, num_channels: int, default_val: float = 1.0) -> th.Tensor:
    """Resize the color channel of an image to a specified number.

    Args:
        color (tensor): Input image (G or RGB(A)).
        num_channels (int): Target number of channels.
        default_val (float, optional): Default value for the alpha channel (if used).
            Defaults to 1.0.

    Raises:
        TypeError: Input color is not a 4-D tensor.
        ValueError: Input channel number is outside 1-4.
        RuntimeError: The input color cannot be resized to the target number of channels.

    Returns:
        Tensor: Image with resized color.
    """
    # Check input validity
    if not isinstance(img, th.Tensor) or img.ndim != 4:
        raise TypeError('Input image must be a 4-D tensor')
    if num_channels not in range(1, 5):
        raise ValueError('Number of channels must be 1-4')

    # Match image with target channels
    C = img.shape[1]
    if C > num_channels:
        img = img.narrow(1, 0, num_channels)
    elif C < num_channels:
        if C == 1 and num_channels > 1:
            img = img.expand(-1, min(num_channels, 3), -1, -1)
        if img.shape[1] == 3 and num_channels == 4:
            img = th.cat((img, th.full_like(img.narrow(1, 0, 1), default_val)), dim=1)

    # Report error if color adjustment failed
    if img.shape[1] != num_channels:
        raise RuntimeError(
            f'Input image color size [{C}] cannot be resized to {num_channels} channels')

    return img


def resize_anchor_color(anchors: th.Tensor, num_channels: int,
                        default_val: float = 1.0) -> th.Tensor:
    """Resize the color channel of an anchor array to a specified number.

    Args:
        anchors (tensor): Input color anchors.
        num_channels (int): Target number of channels.
        default_val (float, optional): Default value for the alpha channel (if used).
            Defaults to 1.0.

    Raises:
        TypeError: Input anchors array is not a 2-D tensor.
        ValueError: Target channel number is outside 1-4.
        RuntimeError: Input anchors cannot be resized to the target number of channels.

    Returns:
        Tensor: Anchors with resized color.
    """
    # Check input validity
    if not isinstance(anchors, th.Tensor) or anchors.ndim != 2:
        raise TypeError('Input anchor array must be a 2-D tensor')
    if num_channels not in range(1, 5):
        raise ValueError('Number of channels must be 1-4')

    # Match anchor array with target channels
    C = anchors.shape[1]

    if C > num_channels + 1:
        if C == 5 and num_channels < 4:
            anchors = anchors[:,:4]
        if anchors.shape[1] == 4 and num_channels == 1:
            anchors = th.hstack((anchors[:,:1], anchors[:,1:].sum(dim=1, keepdim=True) * 0.33))
    elif C < num_channels + 1:
        if C == 2 and num_channels == 3:
            anchors = th.hstack((anchors[:,:1], anchors[:,1:].expand(-1, 3)))
        if anchors.shape[1] == 4 and num_channels == 4:
            anchors = th.hstack((anchors, th.full_like(anchors[:,:1], default_val)))

    # Report error if color channel adjustment failed
    if anchors.shape[1] != num_channels + 1:
        raise RuntimeError(
            f'Input anchor color size [{C - 1}] cannot be resized to {num_channels} channels')

    return anchors


def pad2d(img_in: th.Tensor, n: Union[int, Tuple[int, int]]) -> th.Tensor:
    """Perform circular padding on the last two dimensions of an input image. The padding width (n)
    must not exceed the size of the last two dimensions.

    Prefer using this function over `th.nn.functional.pad` for circular padding until the latter is
    provided with a C++ backend.

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        n (int or tuple of two ints): Padding size in the last two dimensions.

    Raises:
        ValueError: Input image is not a 4D tensor.
        ValueError: Padding width (n) exceeds the size of the last two dimensions.
        ValueError: Padding width (n) is negative.

    Returns:
        Tensor: Padded image.
    """
    # Check input validity
    if img_in.ndim != 4:
        raise ValueError('The input image must be a 4D tensor')

    H, W = img_in.shape[2], img_in.shape[3]
    n_h, n_w = (n, n) if isinstance(n, int) else n

    if n_h > H or n_w > W:
        raise ValueError('Padding width should not exceed the size of the last two dimensions')
    elif n_h < 0 or n_w < 0:
        raise ValueError('Padding width must be non-negative')

    # Perform padding
    img_pad = img_in if n_h == 0 else \
              th.cat((img_in.narrow(2, H - n_h, n_h), img_in, img_in.narrow(2, 0, n_h)), dim=2)
    img_pad = img_pad if n_w == 0 else \
              th.cat((img_pad.narrow(3, W - n_w, n_w), img_pad, img_pad.narrow(3, 0, n_w)), dim=3)

    return img_pad


def grid_sample(img_in: th.Tensor, img_grid: th.Tensor, mode: str = 'bilinear',
                tiling: int = 3, sbs_format: bool = False) -> th.Tensor:
    """Sample an input image by a matrix of grid coordinates with tiling preservation.

    The range of coordinates within an image is [-1, 1] by default. If `sbs_format` is True, the
    range is assumed to be [0, 1] per Substance's convention instead.

    `tiling` (default=3): 0 - no tiling; 1 - horizontal; 2 - vertical; 3 - both.

    Args:
        img_in (tensor): Input image (4D, BxCxHxW) (G or RGB(A)).
        img_grid (tensor): Sampling grid (4D, BxHxWx2).
        mode (str, optional): Sampling mode ('bilinear' or 'nearest'). Defaults to 'bilinear'.
        tiling (int, optional): Tiling mode. Defaults to 3.
        sbs_format (bool, optional): If True, the sampling grid is wrapped within [0, 1];
            otherwise, the sampling grid is wrapped within [-1, 1]. Defaults to False.

    Raises:
        ValueError: Input image or sampling grid is not a 4D tensor.
        ValueError: Tiling mode not in range(4).

    Returns:
        Tensor: Sampled image.
    """
    # Check input validity
    if img_in.ndim != 4 or img_grid.ndim != 4:
        raise ValueError('The input image and the coordinate grid must be 4D tensors')
    if tiling not in (0, 1, 2, 3):
        raise ValueError('The tiling mode must be an integer of 0 to 3.')

    # Pad the input image according to tiling mode
    img_pad = pad2d(img_in, [tiling >> 1, tiling & 1])

    # Generate the sampling grid that preserves tiling
    H, W = img_in.shape[2], img_in.shape[3]
    scales = (W / (W + 2), H / (H + 2))
    scales_all = scales[0] if H == W else to_tensor(scales)

    # Wrap and scale sampling coordinates to account for tiling modes
    if tiling == 3:
        img_grid = img_grid % 1 * 2 - 1 if sbs_format else (img_grid + 1) % 2 - 1
        img_grid = img_grid * scales_all

    elif tiling:
        ind = tiling - 1
        grid_slice = img_grid[..., ind]
        grid_slice = grid_slice % 1 * 2 - 1 if sbs_format else (grid_slice + 1) % 2 - 1
        img_grid[..., ind] = grid_slice * scales[ind]

    # Perform sampling
    img_out = grid_sample_impl(img_pad, img_grid, mode=mode, align_corners=False)

    return img_out


def automatic_resize(img_in: th.Tensor, scale_log2: int, filtering: str = 'bilinear') -> th.Tensor:
    """Progressively resize an input image.

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        scale_log2 (int): Size change relative to the input resolution (after log2).
        filtering (str, optional): 'bilinear' or 'nearest'. Defaults to 'bilinear'.

    Returns:
        Tensor: Resized image.
    """
    # Check input validity
    check_arg_choice(filtering, ['bilinear', 'nearest'], arg_name='filtering')

    # Get input and output sizes (after log2)
    in_size_log2 = int(np.log2(img_in.shape[2]))
    out_size_log2 = max(in_size_log2 + scale_log2, 0)

    # Equal size
    if out_size_log2 == in_size_log2:
        img_out = img_in

    # Down-sampling (regardless of filtering)
    elif out_size_log2 < in_size_log2:
        img_out = img_in
        for _ in range(in_size_log2 - out_size_log2):
            img_out = manual_resize(img_out, -1)

    # Up-sampling (progressive bilinear filtering)
    elif filtering == 'bilinear':
        img_out = img_in
        for _ in range(scale_log2):
            img_out = manual_resize(img_out, 1)

    # Up-sampling (nearest sampling)
    else:
        img_out = manual_resize(img_in, scale_log2, filtering)

    return img_out


def manual_resize(img_in: th.Tensor, scale_log2: int, filtering: str = 'bilinear') -> th.Tensor:
    """Manually resize an input image (all-in-one sampling).

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        scale_log2 (int): Size change relative to input (after log2).
        filtering (str, optional): 'bilinear' or 'nearest'. Defaults to 'bilinear'.

    Returns:
        Tensor: Resized image.
    """
    # Check input validity
    check_arg_choice(filtering, ['bilinear', 'nearest'], arg_name='filtering')

    # Get input and output sizes
    in_size = img_in.shape[2]
    in_size_log2 = int(np.log2(in_size))
    out_size_log2 = max(in_size_log2 + scale_log2, 0)
    out_size = 1 << out_size_log2

    # No need for scaling if both sizes are equal
    if out_size_log2 == in_size_log2:
        img_out = img_in

    else:
        # Compute the sampling grid
        x_coords = th.linspace(1 / out_size - 1, 1 - 1 / out_size, out_size)
        x, y = th.meshgrid(x_coords, x_coords, indexing='xy')
        sample_grid = th.stack([x, y], dim=2).expand(img_in.shape[0], -1, -1, -1)

        # Down-sample or up-sample the input image
        if out_size_log2 < in_size_log2:
            img_out = grid_sample_impl(img_in, sample_grid, mode=filtering, align_corners=False)
        else:
            img_out = grid_sample(img_in, sample_grid, mode=filtering)

    return img_out


def create_mipmaps(img_in: th.Tensor, mipmaps_level: int,
                   keep_size: bool = False) -> List[th.Tensor]:
    """Create mipmap levels for an input image using box filtering.

    Args:
        img_in (tensor): Input image (G or RGB(A)).
        mipmaps_level (int): Number of mipmap levels.
        keep_size (bool, optional): Switch for restoring the original image size after
            downsampling. Defaults to False.

    Returns:
        List[tensor]: Mipmap stack of the input image.
    """
    mipmaps = []
    img_mm = img_in
    last_shape = img_in.shape[2]

    # Successively downsample the input image
    for i in range(mipmaps_level):
        img_mm = manual_resize(img_mm, -1) if img_mm.shape[2] > 1 else img_mm
        mm_shape = img_mm.shape[2]

        # Recover the original image size
        img_mm_entry = img_mm if not keep_size else \
                       mipmaps[-1] if last_shape == 1 else \
                       img_mm.expand_as(img_in) if last_shape == 2 else \
                       automatic_resize(img_mm, i + 1)

        mipmaps.append(img_mm_entry)
        last_shape = mm_shape

    return mipmaps


__all__ = ['resize_color', 'resize_image_color', 'resize_anchor_color', 'pad2d', 'grid_sample',
           'automatic_resize', 'manual_resize', 'create_mipmaps']
