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
            interp_pos = th.lerp(img_bg, th.sqrt(img_bg), img_fg * 2 - 1.0)
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
    anchors = anchors[th.argsort(anchors.select(1, 0))]

    # Compute anchor interval index
    img_in: th.Tensor = img_in.squeeze(1)
    anchor_idx = (img_in.unsqueeze(3) >= anchors.select(1, 0)).sum(3)
    pre_mask = anchor_idx == 0
    app_mask = anchor_idx == num_anchors

    # Make sure every position gets a valid anchor index
    anchor_idx = anchor_idx.sub_(1).clamp_(0, num_anchors - 2)
    anchor_idx_ravel = anchor_idx.ravel()
    B, H, W = anchor_idx.shape

    # Perform interpolation
    img_at_anchor = anchors[anchor_idx_ravel].reshape(B, H, W, num_col)
    img_at_next = anchors[anchor_idx_ravel + 1].reshape(B, H, W, num_col)
    img_at_anchor_pos = img_at_anchor.select(-1, 0)
    img_at_next_pos = img_at_next.select(-1, 0)

    a = (img_in - img_at_anchor_pos) / (img_at_next_pos - img_at_anchor_pos).clamp_min(1e-12)
    a = a if linear_interp else a ** 2 * (3 - a * 2)
    img_out = th.lerp(img_at_anchor.narrow(-1, 1, num_col - 1),
                      img_at_next.narrow(-1, 1, num_col - 1), a.unsqueeze(3))

    # Consider pixels that do not fall into any interpolation interval
    img_out[pre_mask, :] = anchors[0, 1:]
    img_out[app_mask, :] = anchors[-1, 1:]
    img_out = img_out.movedim(3, 1).clamp(0.0, 1.0)

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
    zero = th.zeros_like(img_in.narrow(1, 0, 1))
    s_mask = (l > 0.0) & (l < 1.0)
    s = th.where(s_mask, delta / (1.0 - th.abs(2 * l - 1) + 1e-8), zero)

    ## Compute hue
    h_vol = (img_in.roll(-1, 1) - img_in.roll(1, 1)) / (delta * 6.0).clamp_min(1e-8)
    h_vol = (h_vol + th.linspace(0, 2 / 3, 3).view(-1, 1, 1)) % 1.0
    h = th.where(delta > 0, h_vol.take_along_dim(max_idx, 1), zero)

    # Adjust HSL
    h = (h + hue * 2.0 - 1.0) % 1.0
    s = th.clamp(s + saturation * 2.0 - 1.0, 0.0, 1.0)
    l = th.clamp(l + lightness * 2.0 - 1.0, 0.0, 1.0)

    # Convert HSL back to RGB
    h_unnorm = h * 6.0
    c = (1.0 - th.abs(2.0 * l - 1.0)) * s
    x = c * (1.0 - th.abs(h_unnorm % 2.0 - 1.0))
    m = l - c * 0.5

    h_idx = (h_unnorm - th.linspace(0, 4, 3).view(-1, 1, 1)) % 6
    h_idx = th.where(h_idx >= 3, 6 - h_idx, h_idx)
    rgb = th.where(h_idx < 1, c, x)
    rgb = th.where(h_idx < 2, rgb, zero)
    rgb = rgb + m

    # Append original alpha channel
    img_out = rgb.clamp(0.0, 1.0)
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


@input_check(1)
def passthrough(img_in: th.Tensor) -> th.Tensor:
    """Helper node: Dot (pass-through)

    Args:
        img_in (tensor): Input image (G or RGB(A)).

    Returns:
        Tensor: The same image.
    """
    return img_in


# ---------------------------------------------
# Non-atomic functions
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
