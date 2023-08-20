from typing import Union, Optional, Tuple, List, Dict, Callable, Iterator
from functools import partial
import textwrap
import math

from torch.nn.functional import grid_sample as grid_sample_impl
import torch as th
import numpy as np
import numpy.typing as npt
import taichi as ti

from diffmat.core.base import BaseEvaluableObject
from diffmat.core.operator import resize_color, resize_image_color, create_mipmaps
from diffmat.core.types import ParamValue, FloatValue, FloatVector, IntVector, FXMapJobArray
from diffmat.core.util import to_numpy, to_const
from .patterns import ATOMIC_PATTERNS, ATOMIC_PATTERNS_BATCH, pattern_ti
from .util import get_pattern_image


class FXMapExecutor(BaseEvaluableObject):
    """A job scheduler and executor for FX-map nodes. At runtime, each quadrant node of the FX-map
    graph generates and submits a 'job' (adding some pattern) to the executor in traversal order.
    The executor then sorts and carries out these jobs, forming the output image.
    """
    # Column headers of an FX-map job array
    JOB_HEADERS = [
        'color', 'offset', 'size', 'rotation', 'variation', 'depth',
        'blending', 'filtering', 'image_index'
    ]

    # Columns that will be batched into torch tensors
    BATCH_JOB_HEADERS = ['color', 'offset', 'size', 'rotation', 'variation']

    # Image filtering modes
    FILTER_BILINEAR_MIPMAPS = 0
    FILTER_BILINEAR = 1
    FILTER_NEAREST = 2
    FILTER_DICT = {
        'bilinear_mipmap': FILTER_BILINEAR_MIPMAPS,
        'bilinear': FILTER_BILINEAR,
        'nearest': FILTER_NEAREST
    }

    # Blending modes
    BLEND_ADD = 0
    BLEND_MAX_COPY = 1
    BLEND_DICT = {
        'add': BLEND_ADD,
        'max': BLEND_MAX_COPY,
        'copy': BLEND_MAX_COPY
    }

    def __init__(self, res: int, keep_order: bool = True, **kwargs):
        """Initialize the FX-map executor.
        """
        super().__init__(**kwargs)

        self.res = res
        self.mode = 'gray'
        self.img_list: List[Optional[th.Tensor]] = []

        # The FX-map canvas where patterns are superimposed onto
        self.canvas = th.zeros(1, 1, 1 << res, 1 << res, device=self.device)

        # A one's buffer used for alpha blending
        self.ones = th.ones(1, 1, 1 << res, 1 << res, device=self.device)

        # Initialize the job array. The job array categorizes jobs based on pattern type. Jobs in
        # each category are tabularized under several column headers
        self.jobs: Dict[str, FXMapJobArray] = {}
        for type_ in [*ATOMIC_PATTERNS.keys(), 'image']:
            self.jobs[type_] = {key: [] for key in self.JOB_HEADERS}

        # Maintain pattern order when rendering patterns
        self.keep_order = keep_order

    def reset(self, img_bg: Optional[th.Tensor], *img_list: Optional[th.Tensor],
              mode: str = 'gray', background_color: Union[FloatValue, FloatVector] = 0.0,
              in_place: bool = True):
        """Reset the state of the FX-map executor. This will reset the canvas to a blank image and
        empty all job arrays.
        """
        if mode not in ('gray', 'color'):
            raise ValueError("Color mode must be either 'color' or 'gray'")

        self.mode = mode
        num_channels = 1 if mode == 'gray' else 4

        # Helper function that checks if an image belongs to the specified color mode
        def filter_by_mode(img: Optional[th.Tensor]) -> Optional[th.Tensor]:
            return resize_image_color(img, num_channels) \
                   if img is not None and (img.shape[1] == 1) == (num_channels == 1) else None

        # Only keep images that are compatible to the provided color mode
        self.img_list[:] = [filter_by_mode(img) for img in img_list]

        # Clear the gradient and fill the canvas with zero, then resize it to match the color mode
        canvas = self.canvas.detach_().fill_(0.0) if in_place else th.zeros_like(self.canvas)
        canvas = resize_image_color(canvas, num_channels).contiguous()

        # Apply the background color
        background_color = th.atleast_1d(self._t(background_color))
        canvas.copy_(resize_color(background_color, num_channels).view(-1, 1, 1))

        # Apply a background image (if any)
        if img_bg is not None:
            from diffmat.core.material.functional import blend
            self.canvas = canvas.copy_(img_bg) if mode == 'gray' else blend(img_bg, canvas)
        else:
            self.canvas = canvas

        # Clear the job array
        for job_arr in self.jobs.values():
            for column in job_arr.values():
                column.clear()

    def submit_job(self, **job_kwargs: ParamValue):
        """Submit a pattern generation job to the executor. The job is triaged and recorded into
        the job array according to the pattern type.
        """
        # Translate filtering and blending options to numbers
        for key, trans_dict in zip(('filtering', 'blending'), (self.FILTER_DICT, self.BLEND_DICT)):
            value = job_kwargs.get(key)
            if value is not None and value not in list(trans_dict.values()):
                job_kwargs[key] = trans_dict[value]

        # Get the job array associated with the pattern type
        job_type: str = job_kwargs['type']
        job_arr: FXMapJobArray = self.jobs[job_type]

        # Add the job into the job array if it doesn't point to a null image
        if job_type != 'image' or self.img_list[job_kwargs['image_index']] is not None:
            for header, column in job_arr.items():
                column.append(job_kwargs[header])

    def _batch_values(self, arrs: Union[List[FloatVector], List[FloatValue]],
                      colors: bool = False) -> th.Tensor:
        """Batch a list of floating-point vectors (potentially in different types) into a unified
        torch tensor. Color vectors are aligned to have the same length before batching.
        """
        # Align color vectors to the same length
        if colors:
            is_color = self.mode == 'color'
            num_channels = 4 if is_color else 1
            arrs_aligned: List[FloatVector] = []

            for arr in arrs:
                if isinstance(arr, th.Tensor) and arr.numel() != num_channels:
                    arrs_aligned.append(resize_color(arr, num_channels))
                elif isinstance(arr, list) and not is_color:
                    arrs_aligned.append(arr[0:1])
                elif isinstance(arr, float):
                    arrs_aligned.append((arr, arr, arr, 1.0) if is_color else (arr,))
                else:
                    arrs_aligned.append(arr)

            arrs = arrs_aligned

        # Find chunks of vectors that belong to the same type (they can be batched together)
        # `is_tensor` indicates whether each vector is a tensor (True) or a constant (False)
        is_tensor = np.array([isinstance(arr, th.Tensor) for arr in arrs])
        switch_indices = np.nonzero(np.append(np.diff(is_tensor), [True]))[0] + 1

        # The resulting list of tensors for final batching
        tensors: List[th.Tensor] = []

        # Conversion in chunks
        flag: bool
        i, flag = 0, is_tensor[0]

        for j in switch_indices:
            arr_slice = arrs[i:j]
            if flag:
                tensor = th.stack(arr_slice)
            else:
                tensor = th.tensor(np.array(arr_slice, dtype=np.float32), device=self.device)
            i, flag = j, ~flag

            # Promote each tensor to 2D
            tensor = tensor.unsqueeze(1) if tensor.ndim == 1 else tensor
            tensors.append(tensor)

        # Final batching
        batched_arr = th.cat(tensors)
        return batched_arr

    def _get_transform_data(self, offsets: th.Tensor, sizes: th.Tensor, rotations: th.Tensor,
                            bbox_format: str = 'numpy') -> Tuple[th.Tensor, IntVector, th.Tensor]:
        """Compute transform-related data that will be used in pattern generation, including
        inverse transformation matrices, bounding boxes, and bounding box positions relative to
        pattern centers.
        """
        img_size = 1 << self.res
        device = self.device

        # Precompute transformation and inverse transformation matrices between the patch space
        # and the image space
        cos_rot, sin_rot = th.cos(rotations), th.sin(rotations)
        R = th.cat((cos_rot, -sin_rot, sin_rot, cos_rot), dim=1).view(-1, 2, 2)
        M = R * sizes.unsqueeze(1)
        M_inv = R.transpose(1, 2) / sizes.unsqueeze(2)

        # Transform patch corners to the image space to determine the bounding boxes
        with th.no_grad():
            corners = th.tensor(
                [[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=th.float32, device=device)
            corners_img = corners @ M.transpose(1, 2) + offsets.unsqueeze(1)

        if bbox_format == 'numpy':
            corners_img_np = to_numpy(corners_img)
            bboxes_img = np.concatenate((np.min(corners_img_np, axis=1, keepdims=True),
                                         np.max(corners_img_np, axis=1, keepdims=True)), axis=1)
            bboxes = np.floor((bboxes_img + 1) * (0.5 * img_size) + 0.5).astype(np.int32)
        else:
            bboxes_img = th.cat(th.aminmax(corners_img, dim=1, keepdim=True), dim=1)
            bboxes = th.floor((bboxes_img + 1) * (0.5 * img_size) + 0.5).to(th.int32)

        # Compensate for pattern displacements
        bbox_lb = self._t(bboxes[:, 0] / img_size * 2 - 1)
        bbox_lb_relative = bbox_lb - offsets

        return M_inv, bboxes, bbox_lb_relative

    def _get_mipmap_stack(self, sizes: th.Tensor, mipmapping: bool = False) -> \
                              Tuple[Optional[th.Tensor], th.Tensor, th.Tensor]:
        """Calculate the mipmap stack of input images according to the requested pattern sizes and
        filtering modes.
        """
        img_list = self.img_list

        # Compute mipmap levels
        thresholds = th.tensor([2 ** (-0.5 - i) for i in range(12)], device=self.device)
        scales_min = sizes.detach().abs().min(dim=1)[0]
        mm_levels = (scales_min.unsqueeze(1) < thresholds).sum(dim=1).clamp_max(self.res)
        mm_levels[scales_min == 0] = 0
        max_mm_levels: int = mm_levels.max().item()

        # Stack all input images together (empty inputs are replaced by zeros)
        num_channels = 4 if self.mode == 'color' else 1
        zeros = th.zeros_like(self.ones).expand(1, num_channels, -1, -1)
        img_stack = th.cat([img if img is not None else zeros for img in img_list], dim=0)

        # Get the mipmap stack for all input images (to the maximum level required)
        if mipmapping:
            img_mm_stack = [img_stack, *create_mipmaps(img_stack, max_mm_levels)]

            # Assemble the mipmap stack into a mipmap volume
            vol_h, vol_w = img_stack.shape[-2], img_stack.shape[-1] * 2
            img_mm_volume = th.zeros(num_channels, img_stack.shape[0], vol_h, vol_w,
                                     device=self.device)

            for i, img_mm in enumerate(img_mm_stack):
                pos_h = int((1 - 0.25 ** (i // 2)) * vol_h)
                pos_w = int((1 - 0.25 ** ((i + 1) // 2)) * vol_w)
                mm_h, mm_w = img_mm.shape[-2], img_mm.shape[-1]
                img_mm_volume[..., pos_h:pos_h+mm_h, pos_w:pos_w+mm_w] = img_mm.transpose(0, 1)

            img_mm_volume = img_mm_volume.unsqueeze(0)

        # No mipmapping
        else:
            img_mm_volume = None

        return img_mm_volume, img_stack.transpose(0, 1).unsqueeze(0), mm_levels

    def _get_pattern_func(self, pattern_type: str, sizes: th.Tensor, variations: th.Tensor,
                          filter_modes: IntVector, image_indices: IntVector) -> \
                              Callable[[th.Tensor, int], th.Tensor]:
        """Create a pattern generation worker function based on the pattern type.
        """
        img_list = self.img_list
        is_color = self.mode == 'color'
        grid_sample = partial(grid_sample_impl, align_corners=False)

        if pattern_type == 'image':

            # Check if mipmapping is required
            modes_np = to_numpy(filter_modes)
            indices = self._t((to_numpy(image_indices) + 0.5) / len(img_list) * 2 - 1)

            # For image patterns, precompute the mipmap stacks for mipmap filtering
            mipmapping = any(modes_np == self.FILTER_BILINEAR_MIPMAPS)
            img_mm_volume, img_mm_stack, mm_levels = self._get_mipmap_stack(sizes, mipmapping)

            # Precompute scales and offsets for different mipmap levels on the mipmap volume
            if mipmapping:
                arr = sum(np.ogrid[:self.res+1, :2])[:, ::-1]
                vol_scales = self._t(0.5 ** arr)
                vol_offsets = self._t((1 - 0.25 ** (arr // 2) + 0.5 ** (arr + 1)) * 2 - 1)

            # Image pattern generation function
            def gen_pattern(grid: th.Tensor, i: int) -> th.Tensor:

                # Add image indices to the sampling grid
                length = grid.shape[0]
                index_coords = indices[i:i+length] if len(indices) > 1 else indices.expand(length)
                index_coords = index_coords.view(-1, 1, 1, 1).expand(*grid.shape[:-1], -1)
                vol_grid = th.cat((grid, index_coords), dim=-1).unsqueeze(0)

                # Gather results from all sampling modes
                modes_np_slice = modes_np[i:i+length] if len(modes_np) > 1 else modes_np
                img_res = th.zeros(length, img_mm_stack.shape[1], *grid.shape[1:3],
                                   device=self.device)

                for filtering in self.FILTER_DICT.values():

                    # Detect if the sampling mode is required
                    filtering_mask = modes_np_slice == filtering
                    if any(filtering_mask):

                        ## Bilinear sampling with mipmapping
                        if filtering == self.FILTER_BILINEAR_MIPMAPS:

                            # Scale and offset the volume sampling grid to reach mipmapping levels
                            vol_grid_scaled = vol_grid.clone()
                            mm_levels_slice = mm_levels[i:i+length]

                            vol_grid_coords = vol_grid_scaled[..., :2]
                            vol_grid_coords *= vol_scales[mm_levels_slice].view(-1, 1, 1, 2)
                            vol_grid_coords += vol_offsets[mm_levels_slice].view(-1, 1, 1, 2)

                            # Run bilinear image sampling
                            img_sample = grid_sample(img_mm_volume, vol_grid_scaled)

                        ## Bilinear sampling (no mipmapping)
                        elif filtering == self.FILTER_BILINEAR:
                            img_sample = grid_sample(img_mm_stack, vol_grid)

                        ## Nearest sampling
                        else:
                            img_sample = grid_sample(img_mm_stack, vol_grid, mode='nearest')

                        # Integrate sampling results into the final result
                        img_sample = img_sample.squeeze(0).transpose(0, 1)

                        if all(filtering_mask):
                            img_res[:] = img_sample
                        else:
                            mask = self._at(filtering_mask).view(-1, 1, 1, 1)
                            img_res = th.where(mask, img_sample, img_res)

                # Apply alpha mask to clear out-of-bound pixels
                alpha_mask = ((grid > -1) & (grid <= 1)).all(dim=-1, keepdim=True)
                img_res[:, -1:] *= alpha_mask.movedim(-1, -3)

                return img_res

        # Substitute pattern calculation by image sampling
        elif len(variations) == 1 and pattern_type not in ('square', 'disc'):

            # Generate the pattern image
            img_size = 1 << self.res
            img_pattern = get_pattern_image(
                pattern_type, img_size, img_size, mode=self.mode, var=variations.squeeze())

            # Atomic pattern generation function
            def gen_pattern(grid: th.Tensor, _: int) -> th.Tensor:
                return grid_sample(img_pattern.expand(grid.shape[0], -1, -1, -1), grid)

        else:
            pattern_func = ATOMIC_PATTERNS_BATCH[pattern_type]

            # Atomic pattern generation function
            def gen_pattern(grid: th.Tensor, i: int) -> th.Tensor:

                # Generate the pattern
                vars_ = variations[i:i+grid.shape[0]] if len(variations) > 1 else \
                        variations.expand(grid.shape[0])
                ret = pattern_func(grid, vars_.view(-1, 1, 1, 1))

                # Obtain the grayscale output which is also an alpha mask for the color output
                alpha = ret[0] if isinstance(ret, tuple) else ret
                if not is_color:
                    return alpha.movedim(-1, -3)

                # Construct the color output. The RGB channels are full-white inside the pattern.
                mask = ret[1] if isinstance(ret, tuple) else \
                    ((grid > -1) & (grid <= 1)).all(dim=-1, keepdim=True).float()
                output = th.cat((mask.expand(-1, -1, -1, 3), alpha), dim=-1)
                return output.movedim(-1, -3)

        return gen_pattern

    def _get_blend_func(self) -> \
            Callable[[th.Tensor, th.Tensor, Union[int, th.Tensor], th.Tensor], None]:
        """Create a blending function that deposits a pattern onto the canvas contingent on the
        color mode.
        """
        ones = self.ones

        # Color mode
        if self.mode == 'color':

            def _blend(img_fg: th.Tensor, img_bg: th.Tensor, mode: Union[int, th.Tensor],
                       opacity: th.Tensor):

                # Avoid confusing autograd
                img_bg_clone = img_bg.clone()

                # Available blending modes are 'add' and 'copy'
                img_fg, fg_alpha = img_fg.split(3, dim=1)

                ## Both 'add' and 'copy' are involved
                if isinstance(mode, th.Tensor):
                    img_fg_add = img_fg + img_bg_clone.narrow(1, 0, 3)
                    img_fg = th.lerp(img_fg_add, img_fg, mode)

                ## 'Add' only
                elif mode == self.BLEND_ADD:
                    img_fg = img_fg + img_bg_clone.narrow(1, 0, 3)

                # Apply blending opacity (`img_bg` must be a tensor view)
                img_fg = th.cat((img_fg, ones), dim=1)
                img_bg.copy_(th.lerp(img_bg_clone, img_fg, fg_alpha * opacity))

        # Grayscale mode
        else:

            def _blend(img_fg: th.Tensor, img_bg: th.Tensor, mode: Union[int, th.Tensor],
                       opacity: th.Tensor):

                # Available blending modes are 'add' and 'max'
                ## Both 'add' and 'copy' are involved
                if isinstance(mode, th.Tensor):
                    img_bg_clone = img_bg.clone()
                    img_fg = th.lerp(img_fg + img_bg_clone, th.max(img_fg, img_bg_clone), mode)
                    img_bg.lerp_(img_fg, opacity)

                ## 'Add' only
                elif mode == self.BLEND_ADD:
                    img_bg.add_(img_fg * opacity)

                ## 'Max' only
                else:
                    img_bg.lerp_(th.max(img_fg, img_bg.clone()), opacity)

        return _blend

    def _max_overlap_height(self, bboxes: th.Tensor) -> int:
        """Computes the maximum number of overlapped patterns after superposition.
        """
        img_size = 1 << self.res

        # Construct the sparse difference matrix
        diff = th.zeros(img_size + 2, img_size + 2, dtype=th.int32)
        fill_diff_matrix(diff, bboxes)

        # Calculate the number of overlapping patterns at each pixel from 2D prefix sum
        max_height = diff.cumsum_(1).cumsum_(0).max().item()

        return max_height

    def _draw_patterns(self, pattern_type: str, colors: th.Tensor, offsets: th.Tensor,
                       sizes: th.Tensor, rotations: th.Tensor, variations: th.Tensor,
                       blending_modes: IntVector, opacities: th.Tensor,
                       filter_modes: IntVector = [], image_indices: IntVector = []):
        """Generate patterns and draw patterns on the canvas. Serves as a worker function for the
        `_execute_batched_jobs` method.
        """
        canvas, device, img_size = self.canvas, self.device, 1 << self.res
        is_color = self.mode == 'color'

        # Part I - Bounding boxes
        # --------
        # Compute transformation-related data
        M_inv, bboxes, bbox_lb_relative = \
            self._get_transform_data(offsets, sizes, rotations, bbox_format='torch')

        # Part II - Pattern generation and Tetris sampling
        # --------
        # Create the pattern generator worker function
        gen_pattern = self._get_pattern_func(
            pattern_type, sizes, variations, filter_modes, image_indices)

        # Skip if the current batch is empty
        W, H = th.diff(bboxes, dim=1).squeeze(1).max(dim=0)[0].tolist()
        if min(W, H) <= 0:
            return

        # Pre-allocate sampling grid for patterns (the grid coordinates are relative to the
        # top-left corner in the image space)
        x_coords = th.linspace(1 / img_size, (W * 2 - 1) / img_size, W, device=device)
        y_coords = th.linspace(1 / img_size, (H * 2 - 1) / img_size, H, device=device)
        sample_grid_rel = th.stack(th.meshgrid(x_coords, y_coords, indexing='xy'), dim=2)

        # Get patterns from the worker function
        sample_grid = sample_grid_rel + bbox_lb_relative.unflatten(0, (-1, 1, 1))
        sample_grid = (sample_grid.unsqueeze(3) * M_inv.unflatten(0, (-1, 1, 1))).sum(dim=-1)
        patterns = gen_pattern(sample_grid, 0) * colors.unflatten(1, (-1, 1, 1))

        # Append opacity and blending mode info to construct the pattern volume
        opacities = opacities.view(-1, 1, 1, 1).expand_as(patterns.detach()[:, :1])
        cat_args = [patterns, opacities]

        if len(blending_modes) > 1:
            modes = self._t(blending_modes).view(-1, 1, 1, 1).expand_as(opacities)
            cat_args.append(modes)

        patterns_vol = th.cat(cat_args, dim=1)

        # Construct the Tetris sampling grid
        max_height = self._max_overlap_height(bboxes)
        if max_height <= 0:
            return

        tetris_grid = th.full((max_height, img_size, img_size, 3), 2.0, device=device)
        tetris_height = th.zeros(img_size, img_size, dtype=th.int32, device=device)
        tetris_placement(tetris_grid, tetris_height, bboxes, W, H)

        # In color mode, sort the sampling grid to maintain pattern order
        if self.keep_order:
            tetris_grid = tetris_grid.take_along_dim(tetris_grid[..., 2:].argsort(dim=0), dim=0)

        # Rearrange the generated patterns using Tetris sampling
        patterns_vol = patterns_vol.transpose(0, 1).unsqueeze(0)
        tetris_vol = grid_sample_impl(
            patterns_vol, tetris_grid.unsqueeze(0), mode='nearest', align_corners=False)
        tetris_vol = tetris_vol.squeeze(0).transpose(0, 1)

        # Part III - Pattern blending
        # --------
        # Helper function that composites a foreground pattern onto the background
        _blend = self._get_blend_func()

        # Blend individual layers of the Tetris volume to the canvas
        num_channels = 4 if is_color else 1
        vol_channels = tetris_vol.tensor_split([num_channels], dim=1)
        vol_channels: List[th.Tensor] = [vol_channels[0], *vol_channels[1].split(1, dim=1)]

        for tensors in zip(*[channel.split(1) for channel in vol_channels]):

            # Extract the layer image, blending opacities, and blending modes
            img_layer, img_opacity, *tensors = tensors
            img_mode = tensors[0] if len(tensors) else blending_modes[0]

            # Blend this layer into the background canvas
            _blend(img_layer, canvas, img_mode, img_opacity)

    def _execute_batched_jobs(self, batched_jobs: FXMapJobArray, pattern_type: str,
                              blending_opacity: FloatVector):
        """Execute a batch of jobs of the same type.
        """
        is_color = self.mode == 'color'

        # Extract columns from the batched job array
        colors: th.Tensor = batched_jobs['color']
        offsets: th.Tensor = batched_jobs['offset'] % 1 * 2 - 1
        sizes: th.Tensor = batched_jobs['size']
        rotations: th.Tensor = batched_jobs['rotation'] * (math.pi * 2)

        num_patterns = offsets.shape[0]

        # Extract other columns
        depths: IntVector = batched_jobs['depth']
        variations: th.Tensor = batched_jobs['variation']
        blending_modes: IntVector = batched_jobs['blending']
        filter_modes: Optional[IntVector] = batched_jobs.get('filtering')
        image_indices: Optional[IntVector] = batched_jobs.get('image_index')

        for key, column in batched_jobs.items():
            if len(column) not in (num_patterns, 1):
                raise ValueError(f"Mismatched column sizes: 'offset' ({num_patterns}) and "
                                 f"'{key}' ({len(column)})")

        # Filter mode and image index columns are mandatory when processing image patterns
        is_image_pattern = pattern_type == 'image'
        if is_image_pattern and (filter_modes is None or image_indices is None):
            raise ValueError('Columns related to image patterns must be provided')

        # Tailor the color input to match the color mode
        num_channels = 4 if is_color else 1
        if colors.shape[1] != num_channels:
            if is_color and colors.shape[1] == 1:
                th.cat((colors.expand(-1, 3), th.ones_like(colors)), dim=1)
            elif not is_color and colors.shape[1] > 1:
                colors = colors.narrow(1, 0, 1)
            else:
                raise RuntimeError(f'Can not resize color data from {colors.shape} into '
                                   f'{num_channels} columns')

        # Apply depth scaling to pattern sizes
        depths = np.asarray(batched_jobs['depth'])
        max_depth = len(blending_opacity) - 1
        if len(depths) > 1:
            scale = th.logspace(0, -max_depth, max_depth + 1, base=2.0)[depths].unsqueeze(1)
        else:
            scale = 2.0 ** -depths.item()
        sizes = sizes * scale

        # Compute blending opacity for each pattern
        if isinstance(blending_opacity, th.Tensor):
            opacities = blending_opacity[depths]
        else:
            opacities = self._t(np.asarray(blending_opacity)[depths])

        # Invoke the worker function after preparing all data
        self._draw_patterns(
            pattern_type, colors, offsets, sizes, rotations, variations, blending_modes, opacities,
            filter_modes=filter_modes if is_image_pattern else [],
            image_indices=image_indices if is_image_pattern else [])

    def evaluate(self, blending_opacity: FloatVector,
                 batched_jobs: Optional[Dict[str, FXMapJobArray]] = None,
                 clamp: bool = True) -> th.Tensor:
        """Run pattern generation jobs submitted from the FX-map graph.

        The job array to run from can be overrided using the `batched_jobs` argument, which will
        replace the default job array of the class.
        """
        # Print job summary for debugging
        self.log_job_summary(jobs=batched_jobs)

        # Choose which job array to use
        jobs = batched_jobs if batched_jobs is not None else self.jobs
        job_arr_generator = ((k, v) for k, v in jobs.items() if len(v['offset']))

        # Jobs are processed in the order of pattern type and submission
        for job_type, job_arr in job_arr_generator:

            # Create a batched job array by converting columns of the job array about pattern
            # transformation into torch tensors
            batch = job_arr.copy()
            for header in self.BATCH_JOB_HEADERS:
                column = job_arr[header]
                if not isinstance(column, th.Tensor):
                    if len(column) > 1:
                        batch[header] = self._batch_values(column, header == 'color')
                    elif len(column) == 1:
                        batch[header] = self._t(column[0]).view(1, -1).squeeze(1)
                    else:
                        raise RuntimeError(f"Column '{header}' of job array '{job_type}' is empty")

            # Execute the batched job array
            self._execute_batched_jobs(batch, job_type, blending_opacity)

        # Clean the internal job array if it has been evaluated
        if jobs is self.jobs:
            for job_type, job_arr in job_arr_generator:
                for columns in job_arr.values():
                    columns.clear()

        return self.canvas.clamp(0.0, 1.0) if clamp else self.canvas.clone()

    def to_device(self, device: Union[str, th.device]):
        """Move the FX-map executor to a specified device (i.e., CPU or GPU).

        Note that this step does not affect the content of the job array. Therefore, do not call
        this method when the job array is not empty.
        """
        # Move all internal tensors to the target device
        self.canvas = self.canvas.to(device)
        self.ones = self.ones.to(device)
        self.img_list[:] = [img.to(device) for img in self.img_list]

        super().to_device(device)

    def log_job_summary(self, jobs: Optional[Dict[str, FXMapJobArray]] = None):
        """Log the summary of a job array. The default internal job array is used if one is not
        provided in the arguments.
        """
        summary: List[str] = []

        # Count how many patterns are of each type
        num_patterns_total = 0
        for job_type, job_arr in (jobs if jobs is not None else self.jobs).items():
            num_patterns = len(job_arr['offset'])
            if num_patterns:
                num_patterns_total += num_patterns
                summary.append(f'{job_type}: {num_patterns}')

        summary.append(f'Total: {num_patterns_total} patterns')

        # Log the summary as a body of text with indentation
        summary_text = textwrap.indent('\n'.join(summary), '  ')
        self.logger.debug(f'FX-map job array summary:\n{summary_text}')


class DenseFXMapExecutor(FXMapExecutor):
    """A job scheduler and executor for FX-map nodes which is dedicated to very dense non-image
    patterns (e.g., the 7-th FX-map octave or higher).

    Compared with the regular FX-map executor, this variant is better optimized for performance
    but without auto-differentiation support due to an inordinate computational cost.
    """
    def __init__(self, res: int, **kwargs):
        """Initialize the dense FX-map executor.
        """
        super().__init__(res, **kwargs)

    def reset(self, img_bg: Optional[th.Tensor], *_: Optional[th.Tensor], mode: str = 'gray',
              **kwargs):
        """Reset the state of the FX-map executor. This will reset the canvas to a blank image and
        empty all job arrays.
        """
        # Dense FX-map engine only supports grayscale mode
        if mode != 'gray':
            raise ValueError('Dense FX-map engine only supports grayscale mode')

        # Discard input images
        super().reset(img_bg.detach() if isinstance(img_bg, th.Tensor) else img_bg,
                      mode=mode, **kwargs)

    def submit_job(self, **_: ParamValue):
        """Submit a pattern generation job to the executor. The job is triaged and recorded into
        the job array according to the pattern type.

        This method is forbidden to use for dense FX-map engines.
        """
        raise RuntimeError("The 'submit_job' method is now allowed for dense FX-map engines. "
                           "Please pass the batched job array to the 'evaluate' method directly.")

    def _draw_patterns(self, pattern_type: str, colors: th.Tensor, offsets: th.Tensor,
                       sizes: th.Tensor, rotations: th.Tensor, variations: th.Tensor,
                       blending_modes: IntVector, opacities: th.Tensor, **_):
        """Generate patterns and draw patterns on the canvas. Serves as a worker function for the
        `_execute_batched_jobs` method.
        """
        img_size = 1 << self.res
        canvas = self.canvas.view(img_size, img_size)

        # Dense FX-map engine only supports homogeneous blending modes
        if len(blending_modes) > 1:
            raise ValueError('Please specify a unified blending mode for all patterns')
        blending_mode: int = to_const(blending_modes)[0]

        # Translate pattern type from string to integer
        type_no = list(ATOMIC_PATTERNS.keys()).index(pattern_type)

        # Compute transformation-related data and pattern bounding boxes
        num_patterns = offsets.shape[0]
        bboxes = th.zeros(num_patterns, 2, 2, dtype=th.int32)
        M_inv = th.zeros(num_patterns, 2, 2)
        bbox_lb_relative = th.zeros(num_patterns, 2)

        W, H = compute_transform(
            M_inv, bboxes, bbox_lb_relative, offsets, sizes, rotations, img_size).to_list()

        # Draw all patterns in parallel
        # --------
        # Use Tetris placement if pattern rendering order is required
        if self.keep_order:
            raise NotImplementedError("The 'keep_order' flag is not supported for now")

        # Directly draw patterns onto the canvas
        else:
            draw_patterns_canvas(
                canvas.view(img_size, img_size), type_no, M_inv, bboxes, bbox_lb_relative, colors,
                variations, opacities, blending_mode, W, H)

    def _execute_batched_jobs(self, batched_jobs: FXMapJobArray, pattern_type: str,
                              blending_opacity: FloatVector):
        """Execute a batch of jobs of the same type.
        """
        # Image patterns are not supported
        if pattern_type == 'image':
            raise RuntimeError('Dense FX-map engines do not support image patterns')

        super()._execute_batched_jobs(batched_jobs, pattern_type, blending_opacity)

    def evaluate(self, blending_opacity: FloatVector,
                 batched_jobs: Optional[Dict[str, FXMapJobArray]] = None,
                 clamp: bool = True) -> th.Tensor:
        """Run pattern generation jobs submitted from the FX-map graph or provided using an
        external job array. This method wraps the same method of the base executor by disabling
        auto-differentiation.
        """
        # Run the executor without auto-differentiation
        with th.no_grad():
            return super().evaluate(blending_opacity, batched_jobs=batched_jobs, clamp=clamp)


def plot_bbox_ratios(bbox_ratios: npt.NDArray[np.float64]):
    """Plot bounding box ratios.
    """
    # Calculate total usage
    num_patterns = len(bbox_ratios)
    total_usage = bbox_ratios.prod(axis=1).sum() / (bbox_ratios.max(axis=0).prod() * num_patterns)

    # Plot ratios
    import matplotlib.pyplot as plt
    plt.scatter(*bbox_ratios.T, alpha=0.2)

    max_ratio = bbox_ratios.max()
    plt.xlim(left=0, right=max_ratio * 1.1)
    plt.ylim(bottom=0, top=max_ratio * 1.1)
    plt.title(f'Total usage: {total_usage:.6f}')
    plt.show()


# ----------------------------- #
#        Taichi routines        #
# ----------------------------- #

# Initialize the Taichi runtime
ti.init(arch=ti.gpu)

# Aliases for Taichi data types
ext_arr = ti.types.ndarray()
ivec2 = ti.types.vector(2, int)


@ti.kernel
def fill_diff_matrix(diff: ext_arr, bboxes: ext_arr):
    """Fill the forward difference matrix to calculate maximum Tetris height.
    """
    num_patterns, img_size = bboxes.shape[0], diff.shape[0] - 2

    # Process each pattern
    for pi in range(num_patterns):

        # Read the pattern bounding box
        bbox_lb_x, bbox_lb_y = bboxes[pi, 0, 0], bboxes[pi, 0, 1]
        bbox_ub_x, bbox_ub_y = bboxes[pi, 1, 0], bboxes[pi, 1, 1]

        # Determine positions of difference numbers
        xi, xj = bbox_lb_x % img_size + 1, (bbox_ub_x - 1) % img_size + 2
        yi, yj = bbox_lb_y % img_size + 1, (bbox_ub_y - 1) % img_size + 2

        # Calculate number of panels in X and Y directions
        panels_x = (bbox_ub_x - 1) // img_size - bbox_lb_x // img_size
        panels_y = (bbox_ub_y - 1) // img_size - bbox_lb_y // img_size

        # Fill difference numbers
        diff[0, 0] += panels_x * panels_y
        diff[0, xi] += panels_y
        diff[0, xj] -= panels_y
        diff[yi, 0] += panels_x
        diff[yj, 0] -= panels_x
        diff[yi, xi] += 1
        diff[yi, xj] -= 1
        diff[yj, xi] -= 1
        diff[yj, xj] += 1


@ti.kernel
def tetris_placement(grid: ext_arr, height: ext_arr, bboxes: ext_arr, bbox_w: int, bbox_h: int):
    """Tetris placement function.
    """
    # Read array dimensions
    num_patterns, img_size = bboxes.shape[0], height.shape[0]

    # Process all pixels in parallel
    for pi, ci, cj in ti.ndrange(num_patterns, bbox_h, bbox_w):

        # Read the pattern bounding box
        bbox_lb_x, bbox_lb_y = bboxes[pi, 0, 0], bboxes[pi, 0, 1]
        bbox_ub_x, bbox_ub_y = bboxes[pi, 1, 0], bboxes[pi, 1, 1]

        # Only process in-bound pixels
        if bbox_lb_x + cj < bbox_ub_x and bbox_lb_y + ci < bbox_ub_y:

            # Locate the pixel position in the grid
            gi, gj = (ci + bbox_lb_y) % img_size, (cj + bbox_lb_x) % img_size

            # Get the height at the grid pixel position and write sampling coordinates
            h = ti.atomic_add(height[gi, gj], 1)
            grid[h, gi, gj, 0] = (cj * 2 + 1) / bbox_w - 1
            grid[h, gi, gj, 1] = (ci * 2 + 1) / bbox_h - 1
            grid[h, gi, gj, 2] = (pi * 2 + 1) / num_patterns - 1


@ti.kernel
def compute_transform(M_inv: ext_arr, bboxes: ext_arr, bbox_lb_relative: ext_arr, offsets: ext_arr,
                      sizes: ext_arr, rotations: ext_arr, img_size: int) -> ivec2:
    """Transform data function.
    """
    num_patterns = offsets.shape[0]
    bbox_size = ti.Vector.zero(int, 2)

    for pi in range(num_patterns):

        # Compute transforms and inverse transforms between pattern space and image space
        rot = rotations[pi if rotations.shape[0] > 1 else 0, 0]
        cos_rot, sin_rot = ti.cos(rot), ti.sin(rot)
        size_x = sizes[pi if sizes.shape[0] > 1 else 0, 0]
        size_y = sizes[pi if sizes.shape[0] > 1 else 0, 1]

        M = ti.Matrix([[cos_rot * size_x, -sin_rot * size_y],
                       [sin_rot * size_x, cos_rot * size_y]])

        M_inv[pi, 0, 0], M_inv[pi, 0, 1] = cos_rot / size_x, sin_rot / size_x
        M_inv[pi, 1, 0], M_inv[pi, 1, 1] = -sin_rot / size_y, cos_rot / size_y

        # Compute bounding boxes in image coordinates
        corners = ti.Matrix([[-1, -1], [-1, 1], [1, -1], [1, 1]], dt=float)
        corners_img = corners @ M.transpose()
        bbox = ti.Matrix([[corners_img[0, 0], corners_img[0, 1]],
                          [corners_img[0, 0], corners_img[0, 1]]])

        for i, j in ti.static(ti.ndrange((1, 4), 2)):
            ti.atomic_min(bbox[0, j], corners_img[i, j])
            ti.atomic_max(bbox[1, j], corners_img[i, j])

        # Round bounding boxes to indices
        bbox_int = ti.Matrix.zero(int, 2, 2)
        offset = ti.Vector([offsets[pi, 0], offsets[pi, 1]])

        for i, j in ti.static(ti.ndrange(2, 2)):
            ind = int((bbox[i, j] + offset[j] + 1) * 0.5 * img_size + 0.5)
            bboxes[pi, i, j], bbox_int[i, j] = ind, ind

        # Compensate for pattern displacements
        for j in ti.static(range(2)):
            bbox_lb_relative[pi, j] = bbox_int[0, j] / img_size * 2 - 1 - offset[j]

        # Aggregate into maximum bounding box size
        for j in ti.static(range(2)):
            ti.atomic_max(bbox_size[j], bbox_int[1, j] - bbox_int[0, j])

    return bbox_size


@ti.kernel
def draw_patterns_canvas(canvas: ext_arr, pattern_type: int, M_inv: ext_arr, bboxes: ext_arr,
                         bbox_lb_relative: ext_arr, colors: ext_arr, variations: ext_arr,
                         opacities: ext_arr, blending_mode: int, bbox_w: int, bbox_h: int):
    """Generate atomic patterns and draw them onto the canvas.
    """
    num_patterns, img_size = bboxes.shape[0], canvas.shape[0]

    # Parallel over all pixels
    for pi, ci, cj in ti.ndrange(num_patterns, bbox_h, bbox_w):

        # Read the pattern bounding box and opacity
        bbox_lb_x, bbox_lb_y = bboxes[pi, 0, 0], bboxes[pi, 0, 1]
        bbox_ub_x, bbox_ub_y = bboxes[pi, 1, 0], bboxes[pi, 1, 1]

        # Only process in-bound pixels
        if bbox_lb_x + cj < bbox_ub_x and bbox_lb_y + ci < bbox_ub_y:

            # Locate the pixel position in the grid
            gi, gj = (ci + bbox_lb_y) % img_size, (cj + bbox_lb_x) % img_size

            # Calculate pixel coordinates in the pattern space
            coord_img = ti.Vector([
                bbox_lb_relative[pi, 0] + (cj * 2 + 1) / img_size,
                bbox_lb_relative[pi, 1] + (ci * 2 + 1) / img_size])

            M_inv_ = ti.Matrix([[M_inv[pi, 0, 0], M_inv[pi, 0, 1]],
                                [M_inv[pi, 1, 0], M_inv[pi, 1, 1]]], dt=float)
            coord = M_inv_ @ coord_img

            # Calculate pixel value
            var = variations[pi if variations.shape[0] > 1 else 0]
            col = colors[pi if colors.shape[0] > 1 else 0, 0]
            pval = pattern_ti(coord[0], coord[1], var, pattern_type) * col

            # Apply the current pixel to the canvas
            if blending_mode == FXMapExecutor.BLEND_ADD:
                opacity = opacities[pi if opacities.shape[0] > 1 else 0]
                ti.atomic_add(canvas[gi, gj], pval * opacity)
            else:
                ti.atomic_max(canvas[gi, gj], pval)
