from typing import Union, Optional, Tuple, List, Dict, Callable, Iterator
import itertools
import textwrap
import math

from torch.nn.functional import grid_sample as grid_sample_impl
import torch as th
import numpy as np
import numpy.typing as npt

from diffmat.core.base import BaseEvaluableObject
from diffmat.core.operator import resize_color, resize_image_color, create_mipmaps
from diffmat.core.types import ParamValue, FloatValue, FloatVector, IntVector, FXMapJobArray
from diffmat.core.util import to_numpy
from .patterns import ATOMIC_PATTERNS, ATOMIC_PATTERNS_BATCH


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
    BATCH_JOB_HEADERS = ['color', 'offset', 'size', 'rotation']

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

    def __init__(self, res: int, **kwargs):
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

    def reset(self, img_bg: Optional[th.Tensor], *img_list: Optional[th.Tensor],
              mode: str = 'gray', background_color: Union[FloatValue, FloatVector] = 0.0,
              in_place: bool = True):
        """Reset the state of the FX-map executor. This will reset the canvas to a blank image and
        empty all job arrays.
        """
        if mode not in ('gray', 'color'):
            raise ValueError("Color mode must be either 'color' or 'gray'")

        # Helper function that checks if an image belongs to a color mode
        def filter_by_mode(img: Optional[th.Tensor], mode: str) -> Optional[th.Tensor]:
            return img if img is None or (img.shape[1] == 1) == (mode == 'gray') else None

        # Only keep images that are compatible to the provided color mode
        self.mode = mode
        self.img_list[:] = [filter_by_mode(img, mode) for img in img_list]

        # Clear the gradient and fill the canvas with zero, then resize it to match the color mode
        num_channels = 1 if mode == 'gray' else 4
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

    def _get_transform_data(self, offsets: th.Tensor, sizes: th.Tensor, rotations: th.Tensor) -> \
            Tuple[th.Tensor, npt.NDArray[np.int32], th.Tensor]:
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
            corners_img = th.matmul(corners, M.transpose(1, 2)) + offsets.unsqueeze(1)

        corners_img_np = to_numpy(corners_img)
        bboxes_img = np.concatenate((np.min(corners_img_np, axis=1, keepdims=True),
                                     np.max(corners_img_np, axis=1, keepdims=True)), axis=1)
        bboxes = np.floor((bboxes_img + 1) * (0.5 * img_size) + 0.5).astype(np.int32)

        # Compensate for pattern displacements
        bbox_lb = self._t(bboxes[:, 0] / img_size * 2 - 1)
        bbox_lb_relative = bbox_lb - offsets

        return M_inv, bboxes, bbox_lb_relative

    def _get_mipmap_stack(self, sizes: th.Tensor, filter_modes: IntVector,
                          image_indices: IntVector) -> \
                              Tuple[List[List[th.Tensor]], npt.NDArray[np.int64]]:
        """Calculate the mipmap stack of input images according to the requested pattern sizes and
        filtering modes.
        """
        img_list = self.img_list
        num_patterns = sizes.shape[0]

        # Compute mipmap levels
        thresholds = th.tensor([2 ** (-0.5 - i) for i in range(12)], device=self.device)
        scales_min = sizes.detach().abs().amin(dim=1)
        mm_levels = (scales_min.unsqueeze(1) < thresholds).sum(dim=1)
        mm_levels[scales_min == 0] = 0
        mm_levels = to_numpy(mm_levels)

        # Get the mipmap stack for each input image (to the maximum level required)
        max_mm_levels: List[int] = [0 for _ in img_list]
        for i in range(num_patterns):
            filtering = filter_modes[i if len(filter_modes) > 1 else 0]
            level = mm_levels[i] if filtering == self.FILTER_BILINEAR_MIPMAPS else 0
            image_ind = image_indices[i if len(image_indices) > 1 else 0]
            max_mm_levels[image_ind] = max(max_mm_levels[image_ind], level)

        img_mm_stacks: List[List[th.Tensor]] = []
        for i, img in enumerate(img_list):
            if img is not None:
                img_mm_stacks.append([img] + create_mipmaps(img, max_mm_levels[i]))
            else:
                img_mm_stacks.append([])

        return img_mm_stacks, mm_levels

    def _get_pattern_func(self, pattern_type: str, sizes: th.Tensor, variations: List[FloatValue],
                          filter_modes: IntVector, image_indices: IntVector) -> \
                              Callable[[th.Tensor, int], th.Tensor]:
        """Create a pattern generation worker function based on the pattern type.
        """
        img_list = self.img_list
        is_color = self.mode == 'color'

        if pattern_type == 'image':

            # For image patterns, precompute the mipmap stacks for mipmap filtering
            img_mm_stacks, mm_levels = self._get_mipmap_stack(sizes, filter_modes, image_indices)

            # Image pattern generation function
            def gen_pattern(grid: th.Tensor, i: int) -> th.Tensor:

                # Get the source image from mipmap stack if mipmapping is selected
                filtering = filter_modes[i if len(filter_modes) > 1 else 0]
                image_ind = image_indices[i if len(image_indices) > 1 else 0]

                enable_mipmap = filtering == self.FILTER_BILINEAR_MIPMAPS
                img_in = img_mm_stacks[image_ind][mm_levels[i]] if enable_mipmap else \
                         img_list[image_ind]

                # Perform grid sampling
                mode = 'nearest' if filtering == self.FILTER_NEAREST else 'bilinear'
                return grid_sample_impl(img_in, grid, mode=mode, align_corners=False)

        else:
            pattern_func = ATOMIC_PATTERNS[pattern_type]

            # Atomic pattern generation function
            def gen_pattern(grid: th.Tensor, i: int) -> th.Tensor:

                # Generate the pattern
                variation = variations[i if len(variations) > 1 else 0]
                ret = pattern_func(grid, variation)

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

    def _pattern_generator(self, pattern_type: str, colors: th.Tensor, offsets: th.Tensor,
                           sizes: th.Tensor, rotations: th.Tensor, variations: List[FloatValue],
                           filter_modes: IntVector = [], image_indices: IntVector = []) -> \
                               Iterator[Tuple[Optional[th.Tensor], npt.NDArray[np.int32]]]:
        """An internal generator of 2D-transformed image or other atomic patterns. Yields each
        pattern and its bounding box.
        """
        is_image = pattern_type == 'image'
        img_list = self.img_list
        img_size = 1 << self.res
        device = self.device

        # Compute transformation-related data
        M_inv, bboxes, bbox_lb_relative = self._get_transform_data(offsets, sizes, rotations)
        M_inv_t = M_inv.transpose(1, 2)

        # Pre-allocate sampling grid for patterns (the grid coordinates are relative to the
        # top-left corner in the image space)
        bbox_sizes = np.diff(bboxes, axis=1).squeeze(1)
        bbox_max_cols, bbox_max_rows = bbox_sizes.max(axis=0).tolist()
        x_coords = th.linspace(
            1 / img_size, (bbox_max_cols * 2 - 1) / img_size, bbox_max_cols, device=device)
        y_coords = th.linspace(
            1 / img_size, (bbox_max_rows * 2 - 1) / img_size, bbox_max_rows, device=device)
        sample_grid_rel = th.stack(th.meshgrid(x_coords, y_coords, indexing='xy'), dim=2)

        # Create the pattern generator worker function
        gen_pattern = self._get_pattern_func(
            pattern_type, sizes, variations, filter_modes, image_indices)

        # Generating and compositing all FX-map patterns
        for i in range(offsets.shape[0]):
            bbox = bboxes[i]

            # Yield an empty pattern if the pattern points towards an empty image
            if is_image and img_list[image_indices[i if len(image_indices) > 1 else 0]] is None:
                yield None, bbox
                continue

            # Yield an empty pattern when the bounding box size is zero
            bbox_size_x, bbox_size_y = bbox_sizes[i].tolist()
            if min(bbox_size_x, bbox_size_y) == 0:
                yield None, bbox
                continue

            # Reverse transformation from image space to patch space
            sample_grid = sample_grid_rel[:bbox_size_y, :bbox_size_x]
            sample_grid = th.matmul((sample_grid + bbox_lb_relative[i]).unsqueeze(2), M_inv_t[i])
            sample_grid = sample_grid.movedim(2, 0)

            # Return the pattern and the bounding box
            pattern = gen_pattern(sample_grid, i) * colors[i].view(-1, 1, 1)
            yield pattern, bbox

    def _get_blend_func(self, enable_opacity: bool) -> \
            Callable[[th.Tensor, th.Tensor, int, FloatValue], None]:
        """Create a blending function that deposits a pattern onto the canvas contingent on the
        color mode.
        """
        ones = self.ones

        # Color mode
        if self.mode == 'color':

            def _blend(img_fg: th.Tensor, img_bg: th.Tensor, mode: int,
                       opacity: FloatValue = 1.0):

                # Avoid confusing autograd
                img_bg_clone = img_bg.clone()

                # Available blending modes are 'add' and 'copy'
                img_fg, fg_alpha = img_fg.split(3, dim=1)
                img_fg = img_fg + img_bg_clone.narrow(1, 0, 3) \
                         if mode == self.BLEND_ADD else img_fg
                img_fg = th.cat((img_fg, ones[..., :img_fg.shape[2], :img_fg.shape[3]]), dim=1)

                # Apply blending opacity (`img_bg` must be a tensor view)
                fg_alpha = fg_alpha * opacity if enable_opacity else fg_alpha
                img_bg.copy_(th.lerp(img_bg_clone, img_fg, fg_alpha))

        # Grayscale mode
        else:

            def _blend(img_fg: th.Tensor, img_bg: th.Tensor, mode: int,
                       opacity: FloatValue = 1.0):

                # Available blending modes are 'add' and 'max'
                if mode == self.BLEND_ADD:
                    if isinstance(opacity, float):
                        img_bg.add_(img_fg, alpha=opacity)
                    else:
                        img_bg += img_fg * opacity if enable_opacity else img_fg
                else:
                    # Avoid confusing autograd
                    img_bg_clone = img_bg.clone()

                    img_fg = th.max(img_fg, img_bg_clone)
                    img_out = th.lerp(img_bg_clone, img_fg, opacity) if enable_opacity else img_fg
                    img_bg.copy_(img_out)

        return _blend

    def _execute_batched_jobs(self, batched_jobs: FXMapJobArray, pattern_type: str,
                              blending_opacity: FloatVector):
        """Execute a batch of jobs of the same type.

        Please do not call this method directly. See the `evaluate` method for a public interface.
        """
        canvas = self.canvas
        is_color = self.mode == 'color'
        img_size = 1 << self.res

        # Extract columns from the batched job array
        colors: th.Tensor = batched_jobs['color']
        offsets: th.Tensor = batched_jobs['offset'] % 1 * 2 - 1
        sizes: th.Tensor = batched_jobs['size']
        rotations: th.Tensor = batched_jobs['rotation'] * (math.pi * 2)

        num_patterns = [column.shape[0] for column in (colors, offsets, sizes, rotations)]
        if min(num_patterns) != max(num_patterns):
            raise ValueError(f'Mismatched column sizes: {num_patterns} from '
                             f'(color, offset, size, rotation)')

        # Extract other columns
        variations: List[FloatValue] = batched_jobs['variation']
        blending_modes: IntVector = batched_jobs['blending']
        filter_modes: Optional[IntVector] = batched_jobs.get('filtering')
        image_indices: Optional[IntVector] = batched_jobs.get('image_index')

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
        # Opacity is disabled when it is a constant of 1
        enable_opacity = isinstance(blending_opacity, th.Tensor) or \
                         np.any(np.asarray(blending_opacity) != 1.0)
        opacities: FloatVector

        if not enable_opacity:
            opacities = np.ones(len(depths), dtype=np.float32)
        elif isinstance(blending_opacity, th.Tensor):
            opacities = blending_opacity[depths]
        elif isinstance(blending_opacity, (np.ndarray, list)):
            opacities = np.asarray(blending_opacity)[depths]

        # Helper function that composites a foreground pattern onto the background
        _blend = self._get_blend_func(enable_opacity)

        # Generating and compositing all FX-map patterns
        pattern_generator_args = [pattern_type, colors, offsets, sizes, rotations, variations]
        if is_image_pattern:
            pattern_generator_args.extend([filter_modes, image_indices])
        generator = self._pattern_generator(*pattern_generator_args)

        for i, (pattern, bbox) in enumerate(generator):

            # Skip empty patterns
            if pattern is None:
                continue

            # Get blending mode and opacity
            blending_mode = blending_modes[i if len(blending_modes) > 1 else 0]
            opacity = opacities[i if len(opacities) > 1 else 0]

            # Dissect the sampled pattern into panels and apply them to the canvas
            (bbox_lb_x, bbox_lb_y), (bbox_ub_x, bbox_ub_y) = bbox.tolist()
            panel_lb_x, panel_lb_y = bbox_lb_x // img_size, bbox_lb_y // img_size
            panel_ub_x = (bbox_ub_x + img_size - 1) // img_size
            panel_ub_y = (bbox_ub_y + img_size - 1) // img_size

            ## Only a single panel suffices
            if (panel_lb_x + 1, panel_lb_y + 1) == (panel_ub_x, panel_ub_y):

                # Get the corresponding canvas area
                canvas_xi, canvas_yi = bbox_lb_x % img_size, bbox_lb_y % img_size
                canvas_xj, canvas_yj = bbox_ub_x % img_size, bbox_ub_y % img_size
                canvas_xj = canvas_xj if canvas_xj else img_size
                canvas_yj = canvas_yj if canvas_yj else img_size
                canvas_panel = canvas[..., canvas_yi:canvas_yj, canvas_xi:canvas_xj]

                # Apply the pattern to the FX-Map canvas
                _blend(pattern, canvas_panel, blending_mode, opacity=opacity)
                continue

            ## Multiple panels are necessary (i.e., when the pattern exceeds the image border)
            panel_coords = itertools.product(
                range(panel_lb_y, panel_ub_y), range(panel_lb_x, panel_ub_x))

            for panel_y, panel_x in panel_coords:

                # Crop out the panel on the pattern
                panel_xi = max(panel_x * img_size, bbox_lb_x)
                panel_yi = max(panel_y * img_size, bbox_lb_y)
                panel_xj = min((panel_x + 1) * img_size, bbox_ub_x)
                panel_yj = min((panel_y + 1) * img_size, bbox_ub_y)
                pattern_xi, pattern_yi = panel_xi - bbox_lb_x, panel_yi - bbox_lb_y
                pattern_xj, pattern_yj = panel_xj - bbox_lb_x, panel_yj - bbox_lb_y
                pattern_panel = pattern[..., pattern_yi:pattern_yj, pattern_xi:pattern_xj]

                # Get the corresponding canvas area
                canvas_xi, canvas_yi = panel_xi % img_size, panel_yi % img_size
                canvas_xj, canvas_yj = panel_xj % img_size, panel_yj % img_size
                canvas_xj = canvas_xj if canvas_xj else img_size
                canvas_yj = canvas_yj if canvas_yj else img_size
                canvas_panel = canvas[..., canvas_yi:canvas_yj, canvas_xi:canvas_xj]

                # Apply the pattern to the FX-Map canvas
                _blend(pattern_panel, canvas_panel, blending_mode, opacity=opacity)

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
        job_arr_generator = ((k, v) for k, v in jobs.items() if len(v['color']))

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
                        batch[header] = self._t(column[0]).view(1, -1)
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
            num_patterns = len(job_arr['color'])
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
    # Columns in an FX-map job array that will be batched into torch tensors
    BATCH_JOB_HEADERS = ['color', 'offset', 'size', 'rotation', 'variation']

    def __init__(self, res: int, **kwargs):
        """Initialize the dense FX-map executor.
        """
        super().__init__(res, **kwargs)

    def reset(self, img_bg: Optional[th.Tensor], mode: str = 'gray',
              background_color: Union[FloatValue, FloatVector] = [0.0, 0.0, 0.0, 0.0],
              in_place: bool = True):
        """Reset the state of the FX-map executor. This will reset the canvas to a blank image and
        empty all job arrays.
        """
        with th.no_grad():
            super().reset(img_bg, mode=mode, background_color=background_color, in_place=in_place)

    def submit_job(self, **job_kwargs: ParamValue):
        """Submit a pattern generation job to the executor. The job is triaged and recorded into
        the job array according to the pattern type.
        """
        # Raise an error if the job refers to an image pattern
        if job_kwargs['type'] == 'image':
            raise ValueError('Image patterns are not meant for dense executors')
        else:
            super().submit_job(**job_kwargs)

    def _get_pattern_func(self, pattern_type: str, variations: th.Tensor) -> \
            Callable[[th.Tensor], th.Tensor]:
        """Create a batched pattern generation worker function based on the pattern type.
        """
        is_color = self.mode == 'color'
        pattern_func = ATOMIC_PATTERNS_BATCH[pattern_type]

        # Atomic pattern generation function
        def gen_pattern_batch(grid: th.Tensor) -> th.Tensor:

            # Generate the pattern
            ret = pattern_func(grid, variations.expand(grid.shape[0], 1).view(-1, 1, 1, 1))

            # Obtain the grayscale output which is also an alpha mask for the color output
            alpha = ret[0] if isinstance(ret, tuple) else ret
            if not is_color:
                return alpha.movedim(-1, -3)

            # Construct the color output. The RGB channels are full-white inside the pattern.
            mask = ret[1] if isinstance(ret, tuple) else \
                ((grid > -1) & (grid <= 1)).all(dim=-1, keepdim=True).float()
            output = th.cat((mask.expand(-1, -1, -1, 3), alpha), dim=-1)
            return output.movedim(-1, -3)

        return gen_pattern_batch

    def _pattern_generator(self, pattern_type: str, colors: th.Tensor, offsets: th.Tensor,
                           sizes: th.Tensor, rotations: th.Tensor,
                           variations: List[FloatValue]) -> Iterator[
                               Tuple[Optional[npt.NDArray[np.float32]], npt.NDArray[np.int32]]]:
        """An internal generator of 2D-transformed image or other atomic patterns. Yields each
        pattern and its bounding box.
        """
        img_size = 1 << self.res
        device = self.device

        # Compute transformation-related data
        M_inv, bboxes, bbox_lb_relative = self._get_transform_data(offsets, sizes, rotations)

        # Pre-allocate sampling grid for patterns (the grid coordinates are relative to the
        # top-left corner in the image space)
        bbox_sizes = np.diff(bboxes, axis=1).squeeze(1)
        bbox_max_cols, bbox_max_rows = bbox_sizes.max(axis=0).tolist()
        x_coords = th.linspace(
            1 / img_size, (bbox_max_cols * 2 - 1) / img_size, bbox_max_cols, device=device)
        y_coords = th.linspace(
            1 / img_size, (bbox_max_rows * 2 - 1) / img_size, bbox_max_rows, device=device)
        sample_grid_rel = th.stack(th.meshgrid(x_coords, y_coords, indexing='xy'), dim=2)

        # Create the pattern generator worker function
        gen_pattern = self._get_pattern_func(pattern_type, variations)

        # Reverse transformation from image space to patch space
        sample_grid = sample_grid_rel + bbox_lb_relative.view(-1, 1, 1, 2)
        sample_grid = (sample_grid.unsqueeze(3) * M_inv.view(-1, 1, 1, 2, 2)).sum(dim=-1)

        # Generating all FX-map patterns at once and iterate over the patterns
        patterns = gen_pattern(sample_grid) * colors.unsqueeze(2).unsqueeze(3)
        patterns_np: npt.NDArray[np.float32] = patterns.cpu().numpy()

        for i in range(patterns.shape[0]):
            bbox = bboxes[i]

            # Yield an empty pattern when the bounding box size is zero
            bbox_size_x, bbox_size_y = bbox_sizes[i].tolist()
            if min(bbox_size_x, bbox_size_y) == 0:
                yield None, bbox
            else:
                yield patterns_np[i,:,:bbox_size_y,:bbox_size_x], bbox

    def _get_blend_func(self, enable_opacity: bool) -> \
            Callable[[npt.NDArray[np.float32], npt.NDArray[np.float32], int, float], None]:
        """Create a blending function that deposits a pattern onto the canvas contingent on the
        color mode.
        """
        ones_np: npt.NDArray[np.float32] = to_numpy(self.ones)

        # Color mode
        if self.mode == 'color':

            def _blend(img_fg: npt.NDArray[np.float32], img_bg: npt.NDArray[np.float32],
                       mode: int, opacity: float = 1.0):

                # Available blending modes are 'add' and 'copy'
                img_fg, fg_alpha = np.split(img_fg, [3], axis=1)
                img_fg = img_fg + img_bg[:,:3] if mode == self.BLEND_ADD else img_fg
                img_fg = np.concatenate(
                    (img_fg, ones_np[..., :img_fg.shape[2], :img_fg.shape[3]]), axis=1)

                # Apply blending opacity (`img_bg` must be a tensor view)
                fg_alpha = fg_alpha * opacity if enable_opacity else fg_alpha
                img_bg += (img_fg - img_bg) * fg_alpha

        # Grayscale mode
        else:

            def _blend(img_fg: npt.NDArray[np.float32], img_bg: npt.NDArray[np.float32],
                       mode: int, opacity: float = 1.0):

                # Available blending modes are 'add' and 'max'
                if mode == self.BLEND_ADD:
                    img_bg += img_fg * opacity if enable_opacity else img_fg
                else:
                    img_fg = np.maximum(img_fg, img_bg)
                    if enable_opacity:
                        img_bg += (img_fg - img_bg) * opacity
                    else:
                        img_bg[:] = img_fg

        return _blend

    def _execute_batched_jobs(self, batched_jobs: FXMapJobArray, pattern_type: str,
                              blending_opacity: npt.NDArray[np.float32]):
        """Execute a batch of jobs of the same type. This method wraps the same method of the base
        executor by converting the canvas to a NumPy array back and forth.
        """
        # Convert the canvas to a NumPy array
        canvas: th.Tensor = self.canvas
        self.canvas: npt.NDArray[np.float32] = self.canvas.cpu().numpy()

        super()._execute_batched_jobs(batched_jobs, pattern_type, blending_opacity)

        # Convert the canvas back to torch tensor
        canvas.copy_(th.as_tensor(self.canvas, dtype=th.float32, device=self.device))
        self.canvas = canvas

    def evaluate(self, blending_opacity: npt.NDArray[np.float32],
                 batched_jobs: Optional[Dict[str, FXMapJobArray]] = None,
                 clamp: bool = True) -> th.Tensor:
        """Run pattern generation jobs submitted from the FX-map graph or provided using an
        external job array. This method wraps the same method of the base executor by disabling
        auto-differentiation.
        """
        # Remove any image-related jobs
        if batched_jobs is not None and 'image' in batched_jobs:
            del batched_jobs['image']

        # Run the executor without auto-differentiation
        with th.no_grad():
            return super().evaluate(blending_opacity, batched_jobs=batched_jobs, clamp=clamp)
