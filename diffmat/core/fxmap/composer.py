from typing import Tuple, List, Generator, Union, Optional
import math

import torch as th
import numpy as np
import numpy.typing as npt

from diffmat.core.base import BaseEvaluableObject
from diffmat.core.types import FloatValue, FloatVector, FXMapJobArray, JobArrayFunction
from diffmat.core.util import to_numpy, to_const
from .engine_v2 import FXMapExecutor, DenseFXMapExecutor
from .util import get_pattern_pos, get_group_pos, get_opacity


class ChainFXMapComposer(BaseEvaluableObject):
    """FX-map scheduler class for FX-map graphs shaped like a chain of quadrant nodes and with
    continuous scaling. Currently image patterns are not supported.
    """
    def __init__(self, res: int, pattern_type: str, mode: str = 'gray',
                 background_color: Union[FloatValue, FloatVector] = 0.0,
                 roughness: FloatValue = 0.0, global_opacity: FloatValue = 1.0, **kwargs):
        """Initialize the FX-map composer.
        """
        super().__init__(**kwargs)

        if pattern_type == 'image':
            raise NotImplementedError('Image patterns are not supported for now')

        self.res = res
        self.pattern_type = pattern_type
        self.background_color = background_color
        self.mode = mode
        self.roughness = roughness
        self.opacity = global_opacity

        # The maximum depth where patterns are rendered
        self.max_render_depth = res

        # The maximum depth where patterns are rendered with differentiability
        self.max_diff_render_depth = -1

        # Chunk size in the group dimension for creating the random number pool
        self.rand_chunk_size = 32

    def _rand_pool(self, depths: Tuple[int, int], rand_sizes: List[int], conns: List[int],
                   num_rendered_levels: int, num_groups: int) -> List[th.Tensor]:
        """Set up the random number pool according to graph structure, the actual number of levels
        rendered, and the number of pattern groups involved.
        """
        if num_rendered_levels <= 0:
            return

        # Check input validity
        depth_l, depth_r = depths
        if len(rand_sizes) != depth_r - depth_l:
            raise ValueError('The length of random number size does not match the depth interval')
        elif len(conns) != depth_r - depth_l - 1:
            raise ValueError('The length of connections does not match the depth interval')

        # Compute the number of patterns needed at each depth level
        num_conns = (np.array(conns) % 16 // np.expand_dims(2 ** np.arange(4), 1) % 2).sum(axis=0)
        num_patterns = np.cumprod(np.insert(num_conns, 0, 1)) * 2 ** (depth_l * 2)

        # Compute how many random numbers are needed at each depth level
        rand_sizes: npt.NDArray[np.int64] = num_patterns * np.array(rand_sizes)
        rand_size_total = rand_sizes.sum()

        # Generate the random number pool by chunks in the group dimension
        rand_pools: List[List[th.Tensor]] = [[] for _ in range(num_rendered_levels)]
        rand_chunk_size = self.rand_chunk_size
        num_chunks = (num_groups + rand_chunk_size - 1) // rand_chunk_size
        last_chunk_full = not num_groups % rand_chunk_size

        for chunk_id in range(num_chunks):

            # Generate random numbers of this chunk and split across all depth levels
            chunk_size = rand_chunk_size if chunk_id < num_chunks - 1 or last_chunk_full else \
                         num_groups % rand_chunk_size
            rands = th.rand(chunk_size, rand_size_total).split(rand_sizes.tolist(), dim=1)

            # Append the rand numbers to respective depth levels
            rand_depth: th.Tensor
            num: int

            for i, (rand_depth, num) in enumerate(zip(rands, num_patterns)):
                if i < num_rendered_levels:
                    rand_pools[i].append(rand_depth.unflatten(1, (num, -1)).contiguous())

        # Combine the random numbers at each depth level
        return [th.cat(column) for column in rand_pools]

    def _depth_iter(self, depths: Tuple[int, int], conns: List[int], scale: int = 1) -> \
                        Generator[Tuple[Optional[th.Tensor], ...], th.Tensor, None]:
        """A coroutine that iterates over all rendered depths and yields inputs to the job array
        function.
        """
        depth_l, num_depths = depths[0], depths[1] - depths[0]

        # Starting pattern positions and offsets (assuming fully-connected)
        pos = self._t((get_pattern_pos(depth_l) - 0.5) / scale + 0.5)
        pos_offsets_np = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=np.float32)
        pos_offsets = self._t(pos_offsets_np * 2 ** (-2 - depth_l) / scale)

        # Calculate initial branch offsets
        group_pos = self._t(get_group_pos(scale, relative=True).unsqueeze(1))
        branch_offsets = group_pos.expand(-1, 1 << (depth_l * 2), -1)

        for i in range(num_depths):

            # Provide input to the job array construction function and get the updated
            # pattern-wise branch offsets for the next octave
            branch_offsets = yield pos, branch_offsets

            # Advance to the next depth layer
            if i < num_depths - 1:
                conn_mask = [bool(conns[i] // val % 2) for val in (1, 2, 4, 8)]
                pos = (pos.unsqueeze(1) + pos_offsets[conn_mask]).view(-1, 2)
                pos_offsets *= 0.5
                branch_offsets = branch_offsets.repeat_interleave(sum(conn_mask), dim=1)

        yield None, None

    def evaluate(self, job_arr_func: JobArrayFunction, depths: Union[int, Tuple[int, int]],
                 rand_sizes: List[int], conns: Optional[List[int]] = None, scale: int = 1,
                 keep_order: bool = True) -> th.Tensor:
        """Follow the chain of FX-map in a certain depth range to compose an FX-map image.
        """
        depth_l, depth_r = (0, depths) if isinstance(depths, int) else depths
        conns = [0xf for _ in range(depth_l, depth_r - 1)] if conns is None else conns
        reset_kwargs = {'mode': self.mode, 'background_color': self.background_color}

        # Calculate depth level shift from the scale parameter and determine the depth range after
        # pattern scaling
        depth_shift = max(math.log2(scale), 0.0)
        depth_r = min(depth_r, math.ceil(self.max_render_depth + 1 - depth_shift))
        depth_diff_r = min(depth_r, math.floor(self.max_diff_render_depth + 1 - depth_shift))

        # Initiate the coroutine
        depth_iter = self._depth_iter((depth_l, depth_r), conns, scale=scale)
        pos, branch_offsets = next(depth_iter)

        # Pre-compute the random number pool
        rand_pool = self._rand_pool(depths, rand_sizes, conns, depth_r - depth_l, scale ** 2)
        rand_iter = iter(rand_pool)

        # Blending opacity
        blending_opacity = get_opacity(self.roughness, depth_r - 1) * self.opacity

        # Collect job arrays for differentiable rendering
        job_arrs: List[FXMapJobArray] = []

        for depth, rands in zip(range(depth_l, depth_diff_r), rand_iter):

            # Construct the job array using the provided function
            # I/O for the job array function:
            # Inputs:
            #   - depth: depth level (int)
            #   - pos: all active pattern positions at the current level (tensor)
            #   - branch_offsets: cumulative branch offsets (tensor)
            #   - rands: random numbers of the current level
            # Outputs:
            #   - job_arr: job array (dict)
            #   - branch_offsets: cumulative branch offsets (tensor)
            job_arr, branch_offsets = job_arr_func(depth, pos, branch_offsets, rands)
            pos, branch_offsets = depth_iter.send(branch_offsets)

            # Add the job array to the rendering queue
            if job_arr:
                job_arrs.append(job_arr)

        # Differentiable rendering
        executor = FXMapExecutor(self.res, keep_order=keep_order, device=self.device)
        executor.reset(img_bg=None, **reset_kwargs)
        fx_map: Optional[th.Tensor] = None

        if job_arrs:
            job_arr = (compose_job_arrays if keep_order else concat_job_arrays)(job_arrs)
            fx_map = executor.evaluate(
                blending_opacity, batched_jobs={self.pattern_type: job_arr}, clamp=False)

        # Create a dense FX-map engine if there are more depths to render
        if depth_r > depth_diff_r:
            executor = DenseFXMapExecutor(self.res, keep_order=False, device=self.device)
            executor.reset(img_bg=fx_map, **reset_kwargs)

        # Continue with fast and non-differentiable rendering
        fx_map_dense: Optional[th.Tensor] = None

        for depth, rands in zip(range(max(depth_diff_r, depth_l), depth_r), rand_iter):

            # Construct the job array using the provided function
            job_arr, branch_offsets = job_arr_func(depth, pos, branch_offsets, rands)
            pos, branch_offsets = depth_iter.send(branch_offsets)

            # Render the job array directly without queueing
            if job_arr:
                fx_map_dense = executor.evaluate(
                    blending_opacity, batched_jobs={self.pattern_type: job_arr}, clamp=False)

        # Collect result from the current group
        # The results on CPU is simply added to the GPU results by difference
        if fx_map is not None and fx_map_dense is not None:
            fx_map = fx_map + (fx_map_dense - fx_map.detach())

        # Final result
        fx_map = fx_map if fx_map is not None else \
                 fx_map_dense if fx_map_dense is not None else executor.canvas

        return fx_map.clamp(0.0, 1.0)


def compose_job_array(job_arr: FXMapJobArray, job_arr_next: FXMapJobArray) -> FXMapJobArray:
    """Compose two batched job arrays from consecutive FX-map depth levels.
    """
    # Check if two job arrays are composable
    if job_arr.keys() != job_arr_next.keys():
        raise ValueError('Job arrays with different column headers cannot be composed')

    # Base length is the job array size of the first job array
    base_length = len(job_arr['offset'])
    next_length = len(job_arr_next['offset'])
    ratio = next_length // base_length
    if next_length % base_length:
        raise ValueError('The size of job array 2 must be a muliplier of job array 1')

    # Initialize an empty result job array
    result: FXMapJobArray = {}

    # Compose each column
    for k, v in job_arr.items():
        v_next = job_arr_next[k]

        # Each column must have the same type
        if type(v) is not type(v_next):
            raise ValueError(f"Mismatched column types at header '{k}': got "
                             f"'{type(v).__name__}' and '{type(v_next).__name__}'")

        # The column length must match the base length
        if len(v) not in (1, base_length):
            raise ValueError(f"Invalid column size at header '{k}' in job array 1: {len(v)} "
                             f"(base length = {base_length})")
        if len(v_next) not in (1, next_length):
            raise ValueError(f"Invalid column size at header '{k}' in job array 2: {len(v_next)} "
                             f"(base length = {base_length})")

        # Special case - both representing the same constant
        if len(v) == 1 and len(v_next) == 1 and \
           to_const(v[0]) == to_const(v_next[0]):
            result[k] = v
            continue

        if isinstance(v, list):
            v_arr_1 = v if len(v) > 1 else v * base_length
            v_arr_2 = v_next if len(v_next) > 1 else v_next * next_length
            v_result = []
            for i, val in enumerate(v_arr_1):
                v_result.append(val)
                v_result.extend(v_arr_2[i * ratio : (i + 1) * ratio])

        elif isinstance(v, np.ndarray):
            v_arr_1 = v if v.shape[0] == base_length else v.repeat(base_length, axis=0)
            v_arr_2 = v_next if v_next.shape[0] > 1 else \
                      v_next.repeat(next_length, axis=0)
            v_result = np.insert(v_arr_2, np.arange(0, next_length, ratio), v_arr_1, axis=0)

        elif isinstance(v, th.Tensor):
            v_arr_1 = v if v.shape[0] == base_length else v.expand(base_length, *v.shape[1:])
            v_arr_2 = v_next if v_next.shape[0] > 1 else \
                      v_next.expand(next_length, *v_next.shape[1:])
            v_result = th.cat((v_arr_1.unsqueeze(1),
                               v_arr_2.unflatten(0, (base_length, ratio))), dim=1)
            v_result = v_result.view(base_length + next_length, *v_result.shape[2:])

        else:
            raise TypeError(f'Unknown column type: {type(v).__name__}')

        # Assign the result to the composed job array
        result[k] = v_result

    return result


def mask_job_array(job_arr: FXMapJobArray, mask: th.Tensor) -> FXMapJobArray:
    """Apply a boolean mask to a job array, which will only keep patterns masked by `True`.
    """
    mask_np = to_numpy(mask)
    mask_const = to_const(mask_np)
    mask_length = len(mask_const)
    mask_all_true = len(mask_const) == 1 and mask_const[0]

    # No operation needed if the mask is full
    if not mask_all_true:
        job_arr = job_arr.copy()

        # Process each column
        for k, v in job_arr.items():

            # Skip columns of uniform values
            if len(v) == 1 and mask_length > 1:
                continue

            if isinstance(v, list):
                v_arr = v if len(v) > 1 else v * mask_length
                v_result = [v_item for v_item, flag in zip(v_arr, mask_const) if flag]
            elif isinstance(v, np.ndarray):
                v_arr = v if v.shape[0] > 1 else v.repeat(mask_length, axis=0)
                v_result = v_arr[mask_np]
            elif isinstance(v, th.Tensor):
                v_arr = v if v.shape[0] > 1 else v.expand(mask_length, *v.shape[1:])
                v_result = v_arr[mask]
            else:
                raise TypeError(f'Unknown column type: {type(v).__name__}')

            # Assign the result to the masked job array
            job_arr[k] = v_result

    return job_arr


def compose_job_arrays(job_arrs: List[FXMapJobArray], apply_mask: bool = False) -> FXMapJobArray:
    """Compose multiple job arrays that represent jobs from several consecutive FX-map depth
    levels. If the job arrays contain a `mask` label, the function optionally applies that mask,
    which will only keep patterns masked by `True`.
    """
    if not job_arrs:
        raise ValueError('Input job array list is empty')

    # Compose job arrays in a bottom-up approach
    result = job_arrs[-1]
    for job_arr in reversed(job_arrs[:-1]):
        result = compose_job_array(job_arr, result)

    # Apply the pattern mask to remove discarded patterns
    if 'mask' in result:
        mask = result.pop('mask')
        result = mask_job_array(result, mask) if apply_mask else result

    return result


def concat_job_arrays(job_arrs: List[FXMapJobArray], apply_mask: bool = False) -> FXMapJobArray:
    """Concatenate multiple job arrays that represent jobs from several consecutive FX-map depth
    levels. If the job arrays contain a `mask` label, the function optionally applies that mask,
    which will only keep patterns masked by `True`.
    """
    if not job_arrs:
        raise ValueError('Input job array list is empty')

    # Collect column headers and job array sizes
    headers = set().union(*(job_arr.keys() for job_arr in job_arrs))
    sizes = [len(job_arr['offset']) for job_arr in job_arrs]

    # Construct the output job array by concatenating each column
    result: FXMapJobArray = {}

    for header in headers:

        # Extract the corresponding columns from all inputs
        columns = [job_arr.get(header) for job_arr in job_arrs]
        columns = [arr for arr in columns if arr is not None]

        # Check columns sizes
        for i, (col, size) in enumerate(zip(columns, sizes)):
            if len(col) != 1 and len(col) != size:
                raise ValueError(f"Invalid column size at header '{header}' in job array {i}: "
                                 f"{len(col)} (#patterns = {size})")

        # Special case - all representing the same constant
        if all(len(col) == 1 for col in columns):
            consts = [to_const(col[0]) for col in columns]
            if consts.count(consts[0]) == len(consts):
                result[header] = columns[0]
                continue

        # Concatenate the columns
        if any(isinstance(col, th.Tensor) for col in columns):
            columns_expand = [th.as_tensor(col).expand(size, *col.shape[1:])
                              for col, size in zip(columns, sizes)]
            result[header] = th.cat(columns_expand, dim=0)

        elif any(isinstance(col, np.ndarray) for col in columns):
            columns_expand = [np.asarray(col) for col in columns]
            columns_expand = [col.repeat(size, axis=0) if len(col) == 1 else col
                              for col, size in zip(columns_expand, sizes)]
            result[header] = np.concatenate(columns_expand, axis=0)

        else:
            result_column = []
            for col, size in zip(columns, sizes):
                result_column += col * size if len(col) == 1 else col
            result[header] = result_column

    # Apply the pattern mask to remove discarded patterns
    if 'mask' in result:
        mask = result.pop('mask')
        result = mask_job_array(result, mask) if apply_mask else result

    return result
