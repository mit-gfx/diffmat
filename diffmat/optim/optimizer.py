from pathlib import Path
from typing import Tuple, Dict, Optional, Any
import math

import torch as th
import pandas as pd

from diffmat.core.base import BaseEvaluableObject
from diffmat.core.material import MaterialGraph, Renderer
from diffmat.core.material.util import input_check
from diffmat.core.io import write_image, save_output_dict_to_sbs
from diffmat.core.types import PathLike
from diffmat.core.util import check_arg_choice, FILTER_OFF, FILTER_YES, FILTER_NO
from diffmat.optim.backend import get_backend
from diffmat.optim.metric import get_metric


class BaseOptimizer(BaseEvaluableObject):
    """Base class for control parameter optimizer for differentiable procedural material graphs.
    """
    def __init__(self, graph: MaterialGraph, param_type: str = 'continuous',
                 algorithm: str = 'adam', param_io_kwargs: Dict[str, Any] = {},
                 backend_kwargs: Dict[str, Any] = {}, metric: str = 'vgg', loss_type: str = 'l1',
                 metric_kwargs: Dict[str, Any] = {}, **kwargs):
        """Initialize the parameter optimizer.

        Args:
            graph (MaterialGraph): Source procedural material graph.
            param_type (str, optional): Optimizable parameter type ('continuous', 'integer', or
                'hybrid'). Defaults to 'continuous'.
            algorithm (str, optional): Node parameter optimization algorithm. Note that each
                parameter type supports different algorithms. Please refer to the `BACKEND_DICT`
                dictionary in `backend.py` for more details. Defaults to 'adam'.
            param_io_kwargs (Dict[str, Any], optional): Keyword arguments to pass into parameter
                value reading and writing methods. Defaults to {}.
            backend_kwargs (Dict[str, Any], optional): Keyword arguments to pass into the node
                parameter optimization algorithm. Defaults to {}.
            metric (str, optional): Optimization objective type ('vgg' or 'fft'). Defaults to 'vgg'.
            loss_type (str, optional): Loss function type ('l1' or 'l2'). Defaults to 'l1'.
            metric_kwargs (Dict[str, Any], optional): Keyword arguments to pass into the texture
                descriptor. Defaults to {}.
            kwargs (Dict[str, Any], optional): Keyword arguments to pass into the base class
                constructor.
        """
        check_arg_choice(param_type, ['continuous', 'integer', 'hybrid'], 'param_type')

        super().__init__(device=graph.device, **kwargs)

        self.graph = graph

        # Create a loss function
        self.loss_func = get_metric(
            metric, loss_type=loss_type, device=self.device, **metric_kwargs)

        # Initialize the backend algorithm
        self.backend = get_backend(
            param_type, algorithm, graph, self.loss_func, param_io_kwargs=param_io_kwargs,
            parent=self, **backend_kwargs)
        self.loss_func.set_parent(self.backend)

        # Show graph info
        self.logger.info(f'Graph name: {graph.name}  #nodes: {len(graph.nodes)}  '
                         f'#params: {self.backend._num_parameters()}')

        # Image headers in the output dict
        self.image_headers = ['render', *Renderer.CHANNELS.keys()]

    def reset(self):
        """Reset the optimizer to its initial state, including graph parameters and the running
        loss minimum.
        """
        # Reset the algorithm backend
        self.backend.reset()

    def _load_state(self, save_file: PathLike) -> int:
        """Load the state of the optimizer (including graph parameters and the Adam optimizer state
        from a local save file. Returns the iteration number.

        Args:
            save_file (PathLike): Path to the checkpoint file to load from.

        Returns:
            int: The iteration number as recorded in the checkpoint file.
        """
        # Load the checkpoint file and set the optimizer state
        state: Dict[str, Any] = th.load(save_file)
        self.backend._set_state(state)

        return state['iter']

    def _save_state(self, iter_num: int, save_file: PathLike):
        """Save the state of the optimizer (including graph parameters and the Adam optimizer state
        to a local file.

        Args:
            iter_num (int): Current iteration number.
            save_file (PathLike): Path to the output checkpoint file.
        """
        # Compile and save the backend state
        th.save(self.backend._get_state(iter_num), save_file)

    def _save_images(self, file_name: str, *imgs: th.Tensor, result_dir: Path = Path('.'),
                     img_format: str = 'png'):
        """Save image outputs from the material graph as local files.

        Args:
            file_name (str): The common file name of each saved image (note that images are written
                to their respective folders so no name conflict will occur).
            imgs (Iterable[Tensor], optional): Source images, assumed to be in the order of
                dictionary keys in `Renderer.CHANNELS` in `diffmat/core/render.py`.
            result_dir (Path, optional): Path to the output folder as a `pathlib.Path` object.
                Defaults to Path('.').
            img_format (str): Output image format ('png' or 'exr'). Defaults to 'png'.
        """
        for img, header in zip(imgs, self.image_headers):
            img_file = (result_dir / header / file_name).with_suffix(f'.{img_format}')
            write_image(img.detach().squeeze(0), img_file, img_format=img_format)

    def _export_sbs(self, sbs_file: PathLike):
        """Export the SVBRDF maps from the graph to an SBS document.

        Args:
            sbs_file (PathLike): Path to the output SBS document.
        """
        # Evaluate the graph using the best parameters
        with th.no_grad():
            maps = self.graph.evaluate_maps()

        # Obtain the output image dictionary and export to SBS document
        output_dict = dict(zip(self.image_headers[1:], maps))
        save_output_dict_to_sbs(output_dict, sbs_file)

    @input_check(1, class_method=True)
    def evaluate(self, img: th.Tensor, num_iters: int = 1000, result_dir: PathLike = '.',
                 load_checkpoint_file: Optional[PathLike] = None, enable_save: bool = True,
                 save_interval: int = 100, update_interval: int = 10,
                 save_output_sbs: bool = False, img_format: str = 'png', **backend_kwargs):
        """Run the optimizer.

        Args:
            img (Tensor): Target image.
            num_iters (int, optional): Number of optimization iterations to run. Defaults to 1000.
            result_dir (PathLike, optional): Path to the output folder. Defaults to '.'.
            load_checkpoint_file (PathLike, optional): Path to a checkpoint file to load from,
                which contains the state of the optimizer (including graph parameters and the
                optimizer state). Defaults to None.
            enable_save (bool, optional): Whether to save intermediate results. Defaults to True.
            save_interval (int, optional): Interval (in iterations) between saving intermediate
                results. Defaults to 100.
            update_interval (int, optional): Interval (in iterations) between updating the
                optimal loss value and parameter values. Defaults to 10.
            save_output_sbs (bool, optional): Whether to save the optimized SVBRDF maps to an SBS
                document. Defaults to False.
            img_format (str, optional): Output image format ('png' or 'exr'). Defaults to 'png'.
            backend_kwargs (Dict[str, Any], optional): Keyword arguments to pass into the backend
                optimization algorithm. Defaults to {}.
        """
        graph, backend, timer, logger = self.graph, self.backend, self.timer, self.logger

        # Prepare save file directories
        if enable_save:

            # Create result directories
            result_dir = Path(result_dir)
            checkpoint_dir = result_dir / 'checkpoints'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            for header in self.image_headers:
                (result_dir / header).mkdir(parents=True, exist_ok=True)

            # Define callback functions for checkpointing
            def save_state_callback(file_name: str, iter_num: int):
                self._save_state(iter_num, checkpoint_dir / f'{file_name}.pth')

            def save_image_callback(file_name: str, images: Tuple[th.Tensor, ...]):
                self._save_images(file_name, *images, result_dir=result_dir, img_format=img_format)

            def save(file_name: str, iter_num: int):
                maps = graph.evaluate_maps()
                save_state_callback(file_name, iter_num)
                save_image_callback(file_name, (graph.renderer(*maps), *maps))

        # Use dummy save functions
        else:
            save_state_callback, save_image_callback, save = [lambda *_: ...] * 3

        # Set the target image for the optimization backend
        backend.set_target_image(img)
        save_image_callback('target', img)

        # Load a previous checkpoint
        if load_checkpoint_file:
            load_iter = self._load_state(load_checkpoint_file) + 1
            logger.info(f'Loaded checkpoint from iter {load_iter}')
            start_iter = load_iter + 1

        # Save the initial state and image results
        else:
            save('initial', 0)
            logger.info('Saved initial state and rendering results')
            start_iter = 0

        # Time the optimization process
        with timer(f'Total optimization time ({num_iters - start_iter} iters)', unit='s'):
            param_min, loss_min, *res = \
                backend.evaluate(
                    num_iters=num_iters, start_iter=start_iter, save_interval=save_interval,
                    update_interval=update_interval, save_state_callback=save_state_callback,
                    save_image_callback=save_image_callback, **backend_kwargs)

        # Load optimized parameters and save the optimized result
        if loss_min < math.inf:
            backend._set_parameters(param_min)
            save('optimized', num_iters)

        # Save auxiliary data
        if enable_save:

            # Save the optimization history
            history: pd.DataFrame = res[0]
            history.to_csv(result_dir / 'history.csv', index=False)

            # Save optimized SVBRDF maps to an SBS document
            if save_output_sbs:
                export_dir = result_dir / 'export'
                export_dir.mkdir(parents=True, exist_ok=True)
                self._export_sbs(export_dir / 'diffmat_optimized.sbs')

        logger.info('Optimization finished')

    # Alias of the `evaluate` function that conforms to more conventional naming
    optimize = evaluate


class Optimizer(BaseOptimizer):
    """Continuous control parameter optimizer for differentiable procedural material graphs.
    """
    def __init__(self, graph: MaterialGraph, lr: float = 5e-4, metric: str = 'vgg',
                 loss_type: str = 'l1', metric_kwargs: Dict[str, Any] = {},
                 filter_exposed: int = FILTER_OFF, filter_generator: int = FILTER_OFF,
                 ablation_mode: str = 'none', **kwargs):
        """Initialize the continuous parameter optimizer.

        Args:
            graph (MaterialGraph): Source procedural material graph.
            lr (float, optional): Learning rate for the Adam optimizer. Defaults to 5e-4.
            metric (str, optional): Optimization objective type ('vgg' or 'fft'). Defaults to 'vgg'.
            loss_type (str, optional): Loss function type ('l1' or 'l2'). Defaults to 'l1'.
            metric_kwargs (Dict[str, Any], optional): Keyword arguments to pass into the
                optimization metric function. Defaults to {}.
            filter_exposed (int, optional): Option for return some or all optimizable parameters
                in the graph.
                    `1 = exclusive`: only exposed parameters are returned;
                    `0 = complement`: only non-exposed parameters are returned.
                    `-1 = all`: all parameters are returned.
                Defaults to `all`.
            filter_generator (int, optional): Option for node parameter visibility contigent on
                whether the node is (not) a generator node. Valid cases are:
                    `1 = yes` means parameters are visible only if the node is a generator;
                    `0 = no` means parameters are visible only if the node is not a generator;
                    `-1 = off` means node parameters are always visible.
                Defaults to `off`.
            ablation_mode (str, optional): Option for excluding some nodes from node parameter
                optimization. This option is useful for ablation studies. Valid options are:
                    `none`: no ablation;
                    `node`: ablate nodes that allow ablation;
                    `subgraph`: ablate predecessor subgraphs of nodes that allow ablation.
                Defaults to 'none'.
            kwargs (Dict[str, Any], optional): Keyword arguments to pass into the base class
                constructor.
        """
        param_io_kwargs = {'filter_exposed': filter_exposed, 'filter_generator': filter_generator}
        backend_kwargs = {'lr': lr}

        # Set the input graph as optimizable
        graph.train(ablation_mode=ablation_mode)

        super().__init__(graph, param_io_kwargs=param_io_kwargs, backend_kwargs=backend_kwargs,
                         metric=metric, loss_type=loss_type, metric_kwargs=metric_kwargs, **kwargs)


class IntegerOptimizer(BaseOptimizer):
    """Integer-valued discrete parameter optimizer for differentiable procedural material graphs.
    """
    def __init__(self, graph: MaterialGraph, algorithm: str = 'bo-ax',
                 backend_kwargs: Dict[str, Any] = {}, metric: str = 'vgg', loss_type: str = 'l1',
                 metric_kwargs: Dict[str, Any] = {}, filter_exposed: int = FILTER_OFF,
                 filter_generator: int = FILTER_OFF, **kwargs):
        """Initialize the integer-valued discrete parameter optimizer.

        Args:
            graph (MaterialGraph): Source procedural material graph.
            algorithm (str, optional): Integer parameter optimization algorithm ('bo-ax' or
                'bo-skopt'). Defaults to 'bo-ax'.
            backend_kwargs (Dict[str, Any], optional): Keyword arguments to pass into the node
                parameter optimization algorithm. Defaults to {}.
            metric (str, optional): Optimization objective type ('vgg' or 'fft'). Defaults to 'vgg'.
            loss_type (str, optional): Loss function type ('l1' or 'l2'). Defaults to 'l1'.
            metric_kwargs (Dict[str, Any], optional): Keyword arguments to pass into the
                optimization metric function. Defaults to {}.
            filter_exposed (int, optional): Option for return some or all optimizable parameters
                in the graph.
                    `1 = exclusive`: only exposed parameters are returned;
                    `0 = complement`: only non-exposed parameters are returned.
                    `-1 = all`: all parameters are returned.
                Defaults to `all`.
            filter_generator (int, optional): Option for node parameter visibility contigent on
                whether the node is (not) a generator node. Valid cases are:
                    `1 = yes` means parameters are visible only if the node is a generator;
                    `0 = no` means parameters are visible only if the node is not a generator;
                    `-1 = off` means node parameters are always visible.
                Defaults to `off`.
            kwargs (Dict[str, Any], optional): Keyword arguments to pass into the base class
                constructor.
        """
        param_io_kwargs = {'filter_exposed': filter_exposed, 'filter_generator': filter_generator}

        # Disable gradient-based optimization
        graph.eval()

        super().__init__(graph, param_type='integer', algorithm=algorithm,
                         param_io_kwargs=param_io_kwargs, backend_kwargs=backend_kwargs,
                         metric=metric, loss_type=loss_type, metric_kwargs=metric_kwargs, **kwargs)


class HybridOptimizer(BaseOptimizer):
    """Non-gradient-based optimizer for hybrid (continous and integer) parameter optimization.
    """
    def __init__(self, graph: MaterialGraph, algorithm: str = 'grid',
                 backend_kwargs: Dict[str, Any] = {}, metric: str = 'vgg', loss_type: str = 'l1',
                 metric_kwargs: Dict[str, Any] = {}, filter_integer: int = FILTER_OFF,
                 filter_exposed: int = FILTER_OFF, filter_generator: int = FILTER_OFF,
                 filter_int_exposed: int = FILTER_OFF, filter_int_generator: int = FILTER_OFF,
                 **kwargs):
        """Initialize the gradient-free hybrid parameter optimizer.

        Args:
            graph (MaterialGraph): Source procedural material graph.
            algorithm (str, optional): Hybrid parameter optimization algorithm ('simanneal'
                simulated annealing or 'grid' coordinate descent + grid search).
                Defaults to 'grid'.
            backend_kwargs (Dict[str, Any], optional): Keyword arguments to pass into the node
                parameter optimization algorithm. Defaults to {}.
            metric (str, optional): Optimization objective type ('vgg' or 'fft').
                Defaults to 'vgg'.
            loss_type (str, optional): Loss function type ('l1' or 'l2'). Defaults to 'l1'.
            metric_kwargs (Dict[str, Any], optional): Keyword arguments to pass into the
                optimization metric function. Defaults to {}.
            filter_integer (int, optional): Option for integer parameter optimization.
                Valid cases are:
                    `1 = yes`: only integer parameters are optimized;
                    `0 = no`: only continuous parameters are optimized;
                    `-1 = hybrid`: both integer and continuous parameters are optimized.
                Defaults to `hybrid`.
            filter_exposed (int, optional): See `filter_exposed` in `Optimizer`.
                Defaults to `all`.
            filter_generator (int, optional): See `filter_generator` in `Optimizer`.
                Defaults to `off`.
            filter_int_exposed (int, optional): See `filter_exposed` in `IntegerOptimizer`.
                Defaults to `all`.
            filter_int_generator (int, optional): See `filter_generator` in `IntegerOptimizer`.
                Defaults to `off`.
            kwargs (Dict[str, Any], optional): Keyword arguments to pass into the base class
                constructor.
        """
        # Set up optimization filters
        param_io_kwargs = {
            'continuous': {
                'filter_exposed': filter_exposed,
                'filter_generator': filter_generator,
            },
            'integer': {
                'filter_exposed': filter_int_exposed,
                'filter_generator': filter_int_generator,
            },
        }

        # Apply integer/continuous parameter switch
        if filter_integer == FILTER_YES:
            del param_io_kwargs['continuous']
        elif filter_integer == FILTER_NO:
            del param_io_kwargs['integer']

        # Disable gradient-based optimization
        graph.eval()

        super().__init__(graph, param_type='hybrid', algorithm=algorithm,
                         param_io_kwargs=param_io_kwargs, backend_kwargs=backend_kwargs,
                         metric=metric, loss_type=loss_type, metric_kwargs=metric_kwargs, **kwargs)
