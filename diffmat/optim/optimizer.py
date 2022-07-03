from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
import time
import math

from torch.optim import Adam
from torch.fft import rfft2
import torch as th
import torch.nn.functional as F
import pandas as pd

from .descriptor import TextureDescriptor
from ..core.base import BaseEvaluableObject
from ..core.graph import MaterialGraph
from ..core.io import write_image, save_output_dict_to_sbs
from ..core.render import Renderer
from ..core.types import PathLike
from ..core.util import input_check


class Optimizer(BaseEvaluableObject):
    """Continuous control parameter optimizer for differentiable procedural material graphs.
    """
    def __init__(self, graph: MaterialGraph, lr: float = 5e-4, metric: str = 'td',
                 opt_level_exposed: int = 2):
        """Initialize the continuous parameter optimizer.

        Args:
            graph (MaterialGraph): Source procedural material graph.
            lr (float, optional): Learning rate for the Adam optimizer. Defaults to 5e-4.
            metric (str, optional): Optimization objective type ('td' or 'fft'). Defaults to 'td'.
            opt_level_exposed (int, optional): Option for optimizing some or all parameters in
                the graph.
                `2 = all`: all parameters will be optimized;
                `1 = exclusive`: only exposed parameters are optimized;
                `0 = complement`: only non-exposed parameters are optimized.
                Defaults to 2.
        """
        super().__init__(device=graph.device)

        # Set the input graph as optimizable
        self.graph = graph
        self.graph.train()

        self.metric = metric
        self.opt_level_kwargs = {'level_exposed': opt_level_exposed}

        # Initialize the Adam optimizer
        self.graph_params = lambda: graph.parameters(**self.opt_level_kwargs)
        self.optimizer = Adam(self.graph_params(), lr=lr)

        # Show graph info
        self.logger.info(f'Graph name: {graph.name}  #nodes: {len(graph.nodes)}  '
                         f'#params: {sum(param.numel() for param in self.graph_params())}')

        # Image headers in the output dict
        self.image_headers = ['render', *Renderer.CHANNELS.keys()]

        # Save the initial state of the graph and the optimizer
        self.init_params = graph.get_parameters_as_tensor()

        # Initialize running loss minimum
        self.loss_min = math.inf
        self.param_min = self.init_params

    def reset(self, lr: float = 5e-4):
        """Reset the optimizer to its initial state, including graph parameters and the internal
        state of the Adam optimizer.

        Args:
            lr (float, optional): Learning rate to override the existing one. Defaults to 5e-4.
        """
        # Reset graph parameters and Adam optimizer
        self.graph.set_parameters_from_tensor(self.init_params)
        self.optimizer = Adam(self.graph_params(), lr=lr)

        # Reset running loss minimum
        self.loss_min = math.inf
        self.param_min = self.init_params

    def _load_state(self, save_file: PathLike) -> int:
        """Load the state of the optimizer (including graph parameters and the Adam optimizer state
        from a local save file. Returns the iteration number.

        Args:
            save_file (PathLike): Path to the checkpoint file to load from.

        Returns:
            int: The iteration number as recorded in the checkpoint file.
        """
        state: Dict[str, Any] = th.load(save_file)
        params: Optional[th.Tensor] = state['param']
        if params is not None:
            self.graph.set_parameters_from_tensor(params.to(self.device))
        self.optimizer.load_state_dict(state['optim'])
        self.loss_min: float = state['loss_min']
        self.param_min: th.Tensor = state['param_min']

        return state['iter']

    def _save_state(self, iter_num: int, save_file: PathLike):
        """Save the state of the optimizer (including graph parameters and the Adam optimizer state
        to a local file.

        Args:
            iter_num (int): Current iteration number.
            save_file (PathLike): Path to the output checkpoint file.
        """
        state = {
            'iter': iter_num,
            'param': self.graph.get_parameters_as_tensor().cpu(),
            'optim': self.optimizer.state_dict(),
            'loss_min': self.loss_min,
            'param_min': self.param_min.cpu()
        }
        th.save(state, save_file)

    def save_images(self, file_name: str, *imgs: th.Tensor, result_dir: Path = Path('.'),
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

    @input_check(1, class_method=True)
    def evaluate(self, img: th.Tensor, num_iters: int = 1000, start_iter: int = 0,
                 load_checkpoint: bool = False, load_checkpoint_dir: Optional[PathLike] = None,
                 save_interval: int = 20, result_dir: PathLike = '.',
                 save_output_sbs: bool = False, img_format: str = 'png'):
        """Run the optimizer.

        Args:
            img (Tensor): Target image as a 4D tensor (1xCxHxW).
            num_iters (int, optional): Number of optimization iterations in total.
                Defaults to 1000.
            start_iter (int, optional): Iteration number to start optimization from. A non-zero
                iteration number is typically specified when loading from a checkpoint save.
                Defaults to 0.
            load_checkpoint (bool, optional): Whether to load a previous checkpoint from iteration
                `start_iter`. Defaults to False.
            load_checkpoint_dir (Optional[PathLike], optional): Checkpoint files folder. If not
                provided, the optimizer will use a default location (the 'checkpoints' sub-folder
                in `result_dir`). Defaults to None.
            save_interval (int, optional): Number of iterations between two checkpoint saves.
                Defaults to 20.
            result_dir (PathLike, optional): Output folder for storing checkpoints, intermediate
                outputs, and the final result. Defaults to '.'.
            save_output_sbs (bool, optional): Export the SVBRDF maps of the optimized texture into
                a SBS document. Defaults to False.
            img_format (str, optional): Output image format ('png' or 'exr'). Defaults to 'png'.

        Raises:
            ValueError: Starting iteration number exceeds the number of iterations in total.
            ValueError: Invalid loss function type.
        """
        graph = self.graph
        optimizer = self.optimizer

        # Create result directories
        result_dir = Path(result_dir)
        checkpoint_dir = result_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for header in self.image_headers:
            (result_dir / header).mkdir(parents=True, exist_ok=True)

        # Save the target image first
        self.save_images('target', img, result_dir=result_dir, img_format=img_format)

        # Check input validity
        if start_iter >= num_iters:
            raise ValueError(f'Start iteration ({start_iter}) exceeds the number of iterations '
                             f'({num_iters})')

        # Load a previous checkpoint
        if load_checkpoint:
            load_dir = Path(load_checkpoint_dir or checkpoint_dir)
            load_iter = self._load_state(load_dir / f'iter_{start_iter}.pth')

            self.logger.info(f'Loaded checkpoint from iter {load_iter}')

        # Save the initial state and image results
        else:
            self._save_state(start_iter, checkpoint_dir / f'iter_{start_iter}.pth')
            with th.no_grad():
                maps = graph.evaluate_maps()
                render = graph.renderer(*maps)
                self.save_images(f'iter_{start_iter}', render, *maps, result_dir=result_dir,
                                 img_format=img_format)

            self.logger.info('Saved initial state and rendering results')

        # Choose the metric function for loss computation
        if self.metric == 'td':
            metric_func = TextureDescriptor(device=self.device).evaluate
        elif self.metric == 'fft':
            metric_func = lambda x: rfft2(x).abs()
        else:
            raise ValueError(f'Invalid metric type: {self.metric}')

        # Compute the metric of the target image
        target_td = metric_func(img.detach().to(self.device))

        # Optimization history (loss, timing) and record of the best iteration
        history: List[Tuple[int, float, float, float]] = []

        # Time the optimization process
        with self.timer(f'Total optimization time ({num_iters - start_iter} iters)'):

            for it in range(start_iter, num_iters):

                # Forward evaluation
                optimizer.zero_grad()

                t_start = time.time()
                maps = graph.evaluate_maps()
                render = graph.renderer(*maps)
                td = metric_func(render)
                t_forward = time.time() - t_start

                # Compute loss and update the running minimum
                loss = F.l1_loss(td, target_td)
                loss_val = loss.detach().item()

                if loss_val < self.loss_min:
                    self.loss_min = loss_val
                    self.param_min = graph.get_parameters_as_tensor()

                # Run backward evaluation
                t_start = time.time()
                loss.backward()
                t_backward = time.time() - t_start

                optimizer.step()

                # Save the state with output images
                if it > start_iter and not (it + 1) % save_interval:
                    self._save_state(it, result_dir / 'checkpoints' / f'iter_{it}.pth')
                    self.save_images(f'iter_{it}', render, *maps, result_dir=result_dir,
                                     img_format=img_format)

                # Log loss and time consumption
                history.append((it, loss_val, t_forward, t_backward))
                self.logger.info(
                    f'Iter {it}: loss = {loss_val:.6f}  '
                    f'forward time - {t_forward * 1e3:.3f} ms  '
                    f'backward time - {t_backward * 1e3:.3f} ms')

        # Save the optimization history
        df = pd.DataFrame(history, columns=['iter', 'loss', 't_forward', 't_backward'])
        df.to_csv(result_dir / 'history.csv', index=False)

        # Save optimized SVBRDF maps to an SBS document
        if save_output_sbs:

            # Create the optimization export folder
            export_dir = result_dir / 'export'
            export_dir.mkdir(parents=True, exist_ok=True)

            # Evaluate the graph using the best parameters to obtain the output image dictionary
            with th.no_grad():
                graph.set_parameters_from_tensor(self.param_min)
                maps = graph.evaluate_maps()
            output_dict = dict(zip(self.image_headers[1:], maps))

            save_output_dict_to_sbs(output_dict, export_dir / 'diffmat_optimized.sbs')

        self.logger.info('Optimization finished')

    # Alias of the `evaluate` function that conforms to more conventional naming
    optimize = evaluate
