from pathlib import Path
from typing import Tuple, List, Dict, Callable, Union, Any

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.functional import l1_loss
from torchvision.utils import make_grid
import torch as th
import torch.nn as nn

from diffmat.core.base import BaseEvaluableObject
from diffmat.core.material import MaterialGraph, Renderer
from diffmat.core.material.util import input_check
from diffmat.core.io import write_image
from diffmat.core.types import DeviceType, PathLike
from diffmat.optim.descriptor import TextureDescriptor
from diffmat.optim.sampler import ParamSampler


class ParamPredictor(BaseEvaluableObject):
    """Control parameter prediction for differentiable procedural material graphs.
    """
    def __init__(self, graph: MaterialGraph, arch_kwargs: Dict[str, Any] = {},
                 sampler_kwargs: Dict[str, Any] = {}, device: DeviceType = 'cpu', **kwargs):
        """Initialize the control parameter predictor.

        Args:
            graph (MaterialGraph): Differentiable material graph for which a predictor is trained.
            arch_kwargs (Dict[str, Any], optional): Architectural parameters for the predictor's
                MLP network. See method `_build_network`. Defaults to {}.
            sampler_kwargs (Dict[str, Any], optional): Control parameters of the predictor's random
                parameter sampler for online training data generation. See the constructor of the
                `ParamSampler` class in `diffmat/optim/sampler.py`. Defaults to {}.
            device (DeviceType, optional): Device placement of the network predictor (e.g., CPU or
                GPU), the same as PyTorch's `device` parameter. Defaults to 'cpu'.
            kwargs (Dict[str, Any], optional): Keyword arguments that will be passed to the parent
                class constructor.
        """
        super().__init__(device=device, **kwargs)

        # Random parameter sampler for generating training data (image -> param) on the fly
        self.graph = graph
        self.sampler = ParamSampler(graph, device=device, **sampler_kwargs)

        # Parameter prediction network architecture
        self._build_network(**arch_kwargs)

    def _build_network(self, num_layers: int = 3, layer_size_max: int = 960,
                       layer_size_param_mult: int = 3, td_pyramid_level: int = 2):
        """Build the parameter prediction neural network by concatenating a texture descriptor
        network and an MLP.

        Args:
            num_layers (int, optional): Number of fully-connected layers in MLP. Defaults to 3.
            layer_size_max (int, optional): Max width of each intermediate layer. Defaults to 960.
            layer_size_param_mult (int, optional): Intermediate layer width as a multiplier to the
                number of optimizable parameters. The width is capped by `layer_size_max`.
                Defaults to 3.
            td_pyramid_level (int, optional): Mipmap stack levels when calculating the texture
                descriptor of an image. The feature vector size grows proportionally with this
                number. Defaults to 2.

        Raises:
            TypeError: Number of layers is not an integer.
            ValueError: Number of layers is non-positive.
        """
        # Check input validity
        if not isinstance(num_layers, int):
            raise TypeError(f'Number of layers for the parameter regression network must be an '
                            f'integer, but got {type(num_layers).__name__}')
        elif num_layers <= 0:
            raise ValueError(f'Number of layers for the parameter regression network must be a '
                             f'positive integer, but got {num_layers}')

        # Texture descriptor network
        self.descriptor = TextureDescriptor(device=self.device)
        self.td_pyramid_level = td_pyramid_level

        # Obtain texture descriptor feature size
        img_size = 1 << self.graph.res
        img_dummy = th.zeros(1, 3, img_size, img_size, device=self.device)
        td_dummy = self.descriptor.evaluate(img_dummy, td_level=td_pyramid_level)
        td_size = td_dummy.shape[1]

        # Construct an MLP feature-to-parameter regression network
        ## Calculate FC layer widths
        num_params = self.sampler._num_parameters()
        fc_size = min(num_params * layer_size_param_mult, layer_size_max)

        ## Build network architecture
        layers: List[nn.Module] = []
        for i in range(num_layers):
            in_features = fc_size if i else td_size
            out_features = fc_size if i < num_layers - 1 else num_params
            activation = nn.LeakyReLU() if i < num_layers - 1 else nn.Sigmoid()
            layers.append(nn.Linear(in_features, out_features, device=self.device))
            layers.append(activation)

        ## Create the MLP
        self.model = nn.Sequential(*layers)

    def train(self, num_epochs: Union[int, Tuple[int, int]] = (100, 50), epoch_iters: int = 100,
              td_loss_weight: Union[float, Tuple[float, float]] = (0.0, 0.1), lr: float = 1e-4,
              lr_decay_coeff: float = 0.5, batch_size: int = 5, start_epoch: int = 0,
              load_checkpoint: bool = False, load_checkpoint_dir: PathLike = '.',
              log_interval: int = 10, save_interval: int = 10, result_dir: PathLike = '.',
              save_validation_imgs: bool = True, img_format: str = 'png'):
        """Train the parameter prediction network using synthetically generated textures from the
        random sampler.

        Args:
            num_epochs (int | Tuple[int, int], optional): Number of training epochs before and
                after introducing perceptual (texture descriptor) loss to the objective function.
                Passing one integer only will disable perceptual loss during training.
                Defaults to (100, 50).
            epoch_iters (int, optional): Number of iterations (mini-batches) in each epoch.
                Defaults to 100.
            td_loss_weight (float | Tuple[float, float], optional): Weight of the perceptual loss
                term before and after being added to the loss function. Defaults to (0.0, 0.1).
            lr (float, optional): Initial learning rate. Defaults to 1e-4.
            lr_decay_coeff (float, optional): Controls the learning decay rate. The learning rate
                of each epoch is calculated by `lr = 1 / (epoch * lr_decay_coeff + 1)`.
                Defaults to 0.5.
            batch_size (int, optional): Mini-batch size. Defaults to 5.
            start_epoch (int, optional): Epoch number to start training from, often set to non-zero
                for checkpoint loading. Defaults to 0.
            load_checkpoint (bool, optional): Whether to load a previous checkpoint from epoch
                `start_epoch`. Defaults to False.
            load_checkpoint_dir (PathLike, optional): Checkpoint files folder. Defaults to '.'.
            log_interval (int, optional): Number of iterations between two consecutive on-screen
                logging, including the epoch/iteration number and the loss value. Defaults to 10.
            save_interval (int, optional): Number of epochs between two checkpoint saves.
                Defaults to 10.
            result_dir (PathLike, optional): Output folder for storing checkpoints and validation
                images, etc. Defaults to '.'.
            save_validation_imgs (bool, optional): Save validation images to local files. The
                images show a side-by-side comparison between ground-truth textures and renderings
                using network-predicted parameters. Defaults to True.
            img_format (str, optional): Image format for saved validation images ('png' or 'exr').
                Defaults to 'png'.

        Raises:
            ValueError: Starting epoch number exceeds the total number of epochs.
            RuntimeError: Batch size is larger than 1 when optimizing with perceptual (texture
                descriptor) loss. This is for debugging only and does not happen in practical use.
        """
        graph, model, descriptor, sampler = self.graph, self.model, self.descriptor, self.sampler
        td_pyramid_level = self.td_pyramid_level
        logger = self.logger

        # Backup initial graph parameters
        init_params = sampler._get_parameters()

        # Create result directories
        result_dir = Path(result_dir)
        checkpoint_dir = result_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if save_validation_imgs:
            valid_result_dir = result_dir / 'validation'
            valid_result_dir.mkdir(parents=True, exist_ok=True)

        # Extract two-phase training plan from input arguments
        if isinstance(num_epochs, int):
            num_epochs = (num_epochs, 0)
        if isinstance(td_loss_weight, float):
            td_loss_weight = (td_loss_weight, td_loss_weight)

        # Check input validity
        if start_epoch >= sum(num_epochs):
            raise ValueError(f'Start epoch ({start_epoch}) exceeds the number of epochs in total '
                             f'({sum(num_epochs)})')

        # Initialize Adam optimizer and learning rate scheduler
        optimizer = Adam(model.parameters(), lr=lr, eps=1e-6)

        decay_func: Callable[[int], float] = lambda epoch: 1 / ((epoch + 1) * lr_decay_coeff + 1)
        scheduler = LambdaLR(optimizer, lr_lambda = decay_func, last_epoch = start_epoch - 1)

        # Load from a previous checkpoint
        if load_checkpoint:
            load_dir = Path(load_checkpoint_dir or checkpoint_dir)
            state_dict: Dict[str, Any] = th.load(load_dir / f'epoch_{start_epoch}.pth')

            self.model.load_state_dict(state_dict['model'])
            optimizer.load_state_dict(state_dict['optim'])
            scheduler.load_state_dict(state_dict['sched'])

            self.logger.info(f'Loaded checkpoint from epoch {start_epoch}')

        # Loss function
        def criterion(pred: th.Tensor, gt: th.Tensor, gt_imgs: th.Tensor,
                      td_loss_weight: float = 0.0) -> th.Tensor:

            # Default L1 loss
            loss = l1_loss(pred, gt)

            # Texture descriptor loss
            if td_loss_weight > 0:
                if pred.shape[0] > 1:
                    raise RuntimeError('Texture descriptor loss does not support batch evaluation')

                # Compute the descriptor of the predicted image
                sampler._set_parameters(pred.squeeze(0))
                pred_img = graph.evaluate()
                pred_td = descriptor.evaluate(pred_img, td_level=td_pyramid_level)

                # Compute loss against the ground truth descriptor
                gt_td = descriptor.evaluate(gt_imgs, td_level=td_pyramid_level)
                loss = loss + l1_loss(pred_td, gt_td) * td_loss_weight

            return loss

        # Batch data generation function
        def gen_batch(batch_size: int) -> Tuple[th.Tensor, th.Tensor]:

            # Generate parameter variation from the sampler
            gt = sampler.sample_batch(batch_size, params=init_params)

            # Compute the rendered textures of the generated parameters
            gt_img_list: List[th.Tensor] = []
            for params in gt.unbind():
                sampler._set_parameters(params)
                gt_img_list.append(graph.evaluate())

            return th.cat(gt_img_list, dim=0), gt

        logger.info('Start network training')

        # Main training loop
        for epoch in range(start_epoch, sum(num_epochs)):

            # Determine the training phase
            switched = epoch >= num_epochs[0]
            batch_size_epoch = batch_size if not switched else 1
            td_loss_weight_epoch = td_loss_weight[0] if not switched else td_loss_weight[1]

            # Compute average loss
            loss_total = 0.0

            for it in range(epoch_iters):

                # Generate a batch of images and their corresponding parameters
                gt_imgs, gt = gen_batch(batch_size_epoch)

                # Forward evaluation
                optimizer.zero_grad()
                gt_td = descriptor.evaluate(gt_imgs, td_level=td_pyramid_level)
                pred: th.Tensor = model(gt_td)
                loss = criterion(pred, gt, gt_imgs, td_loss_weight=td_loss_weight_epoch)

                loss_val = loss.detach().item()
                loss_total += loss_val

                # Backward evaluation
                loss.backward()
                optimizer.step()

                # Print iteration info
                if not (it + 1) % log_interval:
                    logger.info(f'Epoch {epoch}, iter {it}: loss = {loss_val:.6f}')

            # Decay learning rate
            scheduler.step()

            # Perform validation
            with th.no_grad():
                gt_imgs, gt = gen_batch(batch_size_epoch)
                gt_td = descriptor.evaluate(gt_imgs, td_level=td_pyramid_level)
                pred: th.Tensor = model(gt_td)
                loss = criterion(pred, gt, gt_imgs)
                loss_val = loss.item()

            # Optionally save the input vs. predicted texture image pairs during validation in
            # a grid
            if save_validation_imgs:

                # Collect input-prediction pairs
                valid_imgs: List[th.Tensor] = []
                for params, gt_img in zip(pred.unbind(), gt_imgs.unbind()):
                    sampler._set_parameters(params)
                    pred_img = graph.evaluate()
                    valid_imgs.extend((gt_img, pred_img.squeeze(0)))

                # Generate and save the validation image grid
                grid_img = make_grid(valid_imgs, nrow=10, padding=16)
                write_image(grid_img, valid_result_dir / f'epoch_{epoch}', img_format=img_format)

            # Print epoch summary
            loss_total /= epoch_iters
            logger.info(f'Epoch {epoch} summary: average training loss = {loss_total:.6f}, '
                        f'validation loss (parameter only) = {loss_val:.6f}')

            # Save a checkpoint 
            if epoch > start_epoch and not (epoch + 1) % save_interval:
                state_dict: Dict[str, Dict[str, Any]] = {
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'sched': scheduler.state_dict()
                }
                th.save(state_dict, checkpoint_dir / f'epoch_{epoch}.pth')

        logger.info('Network training finished')

        # Restore initial graph parameters
        sampler._set_parameters(init_params)

    @input_check(1, channel_specs='c', class_method=True)
    def evaluate(self, img: th.Tensor, save_result: bool = False, result_dir: PathLike = '.',
                 img_format: str = 'png') -> th.Tensor:
        """Infer parameter values from a batch of input images using the parameter prediction
        network.

        Args:
            img (Tensor): Input mini-batch of images.
            save_result (bool, optional): Save rendered textures from network prediction to local
                files. Defaults to False.
            result_dir (PathLike, optional): Output folder where aforementioned rendered textures
                are stored. Defaults to '.'.
            img_format (str, optional): Image format of rendered textures ('png' or 'exr').
                Defaults to 'png'.

        Returns:
            Tensor: Network-predicted parameter values in the procedural material graph.
        """
        # Obtain predicted parameters
        td = self.descriptor.evaluate(img.to(self.device), td_level=self.td_pyramid_level)
        params: th.Tensor = self.model(td)

        # Save predicted parameters to local files
        if save_result:
            graph = self.graph
            param_kwargs = self.sampler.param_kwargs

            # Back up initial graph parameters
            init_params = graph.get_parameters_as_tensor(**param_kwargs)

            # Create output folders
            result_dir = Path(result_dir)
            image_headers = [*Renderer.CHANNELS.keys(), 'render']
            for header in [*image_headers, 'param']:
                (result_dir / header).mkdir(parents=True, exist_ok=True)

            # Process each paramter group in the mini-batch
            for i, param in enumerate(params.unbind()):

                # Generate SVBRDF maps and the rendering
                with th.no_grad():
                    graph.set_parameters_from_tensor(param, **param_kwargs)
                    maps = graph.evaluate_maps()
                    render = graph.renderer(*maps)

                # Save images to respective folders
                for img, header in zip([*maps, render], image_headers):
                    img_file = result_dir / header / f'params_{i}.{img_format}'
                    write_image(img.detach().squeeze(0), img_file, img_format=img_format)

                # Save the predicted parameters
                param_file = result_dir / 'param' / f'params_{i}.pth'
                th.save({'param': param}, param_file)

            # Restore initial graph parameters
            graph.set_parameters_from_tensor(init_params, **param_kwargs)

        return params
