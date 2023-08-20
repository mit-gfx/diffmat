from functools import partial
from pathlib import Path
from typing import Tuple, List, Dict, Union, Optional, Callable

import torch as th

from diffmat.core.base import BaseEvaluableObject
from diffmat.core.material import MaterialGraph, Renderer
from diffmat.core.io import write_image
from diffmat.core.types import Constant, PathLike, DeviceType
from diffmat.core.util import FILTER_OFF
from diffmat.translator.util import load_config


class ParamSampler(BaseEvaluableObject):
    """Randomly perturb or sample the optimizable parameters in a differentiable material graph.
    """
    def __init__(self, graph: MaterialGraph, mode: str = 'perturb', algo: str = 'uniform',
                 algo_kwargs: Dict[str, Constant] = {}, seed: int = 0,
                 filter_exposed: int = FILTER_OFF, filter_generator: int = FILTER_OFF,
                 device: DeviceType = 'cpu', **kwargs):
        """Initialize the random parameter sampler.

        Args:
            graph (MaterialGraph): Source procedural material graph.
            mode (str, optional): Sampling mode.
                `perturb`: Randomly sample perturbations that are added to original parameters.
                `sample`: Randomly sample parameter values directly.
                Defaults to 'perturb'.
            algo (str, optional): Sampling distribution type. Choices are uniform distribution
                'uniform' and normal distribution 'normal'. Defaults to 'uniform'.
            algo_kwargs (Dict[str, Constant], optional): Parameters of sampling distributions.
                Available parameters include:
                `min`, `max`: Lower/upper bounds of a uniform distribution.
                `mu`, `sigma`: Mean/stddev values of a normal distribution.
                Defaults to {}.
            seed (int, optional): Random seed. Defaults to 0.
            filter_exposed (int, optional): Option for sampling some or all optimizable parameters
                in the graph.
                `2 = all`: all parameters will be sampled;
                `1 = exclusive`: only exposed parameters are sampled;
                `0 = complement`: only non-exposed parameters are sampled.
                Defaults to 2.
            filter_quantize (int, optional): Additional constraint on sampled parameters.
                `2 = all`: all parameters will be sampled;
                `1 = exclusive`: only sample optimizable discrete parameters (quantized continuous
                    parameters);
                `0 = complement`: only sample continuous parameters that do not undergo
                    quantization.
                Defaults to 2.
            device (DeviceType, optional): Device placement of the random parameter sampler, the
                same `device` parameter for creating PyTorch tensors are applicable here.
                Defaults to 'cpu'.

        Raises:
            ValueError: Unknown sampling mode.
            ValueError: The source procedural material graph does not have optimizable parameters.
        """
        if mode not in ('perturb', 'sample'):
            raise ValueError("Sampler mode must be either 'perturb' or 'sample'")

        super().__init__(device=device, **kwargs)

        self.graph = graph
        self.mode = mode

        # Set functions for graph parameter reading and writing
        param_kwargs = {
            'filter_exposed': filter_exposed, 'filter_generator': filter_generator,
            'filter_requires_grad': FILTER_OFF
        }
        self._get_parameters = partial(graph.get_parameters_as_tensor, **param_kwargs)
        self._set_parameters = partial(graph.set_parameters_from_tensor, **param_kwargs)
        self._num_parameters = partial(graph.num_parameters, **param_kwargs)
        self._get_all_parameters = partial(
            graph.get_parameters_as_tensor, filter_requires_grad=FILTER_OFF)

        # Check if there are parameters that pass the user's filter
        num_params = self._num_parameters()
        if not num_params:
            raise ValueError('The material graph does not have optimizable parameters')

        # Show graph info
        self.logger.info(f'Graph name: {graph.name}  #nodes: {len(graph.nodes)}  '
                         f'#params: {num_params}')

        # Setup the sampling function
        self.func = self._get_sampling_func(algo, **algo_kwargs)

        # Image headers in the output dict
        self.image_headers = [*Renderer.CHANNELS.keys(), 'render']

        # Initialize the internal random number generator state without interfering with other
        # random sampling processes
        self.reset(seed)

    def reset(self, seed: int = 0):
        """Reset the internal random number generate state.

        Args:
            seed (int, optional): New random seed.
        """
        # Initialize the internal random number generator state without interfering with other
        # random sampling processes
        with self.temp_rng(seed=seed) as state:
            self.rng_state = state

    def _get_sampling_func(self, algo: str, **algo_kwargs: Constant) -> \
            Callable[[th.Tensor], th.Tensor]:
        """Factory method that produces a sampling worker function.

        Args:
            algo (str): Sampling algorithm name.
            algo_kwargs (Dict[str, Constant], optional): Parameters to the sampling algorithm.

        Raises:
            ValueError: Unrecognized sampling algorithm.

         Returns:
            Callable[[Tensor], Tensor]: A sampling function that takes as input the original
                parameters and generates sampled or perturbed parameters within [0, 1].
        """
        # Uniform sampling
        if algo == 'uniform':
            lb: float; ub: float
            lb, ub = algo_kwargs['min'], algo_kwargs['max']

            def func(params: th.Tensor) -> th.Tensor:
                rands = th.rand_like(params) * (ub - lb) + lb
                return (rands if self.mode == 'sample' else params * (1 + rands)).clamp_(0.0, 1.0)

        # Normal sampling
        elif algo == 'normal':
            mu: float; sigma: float
            mu, sigma = algo_kwargs['mu'], algo_kwargs['sigma']

            def func(params: th.Tensor) -> th.Tensor:
                rands = th.randn_like(params) * sigma + mu
                return (rands if self.mode == 'sample' else rands.add_(params)).clamp_(0.0, 1.0)

        else:
            raise ValueError(f'Unrecognized sampling algorithm: {algo}')

        return func

    def _save_images(self, case_num: int = 0, *imgs: th.Tensor, result_dir: Path = Path('.'),
                     img_format: str = 'png'):
        """Save image outputs from the material graph as local files.

        Args:
            case_num (int, optional): Group index of sampled parameters. Defaults to 0.
            result_dir (Path, optional): Output folder for saved texture images (must be a
                `pathlib.Path` object). Defaults to Path('.').
            img_format (str, optional): Texture image format. Defaults to 'png'.
        """
        for img, header in zip(imgs, self.image_headers):
            img_file = result_dir / header / f'params_{case_num}.{img_format}'
            write_image(img.detach().squeeze(0), img_file, img_format=img_format)

    def sample_batch(self, batch_size: int, params: Optional[th.Tensor] = None) -> th.Tensor:
        """Sample a batch of parameters from the random parameter distribution.

        Args:
            batch_size (int): Batch size.
            params (Optional[Tensor], optional): Manually specify the source parameter values (only
                effective when the sampling mode is 'perturb'). None means using current parameter
                values of the graph. Defaults to None.

        Returns:
            Tensor: Sampled parameter values of size `(batch_size, num_parameters)`.
        """
        # Starting parameters
        params = params if params is not None else self._get_parameters()

        # Perform sampling using the internal random number generator state
        with self.temp_rng(self.rng_state):
            sampled_params = self.func(params.expand(batch_size, params.shape[-1]))
            self.rng_state = self.get_rng_state()

        return sampled_params

    def evaluate(self, num: int = 1, config_file: str = '', save_result: bool = True,
                 result_dir: PathLike = '.', img_format: str = 'png',
                 return_params: bool = False) -> Union[
                     List[th.Tensor], Tuple[List[th.Tensor], List[th.Tensor]]]:
        """Draw from the specified random parameter distribution for a certain number of times.
        Return and save the result images as local files while optionally returning sampled
        parameter values.

        Args:
            num (int, optional): Number of sampled parameter groups, i.e., how many times the
                sampling happens. Defaults to 1.
            config_file (str, optional): Path to a sampling configuration file, which specifies
                the set of parameters to hold designated values during sampling. See method
                `set_parameters_from_config` in the `MaterialGraph` class in
                `diffmat/core/graph.py` for configuration file format. Defaults to ''.
            save_result (bool, optional): Switch for saving sampled parameters and resulting images
                to local files. Defaults to True.
            result_dir (PathLike, optional): Output folder for saved parameters and images.
                Defaults to '.'.
            img_format (str, optional): Output image format ('png' or 'exr'). Defaults to 'png'.
            return_params (bool, optional): Return sampled parameters alongside rendered images.
                Defaults to False.

        Raises:
            ValueError: Number of cases is not a positive integer.

        Returns:
            List[Tensor]: Rendered texture images from sampled graph parameters.
            List[Tensor] (optional): Sampled graph parameters.
        """
        if not isinstance(num, int) or num <= 0:
            raise ValueError('Number of cases must be a positive integer')

        graph = self.graph

        # Create the output image folders
        result_dir = Path(result_dir)
        if save_result:
            for header in [*self.image_headers, 'param']:
                (result_dir / header).mkdir(parents=True, exist_ok=True)

        # Load the sampling config file
        if config_file:
            param_config: Dict[str, Dict[str, Constant]] = load_config(config_file)
            self.logger.info(f'Loaded parameter configuration ({len(param_config)} entries)')
        else:
            param_config = {}

        # Load the internal random number generator state
        with self.temp_rng(self.rng_state):

            # Get the current parameter setting
            params = self._get_parameters()

            # Perturb/sample new parameters using the worker function
            img_list: List[th.Tensor] = []
            new_params_list: List[th.Tensor] = []

            for it in range(num):
                new_params = self.func(params)
                new_params_list.append(new_params)
                self._set_parameters(new_params)
                graph.set_parameters_from_config(param_config)

                # Run forward evaluation using new parameters
                with th.no_grad():
                    maps = graph.evaluate_maps()
                    render = graph.renderer(*maps)
                    img_list.append(render)

                # Save the result images and parameter values
                if save_result:
                    self._save_images(it, *maps, render, result_dir=result_dir,
                                      img_format=img_format)
                    param_file = result_dir / 'param' / f'params_{it}.pth'
                    th.save({'param': self._get_all_parameters()}, param_file)

            # Restore the initial parameters
            self._set_parameters(params)

            # Record the current state
            self.rng_state = self.get_rng_state()

        if return_params:
            return img_list, new_params_list
        else:
            return img_list
