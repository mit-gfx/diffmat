from abc import abstractmethod
from functools import partial
from typing import Tuple, List, Dict, Iterator, Iterable, Callable, Type, Optional, Union, Any
import itertools
import math
import random

from torch.optim import Adam
from simanneal import Annealer
from numpy.random import default_rng
import torch as th
import numpy as np
import numpy.typing as npt
import pandas as pd

from diffmat.core.base import BaseEvaluableObject
from diffmat.core.types import ParamConfig
from diffmat.core.material import MaterialGraph
from diffmat.core.util import to_numpy, FILTER_OFF
from diffmat.optim.metric import BaseMetric


# Type aliases
SaveStateCallback = Callable[[str, int], None]
SaveImageCallback = Callable[[str, Iterable[th.Tensor]], None]
HistoryType = List[List[Union[int, float, str]]]
SAState = npt.NDArray[np.float32]


class BaseBackend(BaseEvaluableObject):
    """Base class for all node parameter optimization backends. A backend functions as an adapter
    between Diffmat and external libraries, and supports parameter checkpointing.
    """
    def __init__(self, graph: MaterialGraph, loss_func: BaseMetric,
                 param_io_kwargs: Dict[str, Any] = {}, **kwargs):
        """Initialize the backend.
        """
        super().__init__(device=graph.device, **kwargs)

        self.graph = graph
        self.loss_func = loss_func
        self.param_io_kwargs = param_io_kwargs

        # Set the best solution as the initial parameters
        self.init_param = self._get_parameters()
        self.loss_min = math.inf
        self.param_min = self.init_param

        # Counter for the number of evaluations
        self.eval_counter = 0

        # Optimization history (iter, loss, timing) and construction function
        default_columns = ['iter', 'loss', 't_forward']
        self.history: HistoryType = []
        self._make_history: Callable[[List[str]], pd.DataFrame] = \
            lambda columns=default_columns: pd.DataFrame(self.history, columns=columns)

    @abstractmethod
    def _get_parameters(self) -> Any:
        """Read optimizable parameters.
        """
        ...

    @abstractmethod
    def _set_parameters(self, _: Any):
        """Assign values to optimizable parameters.
        """
        ...

    @abstractmethod
    def _num_parameters(self) -> int:
        """Count the number of optimizable parameters.
        """
        ...

    def reset(self):
        """Reset the backend to its initial state.
        """
        # Reset the graph parameters to initial values
        self._set_parameters(self.init_params)

        # Reset running loss minima
        self.loss_min = math.inf
        self.param_min = self.init_params

    def set_target_image(self, img: th.Tensor):
        """Set the target image for the loss function.
        """
        self.loss_func.set_target_image(img)

    def _get_worker_state(self) -> Dict[str, Any]:
        """Get the optimization worker's current state. Returns an empty dictionary by default if
        no information needs to be recorded.
        """
        return {}

    def _set_worker_state(self, _: Dict[str, Any]):
        """Set the optimization worker's current state. Does nothing by default.
        """
        pass

    def _get_state(self, iter_num: int) -> Dict[str, Any]:
        """Get the optimization backend's current state, including graph parameters, the best
        solution found, and the optimization worker state.
        """
        # Get graph parameters
        param = self.graph.get_parameters_as_tensor(filter_requires_grad=FILTER_OFF).cpu()
        param_int = self.graph.get_integer_parameters_as_list()

        # Get the optimization worker state
        worker_state = self._get_worker_state()

        # Get the current best solution
        to_cpu = lambda x: x.cpu() if isinstance(x, th.Tensor) else x
        loss_min = self.loss_min
        param_min = to_cpu(self.param_min)

        return {'iter': iter_num, 'param': param, 'param_int': param_int, 'optim': worker_state,
                'loss_min': loss_min, 'param_min': param_min}

    def _set_state(self, state: Dict[str, Any]):
        """Set the optimization backend's state, including graph parameters, the best
        solution found, and the optimization worker state.
        """
        # Set graph parameters
        self.graph.set_parameters_from_tensor(
            state['param'].to(self.device), filter_requires_grad=FILTER_OFF)
        self.graph.set_integer_parameters_from_list(state['param_int'])

        # Set the optimization worker state
        self._set_worker_state(state['optim'])

        # Set the current best solution
        to_device = lambda x: x.to(self.device) if isinstance(x, th.Tensor) else x
        self.loss_min = state['loss_min']
        self.param_min = to_device(state['param_min'])

    def _obj_func(self, params: Any, save_interval: int = 20, update_interval: int = 0,
                  save_state_callback: SaveStateCallback = lambda *_: ...,
                  save_image_callback: SaveImageCallback = lambda *_: ...) -> float:
        """The default objective function to be used in black-box optimizers. The function handles
        checkpointing and history recording by itself.
        """
        graph, loss_func = self.graph, self.loss_func

        # Load the source parameters into the graph
        self._set_parameters(params)

        # Evaluate the loss function using loaded parameters
        with graph.timer() as timer_forward:
            maps = graph.evaluate_maps()
            render = graph.renderer(*maps)
            loss = loss_func(render).item()

        it = self.eval_counter
        self.eval_counter = it + 1

        # Update running minimal loss
        if update_interval > 0 and not (it + 1) % update_interval and loss < self.loss_min:
            self.loss_min = loss
            self.param_min = self._get_parameters()

        # Save the state with output images
        if save_interval > 0 and it and not (it + 1) % save_interval:
            save_state_callback(f'iter_{it}', it)
            save_image_callback(f'iter_{it}', (render, *maps))

        # Log loss and time consumption
        t_forward = timer_forward.elapsed
        self.history.append([it, loss, t_forward])
        self.logger.info(f'Iter {it}: loss = {loss:.6f}  forward time - {t_forward:.3f} ms')

        return loss

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Tuple[List[int], float, Any]:
        """Run the integer optimization function and return the optimal parameter set.
        """
        ...


class Backend(BaseBackend):
    """Base class for all backend algorithms in continuous parameter optimization.
    """
    def __init__(self, graph: MaterialGraph, loss_func: BaseMetric,
                 param_io_kwargs: Dict[str, Any] = {}, lr: float = 5e-4, **kwargs):
        """Initialize the backend.
        """
        super().__init__(graph, loss_func, param_io_kwargs=param_io_kwargs, **kwargs)

        # Function
        self._params_config = partial(graph.get_parameters_as_config, **param_io_kwargs)

        # Initialize the Adam optimizer
        self._parameters = lambda: graph.parameters(**param_io_kwargs)
        self.optimizer = Adam(self._parameters(), lr=lr)

        # Update the history construction function
        default_columns = ['iter', 'loss', 't_forward', 't_backward']
        self._make_history: Callable[..., pd.DataFrame] = \
            lambda: pd.DataFrame(self.history, columns=default_columns)

    def _get_parameters(self) -> th.Tensor:
        """Read optimizable parameters.
        """
        return self.graph.get_parameters_as_tensor(**self.param_io_kwargs)

    def _set_parameters(self, params: th.Tensor):
        """Assign values to optimizable parameters.
        """
        self.graph.set_parameters_from_tensor(params, **self.param_io_kwargs)

    def _num_parameters(self) -> int:
        """Count the number of optimizable parameters.
        """
        return self.graph.num_parameters(**self.param_io_kwargs)

    def reset(self):
        """Reset the backend to its initial state.
        """
        super().reset()

        # Reset the Adam optimizer
        lr = self.optimizer.defaults['lr']
        self.optimizer = Adam(self._parameters(), lr=lr)

    def _get_worker_state(self) -> Dict[str, Any]:
        """Get the optimization worker's current state. Returns an empty dictionary by default if
        no information needs to be recorded.
        """
        return self.optimizer.state_dict()

    def _set_worker_state(self, state: Dict[str, Any]):
        """Set the optimization worker's current state. Does nothing by default.
        """
        self.optimizer.load_state_dict(state)

    def evaluate(self, num_iters: int = 1000, start_iter: int = 0,
                 save_interval: int = 20, update_interval: int = 10,
                 save_state_callback: SaveStateCallback = lambda *_: ...,
                 save_image_callback: SaveImageCallback = lambda *_: ...) -> \
                    Tuple[th.Tensor, float, pd.DataFrame]:
        """Run gradient-based optimization with Adam.
        """
        graph, optimizer, loss_func = self.graph, self.optimizer, self.loss_func
        timer, history, logger = self.timer, self.history, self.logger

        # Clear the optimization history
        history.clear()

        def forward(params: Optional[th.Tensor] = None) -> Tuple[th.Tensor, ...]:
            """Forward evaluation function.
            """
            # Set parameters
            if params is not None:
                self._set_parameters(params)

            # Evaluate the graph and the loss function
            maps = graph.evaluate_maps()
            render = graph.renderer(*maps)
            loss = loss_func(render)

            return (loss, render, *maps)

        def log(it: int, loss_val: float, t_forward: float, t_backward: float):
            """Log run-time metrics to history and on the screen.
            """
            history.append([it, loss_val, t_forward, t_backward])
            logger.info(f'Iter {it}: loss = {loss_val:.6f}  '
                        f'forward time - {t_forward:.3f} ms  '
                        f'backward time - {t_backward:.3f} ms')

        # Optimization loop
        for it in range(start_iter, num_iters):

            # Forward evaluation
            optimizer.zero_grad()
            with timer() as timer_forward:
                loss, render, *maps = forward()

            # Update running minimal loss
            loss_val = loss.detach().item()
            if (it == num_iters - 1 or update_interval > 0 and not (it + 1) % update_interval) \
               and loss_val < self.loss_min:
                self.loss_min = loss_val
                self.param_min = self._get_parameters()

            # Backward evaluation
            with timer() as timer_backward:
                loss.backward()
            optimizer.step()

            # Save the state with output images
            if save_interval > 0 and it and not (it + 1) % save_interval:
                save_state_callback(f'iter_{it}', it)
                save_image_callback(f'iter_{it}', (render, *maps))

            # Log loss and time consumption
            log(it, loss_val, timer_forward.elapsed, timer_backward.elapsed)

        # Compile the optimization history into a data table
        return self.param_min, self.loss_min, self._make_history()


class IntegerBackend(BaseBackend):
    """Base class for all backend algorithms in discrete parameter optimization.
    """
    def __init__(self, graph: MaterialGraph, loss_func: BaseMetric,
                 param_io_kwargs: Dict[str, Any] = {}, **kwargs):
        """Initialize the backend.
        """
        super().__init__(graph, loss_func, param_io_kwargs=param_io_kwargs, **kwargs)

        # Initialize the design space (parameter specifications)
        self._init_design_space()

    def _get_parameters(self) -> List[int]:
        """Read optimizable parameters.
        """
        return self.graph.get_integer_parameters_as_list(**self.param_io_kwargs)

    def _set_parameters(self, params: List[int]):
        """Assign values to optimizable parameters.
        """
        self.graph.set_integer_parameters_from_list(params, **self.param_io_kwargs)

    def _num_parameters(self) -> int:
        """Count the number of optimizable parameters.
        """
        return self.graph.num_integer_parameters(**self.param_io_kwargs)

    def _int_param_config(self) -> ParamConfig:
        """Integer parameter configuration for constructing the design space.
        """
        return self.graph.get_integer_parameters_as_config(**self.param_io_kwargs)

    def _int_param_info(self) -> Iterator[Tuple[int, int, int]]:
        """An iterator over each integer parameter of the graph. Returns parameter information
        such as initial values and ranges.
        """
        # Extract parameter bounds to specify the design space
        int_param_config = self._int_param_config()

        for pc in itertools.chain(*(nc.values() for nc in int_param_config.values())):
            val, low, high = pc['value'], pc['low'], pc['high']

            # Parameter is an integer scalar
            if isinstance(val, int):
                yield val, low, high

            # Parameter is an integer vector
            else:
                low = [low] * len(val) if isinstance(low, int) else low
                high = [high] * len(val) if isinstance(high, int) else high
                for v, vl, vh in zip(val, low, high):
                    yield v, vl, vh

    @abstractmethod
    def _init_design_space(self):
        """Initialize the design space for the optimizer.
        """
        ...


class HybridBackend(IntegerBackend):
    """Base class for backend algorithms in hybrid (both discrete and continuous) parameter
    optimization.
    """
    def __init__(self, graph: MaterialGraph, loss_func: BaseMetric,
                 param_io_kwargs: Dict[str, Dict[str, Any]] = {}, **kwargs):
        """Initialize the hybrid backend.
        """
        # Remove the filter on optimizable values
        if 'continuous' in param_io_kwargs:
            param_io_kwargs['continuous']['filter_requires_grad'] = FILTER_OFF

        # Helper function for splitting filters
        self._split_kwargs = \
            lambda: tuple(param_io_kwargs.get(key, {}) for key in ('continuous', 'integer'))

        super().__init__(graph, loss_func, param_io_kwargs=param_io_kwargs, **kwargs)

    def _get_parameters(self) -> Tuple[th.Tensor, List[int]]:
        """Read optimizable parameters.
        """
        graph, (kwargs, kwargs_int) = self.graph, self._split_kwargs()

        # Read continuous and integer parameters separately
        params = graph.get_parameters_as_tensor(**kwargs) if kwargs else \
                 th.empty(0, device=self.device)
        params_int = graph.get_integer_parameters_as_list(**kwargs_int) if kwargs_int else []

        return params, params_int

    def _set_parameters(self, params: Tuple[th.Tensor, List[int]]):
        """Assign values to optimizable parameters.
        """
        (params, params_int), graph = params, self.graph
        kwargs, kwargs_int = self._split_kwargs()

        # Set continuous and discrete parameters separately
        if kwargs:
            graph.set_parameters_from_tensor(params, **kwargs)
        if kwargs_int:
            graph.set_integer_parameters_from_list(params_int, **kwargs_int)

    def _num_parameters(self) -> int:
        """Count the number of optimizable parameters.
        """
        graph, (kwargs, kwargs_int) = self.graph, self._split_kwargs()

        # Count continuous and discrete parameters separately
        num_params = graph.num_parameters(**kwargs) if kwargs else 0
        num_params += graph.num_integer_parameters(**kwargs_int) if kwargs_int else 0

        return num_params

    def _int_param_config(self) -> ParamConfig:
        """Integer parameter configuration for constructing the design space.
        """
        _, kwargs_int = self._split_kwargs()
        return self.graph.get_integer_parameters_as_config(**kwargs_int) if kwargs_int else {}


class AxBO(IntegerBackend):
    """Bayesian Optimization for integer parameter optimization using the `Ax` library.
    """
    def __init__(self, graph: MaterialGraph, loss_func: BaseMetric, **kwargs):
        """Initialize the Ax BO backend.
        """
        super().__init__(graph, loss_func, **kwargs)

    def _init_design_space(self):
        """Initialize optimizable parameter configurations.
        """
        design_space = [dict(name=f'x{i}', type='range', bounds=[vl, vh], value_type='int')
                        for i, (_, vl, vh) in enumerate(self._int_param_info())]
        self.design_space = design_space

    def _obj_func(self, params: Dict[str, int], **kwargs) -> Dict[str, Tuple[float, float]]:
        """Define the objective function to be used in the optimizer.
        """
        loss = super()._obj_func(list(params.values()), **kwargs)
        return {'loss': (loss, 0.0)}

    def evaluate(self, num_iters: int = 1000, start_iter: int = 0,
                 save_interval: int = 20, update_interval: int = 10,
                 save_state_callback: SaveStateCallback = lambda *_: ...,
                 save_image_callback: SaveImageCallback = lambda *_: ..., **kwargs) -> \
                    Tuple[List[int], float, pd.DataFrame, Any]:
        """Run Bayesian Optimization using a fixed number of evaluations.
        """
        del start_iter, update_interval

        # Reset the number of evaluations
        self.eval_counter = 0

        # Wrap the object function
        obj_func = partial(
            self._obj_func, save_interval=save_interval, save_state_callback=save_state_callback,
            save_image_callback=save_image_callback)

        # Start BO
        from ax import optimize
        best_params, (best_loss, _), *ret = optimize(
            self.design_space, obj_func, experiment_name='integer_opt', objective_name='loss',
            minimize=True, total_trials=num_iters, **kwargs)

        return (best_params.values(), best_loss['loss'], self._make_history(), *ret)


class SkoptBO(IntegerBackend):
    """Bayesian Optimization for integer parameter optimization using the `scikit-optimize`
    library.
    """
    def __init__(self, graph: MaterialGraph, loss_func: BaseMetric, **kwargs):
        """Initialize the skopt BO backend.
        """
        super().__init__(graph, loss_func, **kwargs)

    def _init_design_space(self):
        """Initialize optimizable parameter configurations.
        """
        from skopt.space import Integer
        design_space = [Integer(vl, vh, name=f'x{i}')
                        for i, (_, vl, vh) in enumerate(self._int_param_info())]
        self.design_space = design_space

    def evaluate(self, num_iters: int = 1000, start_iter: int = 0,
                 save_interval: int = 20, update_interval: int = 10,
                 save_state_callback: SaveStateCallback = lambda *_: ...,
                 save_image_callback: SaveImageCallback = lambda *_: ..., **kwargs) -> \
                    Tuple[List[int], float, pd.DataFrame, Any]:
        """Run Bayesian Optimization using a fixed number of evaluations.
        """
        del start_iter, update_interval

        # Reset the number of evaluations
        self.eval_counter = 0

        # Wrap the object function
        obj_func = partial(
            self._obj_func, save_interval=save_interval, save_state_callback=save_state_callback,
            save_image_callback=save_image_callback)

        # Start BO
        from skopt import gp_minimize
        res = gp_minimize(
            obj_func, self.design_space, n_calls=num_iters,
            n_initial_points=max(10, len(self.design_space) * 2), noise=1e-10, **kwargs)

        return res.x, res.fun, self._make_history(), res


class SimAnneal(HybridBackend):
    """Simulated annealing for hybrid parameter optimization using the `simanneal` library.
    """
    def __init__(self, graph: MaterialGraph, loss_func: BaseMetric, seed: int = -1,
                 T_max: float = 5e-3, T_min: float = 1e-6, max_steps: int = 10000,
                 pt_prob: float = 0.1, pt_min: float = -0.05, pt_max: float = 0.05, **kwargs):
        """Initialize the simulated annealing backend.
        """
        super().__init__(graph, loss_func, **kwargs)

        # Save hyperparameters of the simulated annealing algorithm
        self.T_max, self.T_min, self.max_steps = T_max, T_min, max_steps
        self.pt_prob, self.pt_min, self.pt_max = pt_prob, pt_min, pt_max

        # Create a NumPy random number generator for parameter perturbation
        self.rng = default_rng(seed if seed >= 0 else None)

        # Record the initial random number generator state without disrupting the default random
        # number stream
        rng_state_py = random.getstate()
        random.seed(seed if seed >= 0 else None)

        self.init_worker_state = self._get_worker_state()

        random.setstate(rng_state_py)

        # The current random number generator state
        self.worker_state = self.init_worker_state

    def reset(self):
        """Reset the simulated annealing algorithm to its initial state.
        """
        # Reset graph parameters and the best solution
        super().reset()

        # Reset the current random number generator state
        self.rng.bit_generator.state = self.init_worker_state['rng']
        self.worker_state = self.init_worker_state

    def _get_worker_state(self) -> Dict[str, Any]:
        """Get the current state of the simulated annealing algorithm.
        """
        self.worker_state = {'rng': self.rng.bit_generator.state, 'worker': random.getstate()}
        return self.worker_state

    def _set_worker_state(self, state: Dict[str, Any]):
        """Set the current state of the simulated annealing algorithm.
        """
        self.rng.bit_generator.state = state['rng']
        self.worker_state = state

    def _init_design_space(self):
        """Initialize integer parameter ranges.
        """
        # Extract parameter ranges for integer parameters
        int_param_ranges = \
            np.array([[vl, vh] for _, vl, vh in self._int_param_info()], dtype=np.float32)
        if len(int_param_ranges):
            self.int_param_ranges = (int_param_ranges[:, 0], int_param_ranges[:, 1])
        else:
            self.int_param_ranges = (int_param_ranges, int_param_ranges)

        # Converter functions between graph parameters and the worker's internal representation
        def to_params(sa_state: SAState) -> Tuple[th.Tensor, List[int]]:
            I = len(self.int_param_ranges[0])
            return self._t(sa_state[I:]), sa_state[:I].astype(int).tolist()

        def to_sa_state(params: Tuple[th.Tensor, List[int]]) -> SAState:
            params, params_int = params
            params_np = to_numpy(params)
            params_int_np = np.array(params_int, dtype=np.float32)
            return np.concatenate((params_int_np, params_np))

        self._to_params = to_params
        self._to_sa_state = to_sa_state

    def _worker_move(self, sa_state: SAState) -> SAState:
        """Randomly perturb the state of the simulated annealing worker.
        """
        rng = self.rng

        # Sample perturbation mask and magnitude
        size = len(sa_state)
        pt_mask = rng.uniform(size=size) <= max(self.pt_prob, 1.0 / size)
        pt_rands = rng.uniform(low=self.pt_min, high=self.pt_max, size=size)

        # Calculate integer parameter perturbations
        ipl, ipr = self.int_param_ranges
        I = len(ipl)
        pt_int = pt_rands[:I] * (ipr - ipl)
        pt_int = (np.sign(pt_int) * np.ceil(np.abs(pt_int))).astype(np.float32)

        # Update integer parameters and continuous parameters separately
        sa_state_update = np.empty_like(sa_state)
        sa_state_update[:I] = np.clip(sa_state[:I] + pt_int, ipl, ipr)
        sa_state_update[I:] = np.clip(sa_state[I:] + pt_rands[I:], 0.0, 1.0)

        return np.where(pt_mask, sa_state_update, sa_state)

    def _worker_energy(self, sa_state: SAState, **obj_kwargs) -> float:
        """Compute the loss function given a simulated annealing worker state.
        """
        return self._obj_func(self._to_params(sa_state), **obj_kwargs)

    def evaluate(self, num_iters: int = 10000, start_iter: int = 0,
                 save_interval: int = 500, update_interval: int = 100,
                 save_state_callback: SaveStateCallback = lambda *_: ...,
                 save_image_callback: SaveImageCallback = lambda *_: ...) -> \
                    Tuple[Tuple[th.Tensor, List[int]], float, pd.DataFrame]:
        """Run simulated annealing.
        """
        # Calculate starting and ending temperatures
        T_max, T_min, max_steps = self.T_max, self.T_min, self.max_steps
        T_factor = -math.log(T_max / T_min)
        T_start = T_max * math.exp(T_factor * start_iter / max_steps)
        T_end = T_max * math.exp(T_factor * (start_iter + num_iters) / max_steps)

        # Construct the simulated annealing worker using the current parameters
        init_state = self._to_sa_state(self._get_parameters())
        move_func = self._worker_move
        energy_func = partial(
            self._worker_energy, save_interval=save_interval,
            save_state_callback=save_state_callback, save_image_callback=save_image_callback)

        worker = SimAnnealWorker(init_state, move_func, energy_func)
        worker.Tmax, worker.Tmin, worker.steps = T_start, T_end, num_iters
        worker.updates = num_iters // update_interval

        # Start the worker after loading the local random number generator state
        rng_state_py = random.getstate()
        random.setstate(self.worker_state['worker'])

        param_min, loss_min = worker.anneal()
        self.param_min, self.loss_min = self._to_params(param_min), loss_min

        random.setstate(rng_state_py)

        return self.param_min, self.loss_min, self._make_history()


class GridSearch(HybridBackend):
    """Grid search for hybrid parameter optimization.
    """
    def __init__(self, graph: MaterialGraph, loss_func: BaseMetric,
                 param_io_kwargs: Dict[str, Dict[str, Any]] = {}, **kwargs):
        """Initialize the grid search backend.
        """
        super().__init__(graph, loss_func, param_io_kwargs=param_io_kwargs, **kwargs)

    def _init_design_space(self):
        """Initialize integer parameter ranges.
        """
        self.int_param_ranges = [(vl, vh) for _, vl, vh in self._int_param_info()]

    def evaluate(self, num_iters: int = 3, start_iter: int = 0,
                 save_interval: int = 0, update_interval: int = 0,
                 save_state_callback: SaveStateCallback = lambda *_: ...,
                 save_image_callback: SaveImageCallback = lambda *_: ...,
                 search_res: int = 100) -> Tuple[Tuple[th.Tensor, List[int]], float, Any]:
        """Run grid search.
        """
        del start_iter, save_interval, update_interval
        graph, loss_func = self.graph, self.loss_func
        logger, (kwargs, kwargs_int) = self.logger, self._split_kwargs()

        # Read the current parameters
        params, params_int = self._get_parameters()

        # Helper functions for optimizing parameter value in 1D
        def line_search(params: Union[th.Tensor, List[int]], index: int,
                        vals: Union[List[float], List[int]],
                        type: str = 'float') -> Tuple[float, float]:

            # Type-specific parameter setting function
            if type == 'float':
                set_parameters = partial(graph.set_parameters_from_tensor, **kwargs)
            else:
                set_parameters = partial(graph.set_integer_parameters_from_list, **kwargs_int)

            # Evaluating the loss function by varying the parameter value
            losses = np.zeros(len(vals))
            for i, val in enumerate(vals):
                params[index] = val
                set_parameters(params)
                losses[i] = loss_func(graph.evaluate()).detach().item()

            # Calculate the best parameter value
            min_i = losses.argmin()
            min_val = vals[min_i]
            params[index] = min_val
            set_parameters(params)

            return min_val, losses[min_i]

        # Run grid search for each parameter in turns
        for it in range(num_iters):

            # Integer parameters (search range: [vl, vh], step: 1)
            int_param_ranges = self.int_param_ranges

            for i in range(len(params_int)):
                vl, vh = int_param_ranges[i]
                min_val, min_loss = line_search(params_int, i, list(range(vl, vh + 1)), type='int')

                logger.info(f'Param (int) {i}: min value = {min_val}, loss = {min_loss:.6g}')

            # Continuous parameters (search range: [0, 1], step: 0.01)
            for i in range(len(params)):
                val_range = np.linspace(0, 1, search_res + 1).tolist()
                min_val, min_loss = line_search(params, i, val_range)

                logger.info(f'Param {i}: min value = {min_val:.3g}, loss = {min_loss:.6g}')

            # Save a checkpoint
            self._set_parameters((params, params_int))
            self.param_min, self.loss_min = (params, params_int), min_loss

            maps = graph.evaluate_maps()
            save_state_callback(f'iter_{it}', it)
            save_image_callback(f'iter_{it}', (graph.renderer(*maps), *maps))

        return self.param_min, self.loss_min, self._make_history()


def get_backend(type: str, name: str, *args, **kwargs) -> BaseBackend:
    """Create an integer optimization backend by name. Other arguments are passed to the
    constructor of the backend class.
    """
    if type not in BACKEND_DICT:
        raise ValueError(f'Unknown optimization backend type: {type}. Valid options are '
                         f'{list(BACKEND_DICT.keys())}')
    elif name not in BACKEND_DICT[type]:
        raise ValueError(f'Unknown optimization backend name: {name}. Valid options are '
                         f'{list(BACKEND_DICT[type].keys())}.')

    return BACKEND_DICT[type][name](*args, **kwargs)


# Dictionary of backends
BACKEND_DICT: Dict[str, Dict[str, Type[BaseBackend]]] = {
    'continuous': {'adam': Backend},
    'integer': {'bo-ax': AxBO, 'bo-skopt': SkoptBO},
    'hybrid': {'simanneal': SimAnneal, 'grid': GridSearch},
}


# Helper/worker functions and classes
class SimAnnealWorker(Annealer):
    """Simulated annealing for continuous parameter optimization using the `simanneal` library.
    """
    copy_strategy = 'method'

    def __init__(self, initial_state: SAState, move_func: Callable[[SAState], SAState],
                 energy_func: Callable[[SAState], float], **kwargs):
        """Initialize the simulated annealing backend.
        """
        super().__init__(initial_state=initial_state, **kwargs)

        self.move_func = move_func
        self.energy_func = energy_func

    def move(self):
        """Perturb continuous parameters.
        """
        self.state[:] = self.move_func(self.state)

    def energy(self) -> float:
        """Calculate image loss.
        """
        return self.energy_func(self.state)
