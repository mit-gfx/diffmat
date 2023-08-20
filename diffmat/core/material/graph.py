from typing import List, Dict, Tuple, Any, Union, Optional, Iterator, Callable
from operator import itemgetter
from collections import deque
import itertools

import torch as th
import yaml

from diffmat.core.base import BaseParameter, BaseGraph
from diffmat.core.types import Instruction, GraphSummary, ParamConfig, IntParamValue, PathLike
from diffmat.core.util import FILTER_OFF, FILTER_NO, FILTER_YES
from .base import BaseMaterialNode
from .functional import resize_image_color
from .node import ExternalInputNode
from .render import Renderer
from .util import (
    get_parameters, get_parameters_as_config, set_parameters_from_config, get_integer_parameters,
    set_integer_parameters_from_list, get_integer_parameters_as_config,
    set_integer_parameters_from_config
)


class MaterialGraph(BaseGraph[BaseMaterialNode]):
    """Differentiable material graph class.
    """
    def __init__(self, nodes: List[BaseMaterialNode], name: str, res: int,
                 external_inputs: Dict[str, th.Tensor] = {},
                 exposed_params: List[BaseParameter] = [],
                 render_params: Dict[str, Any] = {}, use_alpha: bool = True,
                 seed: int = 0, **kwargs):
        """Initialize a material graph.

        Args:
            nodes (List[BaseMaterialNode]): List of material graph nodes.
            name (str): Graph name.
            res (int): Output texture resolution (after log2).
            external_inputs (Dict[str, Tensor], optional): Dictionary of external input images
                such as dependent files or imported noise textures. Defaults to {}.
            exposed_params (List[BaseParameter], optional): List of exposed parameters.
                Defaults to [].
            render_params (Dict[str, Any], optional): Parameters for rendering SVBRDF maps into
                a synthetic image (e.g., lighting, camera distance, etc.). Defaults to {}.
            use_alpha (bool, optional): Enable alpha channel processing in the graph.
                Defaults to True.
            seed (int, optional): Graph-wide random seed. Defaults to 0.
            kwargs (Dict[str, Any]): Keyword arguments to pass into the parent class constructor.
        """
        super().__init__(nodes, **kwargs)

        self.name = name
        self.res = res
        self.external_inputs = external_inputs
        self.exposed_params = exposed_params
        self.renderer = Renderer(**render_params, device=self.device)
        self.use_alpha = use_alpha
        self.seed = seed

        # Link the parent of exposed parameters to this graph
        self.link_as_parent(*self.exposed_params)

        # List of instructions for evaluating the graph
        self.program: List[Instruction] = []

        # Runtime memory of the material graph (stores intermediate outputs)
        self.memory: Dict[str, th.Tensor] = {}

        # Topologically sort the nodes
        self._sort_nodes()

        # Place output nodes at the end of the sorted sequence
        orders = {channel: i for i, channel in enumerate(Renderer.CHANNELS.keys())}
        self.nodes.sort(key=lambda node: orders.get(node.name, 0))

    def compile(self):
        """Compile the material graph into a program that contains a sequence of instructions for
        evaluation. Each node is translated into an instruction that comprises the following info:
            - op: node/operation name;
            - args: name(s) of input texture maps to read from runtime memory;
            - result: name(s) of output texture maps to write back to runtime memory.
        """
        # Compile each individual node (perform static type checking)
        exposed_param_levels = {p.name: p.output_level for p in self.exposed_params}

        for node in self.nodes:
            node.compile(exposed_param_levels, master_seed=self.seed)

        # Helper function for generating the variable name associated with a node output
        get_variable_name: Callable[[str, str], str] = \
            lambda name, output: f'{name}_{output}' if output else name

        # Translate node sequence into program sequence
        # It is assumed that the I/O order in node configuration files is consistent with that in
        # the node function. Also, the node.outputs dictionary read from YAML must preserve the
        # said I/O order
        self.program.clear()

        for node in self.nodes:
            op_name = node.name
            op_args = [get_variable_name(*val) if val else None for val in node.inputs.values()]
            op_result = [get_variable_name(op_name, key) for key in node.outputs]
            self.program.append({'op': op_name, 'args': op_args, 'result': op_result})

        # Print the program
        program_str = [f'Compiled material graph program ({len(self.program)} nodes):']
        for inst in self.program:
            program_str.append(
                f"  {inst['op']}: ({', '.join(str(s) for s in inst['args'])}) -> "
                f"{', '.join(inst['result'])}")
        self.logger.info('\n'.join(program_str))

    def evaluate_maps(self, benchmarking: bool = False) -> Tuple[th.Tensor, ...]:
        """Evaluate the compiled program of the material graph.

        Args:
            benchmarking (bool, optional): Whether to benchmark the execution time of each node.

        Raises:
            RuntimeError: The material graph has not be compiled into an executable program.

        Returns:
            Tuple[Tensor, ...]: Sequence of output SVBRDF maps (the order is specified by the keys
                of `Renderer.CHANNELS` in `render.py`) in the form of torch tensors.
        """
        # Check if the material graph has been compiled
        if not self.program:
            raise RuntimeError("The material graph has not been compiled. Please invoke the "
                               "'compile' method first.")

        # Evaluate exposed parameters
        # Promote 0D tensors in the variable dictionary to 1D for function graph evaluation
        exposed_params = {p.name: p.evaluate() for p in self.exposed_params}
        exposed_params = {key: th.atleast_1d(val) if isinstance(val, th.Tensor) else val \
                          for key, val in exposed_params.items()}

        # Clear runtime memory
        memory = self.memory
        memory.clear()

        # Build a node dictionary indexed by name
        node_dict = {node.name: node for node in self.nodes}

        # Initialize per-node timing dictionary for benchmarking mode
        if benchmarking:
            node_t_forward: Dict[str, float] = {}
            node_t_backward: Dict[str, float] = {}

        # Execute the program
        global_options = {'use_alpha': self.use_alpha}

        ## Change default torch device to the device where the graph is placed
        is_on_cuda = self.device.type == 'cuda'
        is_cuda_default = th.empty([]).is_cuda
        if is_on_cuda != is_cuda_default:
            th.set_default_tensor_type(th.cuda.FloatTensor if is_on_cuda else th.FloatTensor)

        for inst in self.program:

            # Evaluate the current node and convert the result to a tuple
            node = node_dict[inst['op']]

            ## External input node
            if isinstance(node, ExternalInputNode):
                result = node.evaluate(self.external_inputs, **global_options)

            ## Regular material nodes
            else:
                args = [memory.get(arg_name) for arg_name in inst['args']]

                # Non-benchmarking mode: normal execution
                if not benchmarking:
                    result = node.evaluate(*args, exposed_params=exposed_params, **global_options)

                # Benchmarking mode: time forward and backward pass independently
                else:
                    result, node_t_forward[node.name] = node.benchmark_forward(
                        *args, exposed_params=exposed_params, **global_options)
                    _, node_t_backward[node.name] = node.benchmark_backward(
                        *args, exposed_params=exposed_params, **global_options)

            result = (result,) if not isinstance(result, (list, tuple)) else result

            # Store output tensors into memory
            memory.update({key: val for key, val in zip(inst['result'], result) if key})

        # Read output SVBRDF maps
        img_size = 1 << self.res
        outputs: List[th.Tensor] = []

        for channel, (is_color, default_val) in Renderer.CHANNELS.items():
            n_channels = 1 if not is_color else 4 if self.use_alpha else 3

            # Read from runtime memory or construct default output maps
            if channel in memory:
                img = resize_image_color(memory[channel], n_channels)
            else:
                img = th.full((1, n_channels, img_size, img_size), default_val, device=self.device)
                if channel == 'normal':
                    img[:, :2] = img.narrow(1, 0, 2) * 0.5
                memory[channel] = img

            outputs.append(img)

        # Reset default device to the previous setting
        if is_on_cuda != is_cuda_default:
            th.set_default_tensor_type(th.cuda.FloatTensor if is_cuda_default else th.FloatTensor)

        # Print benchmarking result, including the slowest nodes in forward and backward
        if benchmarking:
            result_length = 10
            slowest_forward = list(
                sorted(node_t_forward.items(), key=itemgetter(1), reverse=True))[:result_length]
            slowest_backward = list(
                sorted(node_t_backward.items(), key=itemgetter(1), reverse=True))[:result_length]

            # Format result in strings
            str_slowest_forward = ', '.join(
                f'{k} ({v * 1e3:.3f} ms)' for k, v in slowest_forward)
            str_slowest_backward = ', '.join(
                f'{k} ({v * 1e3:.3f} ms)' for k, v in slowest_backward)

            self.logger.info(f'Slowest nodes (forward): {str_slowest_forward}')
            self.logger.info(f'Slowest nodes (backward): {str_slowest_backward}')

        return tuple(outputs)

    def evaluate(self, benchmarking: bool = False) -> th.Tensor:
        """Evaluate the compiled program of the material graph and generate a rendered image of the
        resulting texture.

        This method chains a call to the `evaluate_maps` method and the differentiable render.

        Args:
            benchmarking (bool, optional): Whether to benchmark the execution time of each node.

        Returns:
            Tensor: Rendering of output SVBRDF maps from the differentiable procedural material
                graph.
        """
        return self.renderer(*self.evaluate_maps(benchmarking=benchmarking))

    def train(self, ablation_mode: str = 'none'):
        """Set the material graph to training state, which sets all optimizable parameters to
        require gradient.

        Args:
            ablation_mode (str, optional): Option for excluding some nodes from node parameter
                optimization. This option is useful for ablation studies. Valid options are:
                    `none`: no ablation;
                    `node`: ablate nodes that allow ablation;
                    `subgraph`: ablate predecessor subgraphs of nodes that allow ablation.
                Defaults to 'none'.

        Raises:
            ValueError: Invalid ablation mode.
        """
        if ablation_mode not in ('none', 'node', 'subgraph'):
            raise ValueError(f'Invalid ablation mode: {ablation_mode}')

        # Set exposed parameters to require gradient
        for param in self.parameters(filter_exposed=FILTER_YES, filter_requires_grad=FILTER_NO):
            param.requires_grad_(True)

        # Set node parameters to require gradient
        ## No ablation
        if ablation_mode == 'none':
            for node in self.nodes:
                node.train()

        ## Ablate nodes that allow ablation
        elif ablation_mode == 'node':
            for node in self.nodes:
                node.eval() if node.allow_ablation else node.train()

        ## Ablate all predecessor nodes of nodes that allow ablation
        else:
            node_dict = {node.name: node for node in self.nodes}
            ablation_dict = {node.name: node.allow_ablation for node in self.nodes}

            # Run BFS
            queue = deque(node for node in self.nodes if node.allow_ablation)
            while queue:
                node = queue.popleft()
                for node_pred in (pair[0] for pair in node.inputs.values() if pair is not None):
                    if not ablation_dict[node_pred]:
                        ablation_dict[node_pred] = True
                        queue.append(node_dict[node_pred])

            for name, node in node_dict.items():
                node.eval() if ablation_dict[name] else node.train()

    def eval(self):
        """Set the material graph to evaluation state, which clears the `requires_grad` attribute
        of all optimizable parameters.
        """
        # Set exposed parameters not to require gradient
        for param in self.parameters(filter_exposed=FILTER_YES):
            param.requires_grad_(False)

        # Set node parameters not to require gradient
        for node in self.nodes:
            node.eval()

    def parameters(self, filter_exposed: int = FILTER_OFF, filter_generator: int = FILTER_OFF,
                   filter_requires_grad: int = FILTER_YES, detach: bool = False,
                   flatten: bool = False) -> Iterator[th.Tensor]:
        """An iterator over optimizable parameter values in the material graph (tensor views rather
        than copies). When called with default arguments, the returned iterator can be the input to
        PyTorch optimizers (e.g., Adam).

        Args:
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
            filter_requires_grad (int, optional): Option for filtering out parameters that require
                gradient. Valid cases are:
                    `1 = yes` means parameters that require gradient are returned;
                    `0 = no` means parameters that don't require gradient are returned;
                    `-1 = off` means all parameters are returned.
                Defaults to `yes`.
            detach (bool, optional): Whether returned tensor views are detached (i.e., don't
                require gradient). Defaults to False.
            flatten (bool, optional): Whether returned tensor views are flattened.
                Defaults to False.

        Yields:
            Iterator[Tensor]: Tensor views of optimizable node parameter values.
        """
        kwargs = {
            'filter_requires_grad': filter_requires_grad,
            'detach': detach, 'flatten': flatten
        }

        # Collect parameter values from optimizable parameters
        if filter_exposed != FILTER_NO:
            yield from get_parameters(self.exposed_params, **kwargs)

        # Return other node parameters
        if filter_exposed != FILTER_YES:
            for node in self.nodes:
                yield from node.parameters(**kwargs, filter_generator=filter_generator)

    def num_parameters(self, **kwargs) -> int:
        """Count the number of optimizable parameter values (floating-point numbers) in the
        material graph.

        Args:
            kwargs (Dict[str, Any], optional): Keyword arguments to be passed into the `parameters`
                method.

        Returns:
            int: Aggregated number of optimizable parameter values (elements).
        """
        return sum(view.shape[0] for view in
                   self.parameters(**kwargs, detach=True, flatten=True))

    def get_parameters_as_tensor(self, detach: bool = True, **kwargs) -> Optional[th.Tensor]:
        """Get the values of optimizable parameters of the material graph as a 1D torch tensor.
        Returns None if there is no optimizable parameters in the graph.

        Args:
            detach (bool, optional): Whether the returned tensor is detached (i.e., doesn't require
                gradient). Defaults to True.
            kwargs (Dict[str, Any], optional): Keyword arguments to be passed into the `parameters`
                method.

        Returns:
            Optional[Tensor]: Flattened concatenation of optimizable parameters in the graph,
                or None if the graph doesn't have optimizable parameters.
        """
        # Pack the flattened views of optimizable parameters into a 1D vector
        param_views = list(self.parameters(**kwargs, detach=detach, flatten=True))
        param_vec = th.cat(param_views) if param_views else None

        return param_vec

    def set_parameters_from_tensor(self, values: th.Tensor, **kwargs):
        """Set the optimizable parameters of the material graph from a 1D torch tensor.

        Args:
            values (tensor, optional): Source parameter values (must be 1D tensor).
            kwargs (Dict[str, Any], optional): Keyword arguments to be passed into the `parameters`
                method.

        Raises:
            ValueError: The input is not a tensor or doesn't have a 1D shape.
            RuntimeError: This material graph doesn't have optimizable parameters.
            RuntimeError: The size of the input tensor does not match the number of optimizable
                parameters in the graph.
        """
        # Check if the input is a 1D torch tensor
        if not isinstance(values, th.Tensor) or values.ndim != 1:
            raise ValueError('The input must be a 1D torch tensor.')

        values = values.detach()

        # Obtain flattened views of optimizable parameter tensors
        param_views = list(self.parameters(**kwargs, detach=True, flatten=True))
        if not param_views:
            raise RuntimeError('This material graph does not have optimizable parameters')

        # Check if the number of parameters provided match the number that this node has
        num_params = [view.shape[0] for view in param_views]
        if sum(num_params) != values.shape[0]:
            raise RuntimeError(f'The size of the input tensor ({values.shape[0]}) does not match '
                               f'the optimizable parameters ({sum(num_params)}) in this graph')

        # Update the parameter values
        pos = 0
        for view, size in zip(param_views, num_params):
            view.copy_(values.narrow(0, pos, size))
            pos += size

    def get_parameters_as_config(self, filter_exposed: int = FILTER_OFF,
                                 filter_generator: int = FILTER_OFF,
                                 constant: bool = False) -> ParamConfig:
        """Return node parameters of the material graph as a nested dict-type configuration in the
        following format:
        ```yaml
        exposed:
          {exposed_param_name}: # x many
            value: {exposed_param_value}
            normalize: False/True # optional for optimizable parameters
        {node_name}: # x many
          {param_name}: # x many
            value: {param_value}
            normalize: False/True
        ```

        Args:
            filter_exposed (int, optional): See `parameters` method for details. Defaults to
                `-1 = all`.
            filter_generator (int, optional): See `parameters` method for details. Defaults to
                `-1 = off`.
            constant (bool, optional): Whether to convert parameter values to literals (float, int,
                or bool-typed constants). Defaults to False.

        Returns:
            ParamConfig: Parameter configuration as outlined above.
        """
        # Collect parameter configs from exposed parameters
        config = {'exposed': get_parameters_as_config(self.exposed_params, constant=constant)} \
                 if filter_exposed != FILTER_NO else {}

        # Collect parameter configs from material nodes
        if filter_exposed != FILTER_YES:
            kwargs = {'filter_generator': filter_generator, 'constant': constant}
            config.update({n.name: n.get_parameters_as_config(**kwargs) for n in self.nodes})

        return {k: v for k, v in config.items() if v}

    def set_parameters_from_config(self, config: ParamConfig):
        """Set node parameters of the material graph from a nested dict-type configuration in the
        following format:
        ```yaml
        exposed:
          {exposed_param_name}: # x many
            value: {exposed_param_value}
            normalize: False/True # optional for optimizable parameters
        {node_name}: # x many
          {param_name}: # x many
            value: {param_value}
            normalize: False/True
        ```

        Args:
            config (ParamConfig): Parameter configuration as outlined above.
        """
        # Set exposed parameter values by configuration
        config = config.copy()
        set_parameters_from_config(self.exposed_params, config.pop('exposed', {}))

        # Set parameter values in each node by configuration
        node_dict = {n.name: n for n in self.nodes}

        for node_name, node_param_config in config.items():
            node_dict[node_name].set_parameters_from_config(node_param_config)

    def integer_parameters(self, filter_exposed: int = FILTER_OFF,
                           filter_generator: int = FILTER_OFF) -> Iterator[IntParamValue]:
        """An iterator that traverses all optimizable integer parameters in a material graph.

        Args:
            filter_exposed (int, optional): See `parameters` method for details. Defaults to
                `-1 = all`.
            filter_generator (int, optional): See `parameters` method for details. Defaults to
                `-1 = off`.

        Yields:
            Iterator[IntParamValue]: Optimizable integer parameter values.
        """
        # Collect exposed integer parameters
        if filter_exposed != FILTER_NO:
            yield from get_integer_parameters(self.exposed_params)

        # Collect integer parameters from each node
        if filter_exposed != FILTER_YES:
            for node in self.nodes:
                yield from node.integer_parameters(filter_generator=filter_generator)

    def num_integer_parameters(self, **kwargs) -> int:
        """Count the number of optimizable integer parameters in the material graph.

        Args:
            kwargs (Dict[str, Any], optional): Keyword arguments to be passed into the
                `integer_parameters` method.

        Returns:
            int: Aggregated number of optimizable integer parameters.
        """
        return sum(1 if isinstance(val, int) else len(val)
                   for val in self.integer_parameters(**kwargs))

    def get_integer_parameters_as_list(self, **kwargs) -> List[int]:
        """Get the values of optimizable integer parameters of the material graph as a list.

        Args:
            kwargs (Dict[str, Any], optional): Keyword arguments to be passed into the
                `integer_parameters` method.

        Returns:
            List[int]: List of optimizable integer parameter values.
        """
        # Pack the integer parameter values into a list
        param_list: List[int] = []
        for val in self.integer_parameters(**kwargs):
            param_list.append(val) if isinstance(val, int) else param_list.extend(val)

        return param_list

    def set_integer_parameters_from_list(self, values: List[int], filter_exposed: int = FILTER_OFF,
                                         **kwargs):
        """Set optimizable integer parameter values of the material graph from an integer list.

        Args:
            values (List[int]): Source parameter values.
            filter_exposed (int, optional): See `parameters` method for details. Defaults to
                `-1 = all`.
            kwargs (Dict[str, Any], optional): Keyword arguments to be passed into the
                `integer_parameters` method.

        Raises:
            ValueError: The length of the input list does not match the number of optimizable
                parameters in the graph.
        """
        # Make sure the input parameter list matches the numer of integer parameters
        num_params = self.num_integer_parameters(filter_exposed=filter_exposed, **kwargs)
        if len(values) != num_params:
            raise ValueError(f'The length of the input list ({len(values)}) does not match '
                             f'the optimizable parameters ({num_params}) in this graph')

        # Unflatten the parameter list and set parameter values for exposed parameters
        pos = 0

        if filter_exposed != FILTER_NO:
            num_exposed_params = self.num_integer_parameters(filter_exposed=FILTER_YES)
            if num_exposed_params:
                set_integer_parameters_from_list(self.exposed_params, values[:num_exposed_params])
                pos = num_exposed_params

        # Set parameter values for other integer node parameters
        if filter_exposed != FILTER_YES:
            for node in self.nodes:
                param_length = node.num_integer_parameters(**kwargs)
                if param_length:
                    node.set_integer_parameters_from_list(values[pos:pos+param_length])
                    pos += param_length

    def get_integer_parameters_as_config(self, filter_exposed: int = FILTER_OFF,
                                         filter_generator: int = FILTER_OFF) -> ParamConfig:
        """Return optimizable integer parameter values of the material graph as a dict-type
        configuration in the following format:
        ```yaml
        exposed:
          {exposed_param_name}: # x many
            value: {exposed_param_value}
            low: {exposed_param_low_bound}
            high: {exposed_param_high_bound}
        {node_name}: # x many
          {param_name}: # x many
            value: {param_value}
            low: {param_low_bound}
            high: {param_high_bound}
        ```

        Args:
            filter_exposed (int, optional): See `parameters` method for details. Defaults to
                `-1 = all`.
            filter_generator (int, optional): See `parameters` method for details. Defaults to
                `-1 = off`.

        Returns:
            ParamConfig: Parameter configuration as outlined above.
        """
        # Collect exposed parameter configuration
        config = {'exposed': get_integer_parameters_as_config(self.exposed_params)} \
                 if filter_exposed != FILTER_NO else {}

        # Collect parameter configs from material nodes
        if filter_exposed != FILTER_YES:
            config.update(
                {n.name: n.get_integer_parameters_as_config(filter_generator=filter_generator)
                 for n in self.nodes})

        return {k: v for k, v in config.items() if v}

    def set_integer_parameters_from_config(self, config: ParamConfig):
        """Set optimizable integer parameter values of the material graph from a nested dict-type
        configuration in the following format:
        ```yaml
        {node_name}: # x many
          {param_name}: # x many
            value: {param_value}
        ```

        Args:
            config (ParamConfig): Parameter configuration as outlined above.
        """
        # Set exposed parameter values by configuration
        config = config.copy()
        set_integer_parameters_from_config(self.exposed_params, config.pop('exposed', {}))

        # Set parameter values in each node by configuration
        node_dict = {n.name: n for n in self.nodes}

        for node_name, node_param_config in config.items():
            node_dict[node_name].set_integer_parameters_from_config(node_param_config)

    def load_parameters_from_file(self, file: PathLike):
        """Load continuous and integer parameter values of the material graph from an external file
        (in PyTorch checkpoint format).

        Args:
            file (PathLike): Path to the checkpoint file containing parameter values.
        """
        # Read continuous and integer parameters from file
        state_dict = th.load(file)
        init_params, init_params_int = state_dict.get('param'), state_dict.get('param_int')

        # Load parameters into the material graph
        if init_params is not None:
            self.set_parameters_from_tensor(self._t(init_params), filter_requires_grad=False)
        if init_params_int is not None:
            self.set_integer_parameters_from_list(init_params_int)

    def summarize(self, filename: str) -> GraphSummary:
        """Generate a summary of graph status containing nodes and parameters. The summary is
        returned and also saved into a local file in YAML format.

        Args:
            filename (str): Path to the saved summary file.

        Returns:
            GraphSummary: A dictionary that summarizes essential information of the graph,
                including name, summaries of graph nodes, and exposed parameters.
        """
        summary: GraphSummary = {
            'name': self.name,
            'nodes': {n['name']: {'input': n['input'], 'param': n['param']} \
                      for n in (node.summarize() for node in self.nodes)},
            'param': dict(tuple(p.summarize().values()) for p in self.exposed_params)
        }

        # Save the summary to a file
        with open(filename, 'w') as f:
            yaml.dump(summary, f, sort_keys=False)

        return summary

    def to_device(self, device: Union[str, th.device]):
        """Move the material graph to a specified device (e.g., CPU or GPU).

        Args:
            device (DeviceType, optional): Target device ID. Defaults to 'cpu'.
        """
        # Move data members to the target device
        for obj in itertools.chain(self.exposed_params, [self.renderer]):
            obj.to_device(device)
        for key, val in self.external_inputs.items():
            self.external_inputs[key] = val.to(device)

        # Move nodes and set the current device
        super().to_device(device)
