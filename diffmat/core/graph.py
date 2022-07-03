from typing import List, Dict, Tuple, Any, Union, Optional, Iterator, Callable
import itertools

import torch as th
import yaml

from .base import BaseGraph, BaseParameter
from .node import MaterialNode, ExternalInputNode
from .functional import resize_image_color
from .render import Renderer
from .types import Instruction, Constant, GraphSummary


class MaterialGraph(BaseGraph[MaterialNode]):
    """Differentiable material graph class.
    """
    def __init__(self, nodes: List[MaterialNode], name: str, res: int,
                 external_inputs: Dict[str, th.Tensor] = {},
                 exposed_params: List[BaseParameter] = [],
                 render_params: Dict[str, Any] = {}, use_alpha: bool = True,
                 seed: int = 0, **kwargs):
        """Initialize a material graph.

        Args:
            nodes (List[MaterialNode]): List of material graph nodes.
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

    def evaluate_maps(self) -> Tuple[th.Tensor, ...]:
        """Evaluate the compiled program of the material graph.

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
        # Promote 0D tensors in the variable dictionary to 1D
        exposed_params = {p.name: p.evaluate() for p in self.exposed_params}
        exposed_params = {key: th.atleast_1d(val) if isinstance(val, th.Tensor) else val \
                          for key, val in exposed_params.items()}

        # Clear runtime memory
        memory = self.memory
        memory.clear()

        # Build a node dictionary indexed by name
        node_dict = {node.name: node for node in self.nodes}

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
                result = node.evaluate(*args, exposed_params=exposed_params, **global_options)

            result = (result,) if isinstance(result, th.Tensor) else result

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

        return tuple(outputs)

    def evaluate(self) -> th.Tensor:
        """Evaluate the compiled program of the material graph and generate a rendered image of the
        resulting texture.

        This method chains a call to the `evaluate_maps` method and the differentiable render.

        Returns:
            Tensor: Rendering of output SVBRDF maps from the differentiable procedural material
                graph.
        """
        return self.renderer(*self.evaluate_maps())

    def train(self):
        """Set the material graph to training state, which sets all optimizable parameters to
        require gradient.
        """
        for param in self.parameters(level_exposed=1):
            param.requires_grad_(True)

        # Set node parameters to require gradient
        for node in self.nodes:
            node.train()

    def eval(self):
        """Set the material graph to evaluation state, which clears the `requires_grad` attribute
        of all optimizable parameters.
        """
        for param in self.parameters(level_exposed=1):
            param.requires_grad_(False)

        # Set node parameters not to require gradient
        for node in self.nodes:
            node.eval()

    def parameters(self, level_exposed: int = 2, detach: bool = False,
                   flatten: bool = False) -> Iterator[th.Tensor]:
        """An iterator over optimizable parameter values in the material graph (tensor views rather
        than copies). When called with default arguments, the returned iterator can be the input to
        PyTorch optimizers (e.g., Adam).

        Args:
            level_exposed (int, optional): Option for return some or all optimizable parameters
                in the graph.
                `2 = all`: all parameters will be returned;
                `1 = exclusive`: only exposed parameters are returned;
                `0 = complement`: only non-exposed parameters are returned.
                Defaults to 2.
            detach (bool, optional): Whether returned tensor views are detached (i.e., don't
                require gradient). Defaults to False.
            flatten (bool, optional): Whether returned tensor views are flattened.
                Defaults to False.

        Yields:
            Iterator[Tensor]: Tensor views of optimizable node parameter values.
        """
        # Collect parameter values from optimizable parameters
        if level_exposed >= 1:
            for param in (p for p in self.exposed_params if p.IS_OPTIMIZABLE):
                data: th.Tensor = param.data
                data = data.detach() if detach else data
                data = data.view(-1) if flatten else data
                yield data

        # Return other node parameters
        if level_exposed != 1:
            node_param_generators = \
                (node.parameters(detach=detach, flatten=flatten) for node in self.nodes)
            yield from itertools.chain(*node_param_generators)

    def num_parameters(self, level_exposed: int = 2) -> int:
        """Count the number of optimizable parameter values (floating-point numbers) in the
        material node.

        Args:
            level_exposed (int, optional): See `levels_exposed` in the `parameters` method.
                Defaults to 2.

        Returns:
            int: Aggregated number of optimizable parameter values (elements).
        """
        num_param_values = 0

        # Count exposed parameters
        if level_exposed >= 1:
            exposed_param_views = self.parameters(1, True, True)
            num_param_values += sum(view.shape[0] for view in exposed_param_views)

        # Count other node parameters
        if level_exposed != 1:
            num_param_values += sum(node.num_parameters() for node in self.nodes)

        return num_param_values

    def get_parameters_as_tensor(self, level_exposed: int = 2) -> Optional[th.Tensor]:
        """Get the values of optimizable parameters of the material graph as a 1D torch tensor.
        Returns None if there is no optimizable parameters in the graph.

        Args:
            level_exposed (int, optional): See `levels_exposed` in the `parameters` method.
                Defaults to 2.

        Returns:
            Optional[Tensor]: Flattened concatenation of optimizable parameters in the graph,
                or None if the graph doesn't have optimizable parameters.
        """
        # Collect parameters from optimizable exposed parameters
        if level_exposed >= 1:
            exposed_param_values = list(self.parameters(1, detach=True, flatten=True))
            exposed_param_values = th.cat(exposed_param_values) if exposed_param_values else None

        # Combine all parameter values into a 1D tensor
        all_param_values = [exposed_param_values]
        if level_exposed != 1:
            all_param_values.extend([node.get_parameters_as_tensor() for node in self.nodes])

        all_param_values = [pv for pv in all_param_values if pv is not None]
        return th.cat(all_param_values) if all_param_values else None

    def set_parameters_from_tensor(self, values: th.Tensor, level_exposed: int = 2):
        """Set the optimizable parameters of the material graph from a 1D torch tensor.

        Args:
            values (tensor, optional): Source parameter values (must be 1D tensor).
            level_exposed (int, optional): See `levels_exposed` in the `parameters` method.
                Defaults to 2.

        Raises:
            ValueError: The input is not a tensor or doesn't have a 1D shape.
            RuntimeError: The size of the input tensor does not match the number of optimizable
                parameters in the graph.
        """
        # Check if the input is a 1D torch tensor
        if not isinstance(values, th.Tensor) or values.ndim != 1:
            raise ValueError('The input must be a 1D torch tensor.')

        values = values.detach()
        pos = 0

        # Update optimizable exposed parameters
        if level_exposed >= 1:
            exposed_param_views = list(self.parameters(1, detach=True, flatten=True))
            num_exposed_params = [view.shape[0] for view in exposed_param_views]
            for view, size in zip(exposed_param_views, num_exposed_params):
                view.copy_(values.narrow(0, pos, size))
                pos += size

        # Update other node parameters
        if level_exposed != 1:
            for node in self.nodes:
                num_node_params = node.num_parameters()
                if num_node_params:
                    node.set_parameters_from_tensor(values.narrow(0, pos, num_node_params))
                    pos += num_node_params

        # Check if all provided parameters have been exhausted
        if pos < values.shape[0]:
            raise RuntimeError(f'The size of the input tensor ({values.shape[0]}) does not match '
                               f'the number of optimizable parameters in the graph ({pos})')

    def set_parameters_from_config(self, config: Dict[str, Dict[str, Dict[str, Constant]]]):
        """Set parameter values of the material graph from a nested dict-type configuration in the
        following format:
        ```
        {node_name}: # x many
          {param_name}: # x many
            value: {param_value}
            normalize: {False/True}
            {other keyword arguments}
        ```

        Args:
            config (Dict[str, Dict[str, Dict[str, Constant]]]): Parameter configuration as
                outlined above.
        """
        # Build node name-to-object dictionary
        node_dict = {node.name: node for node in self.nodes}

        # Set parameter values in each node by configuration
        for node_name, node_param_config in config.items():
            node_dict[node_name].set_parameters_from_config(node_param_config)

    def summarize(self, filename: str) -> GraphSummary:
        """Generate a summary of graph status containing nodes and parameters. The summary is
        returned and also saved into a local file in YAML format.

        Args:
            filename (str): Path to the saved summary file.

        Returns:
            NodeSummary: A dictionary that summarizes essential information of the graph, including
                name, summaries of graph nodes, and exposed parameters.
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
