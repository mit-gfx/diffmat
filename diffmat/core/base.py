from abc import ABC, abstractmethod
from collections import deque
from contextlib import contextmanager
from weakref import ref
from typing import Generic, TypeVar
from typing import Union, Optional, Any, List, Tuple, Dict, Set, Callable, Iterator
import itertools
import logging
import time
import random

import torch as th

from .types import Constant, FloatArray, ParamValue, ParamSummary, NodeSummary
from .types import ConstantDict, InputDict, MultiInputDict, OutputList, MultiOutputDict
from .types import DeviceType
from .util import OL, check_arg_choice


# Generic type definitions
NT = TypeVar('NT', bound='BaseNode')
FNT = TypeVar('FNT', bound='BaseFunctionNode')
PDT = TypeVar('PDT', bound=Union[ParamValue, 'BaseFunctionGraph'])
PST = TypeVar('PST', List['BaseParameter'], ConstantDict)
ICT = TypeVar('ICT', InputDict, MultiInputDict)
OCT = TypeVar('OCT', OutputList, MultiOutputDict)


class BaseEvaluableObject(ABC):
    """A base class for all evaluable objects, including graphs, nodes, and parameters, which
    implement an `evaluate` method.
    """
    def __init__(self, parent: Optional['BaseEvaluableObject'] = None, device: DeviceType = 'cpu'):
        """Initialize the evaluable object, including device placement, a parent object link, and
        a module-wide logger.

        Args:
            parent (BaseEvaluableObject, optional): Link to the parent object for higher-level
                attribute retrieval. Defaults to None.
            device (DeviceType, optional): Device placement of the object (can be any acceptable
                PyTorch device type). Defaults to 'cpu'.
        """
        self.device = th.device(device)

        # The weak reference must be called to obtain the original object
        self.set_parent(parent)

        # Register a logger for all core classes
        self.logger = logging.getLogger('diffmat.core')

    def set_parent(self, parent: Optional['BaseEvaluableObject'] = None):
        """Set the parent object using a weak reference.

        Args:
            parent (BaseEvaluableObject, optional): Link to the parent object for context lookup.
                Defaults to None.
        """
        self.parent = ref(parent) if parent is not None else None

    def link_as_parent(self, *objs: 'BaseEvaluableObject'):
        """Set the parent of a list of evaluable objects to this object.

        Args:
            objs (Sequence[BaseEvaluableObject]): List of objects whose parent link will point to
                this object.
        """
        for obj in objs:
            obj.set_parent(self)

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Any:
        """Define the functionality of an evaluable object. Must be implemented by child classes.
        """
        ...

    def to_device(self, device: DeviceType = 'cpu'):
        """Move the evaluable object to a specified device (e.g., CPU or GPU). Effective to torch
        tensors only.

        Args:
            device (DeviceType, optional): Device placement of the object (can be any acceptable
                PyTorch device type). Defaults to 'cpu'.
        """
        self.device = th.device(device)

    def _t(self, data: FloatArray) -> th.Tensor:
        """Helper function that converts any data to float32 torch tensor using the local device
        label.

        Args:
            data (FloatArray): Input data (can be an instance, list, or array of numbers).

        Returns:
            Tensor: th.Tensor object that contains the input data in th.float32 data type.
        """
        return th.as_tensor(data, dtype=th.float32, device=self.device)

    def _at(self, data: FloatArray) -> th.Tensor:
        """Helper function that converts any data to torch tensor using the local device label and
        without specifying the data type.

        Args:
            data (FloatArray): Input data.

        Returns:
            Tensor: th.Tensor object that contains the input data.
        """
        return th.as_tensor(data, device=self.device)

    @contextmanager
    def timer(self, header: str, log_level: str = 'info', unit: str = 'ms') -> Iterator[None]:
        """A context manager that times the code inside.

        Args:
            header (str): On-screen header of the timer (e.g., describing its content).
            log_level (str, optional): Message level of the timer ('info' or 'debug').
                Defaults to 'info'.
            unit (str, optional): Time unit ('ms' or 's'). Defaults to 'ms'.

        Yields:
            Iterator[None]: Placeholder return value.
        """
        # Check input validity
        check_arg_choice(log_level, ['info', 'debug'], arg_name='log_level')
        check_arg_choice(unit, ['ms', 's'], arg_name='unit')

        t_start = time.time()
        yield
        t_duration = time.time() - t_start
        t_str = f'{t_duration * 1e3:.3f} ms' if unit == 'ms' else f'{t_duration:.3f} s'
        getattr(self.logger, log_level)(f'{header}: {t_str}')

    @contextmanager
    def temp_rng(self, state: Optional[th.Tensor] = None, seed: int = 0) -> \
            Iterator[Optional[th.Tensor]]:
        """A context manager that temporarily resets the random number generator state when running
        the code inside and restores the previous state upon exit.

        Args:
            state (tensor, optional): The random number generator state to temporarily reset to.
                If set to None, the sandbox RNG environment will be reset using a random seed.
                Defaults to None.
            seed (int, optional): The random seed for RNG re-initialization when `state` is None.
                Defaults to 0.

        Yields:
            Optional[Tensor]: Returns the initial RNG state after being reset by a random seed.
        """
        # Save the current random number generator states
        rng_state_cpu = th.get_rng_state()
        rng_state_gpu = th.cuda.get_rng_state_all()
        rng_state_py = random.getstate()

        # Set the new random number generator state
        if state is not None:
            self.set_rng_state(state)
            yield None
        else:
            th.manual_seed(seed)
            random.seed(seed)
            yield self.get_rng_state()

        # Restore the previous states
        th.set_rng_state(rng_state_cpu)
        th.cuda.set_rng_state_all(rng_state_gpu)
        random.setstate(rng_state_py)

    def get_rng_state(self) -> th.Tensor:
        """Get the random number generator state of the local device.

        Returns:
            Tensor: Current RNG state.
        """
        on_cuda = self.device.type == 'cuda'
        return th.cuda.get_rng_state(self.device) if on_cuda else th.get_rng_state()

    def set_rng_state(self, state: th.Tensor):
        """Set the random number generator state of the local device.

        Args:
            state (tensor): Target RNG state.
        """
        on_cuda = self.device.type == 'cuda'
        th.cuda.set_rng_state(state, self.device) if on_cuda else th.set_rng_state(state)


class BaseParameter(Generic[PDT], BaseEvaluableObject):
    """A base class for all parameter objects such as constants, optimizable parameters, and
    dynamic parameters. It has an optional reference to the parent (node) object.

    Static members:
        IS_OPTIMIZABLE (bool): Whether the parameter is considered optimizable. Defaults to False.
        IS_DYNAMIC (bool): Whether the parameter holds a dynamic value as defined using a function
            graph instead of a literal or tensor. Defaults to False.
    """
    # If the parameter is optimizable
    IS_OPTIMIZABLE = False

    # If the parameter holds dynamic value (defined by a value processor)
    IS_DYNAMIC = False

    def __init__(self, name: str, data: PDT, **kwargs):
        """Initialize the base parameter object.

        Args:
            name (str): Parameter name.
            data (PDT): Parameter data (of generic type).
            kwargs (Dict[str, Any]): Keyword arguments to pass into the parent class constructor.
        """
        super().__init__(**kwargs)

        self.name = name
        self.data = data

        # Link the parent of function node graph to this parameter (dynamic parameters only)
        if self.IS_DYNAMIC:
            self.link_as_parent(data)

    @abstractmethod
    def output_level(self) -> int:
        """Get the category (or level) of the parameter value for static type checking in function
        graph translation.

        Returns:
            int: Parameter value level.
        """
        ...

    @abstractmethod
    def set_value(self, value: PDT):
        """Set parameter value from an external object.

        Args:
            value (PDT): Parameter data (of generic type).
        """
        ...

    def summarize(self) -> ParamSummary:
        """Output a snapshot-like summary of the parameter for on-screen display, including its
        name and current value. Tensors are casted back to lists.

        Returns:
            ParamSummary: Parameter summary dictionary.
        """
        if self.IS_DYNAMIC:
            value = 'dynamic'
        else:
            value = self.evaluate()
            if isinstance(value, th.Tensor):
                value = value.detach().cpu().tolist()

        return {'name': self.name, 'value': value}


class BaseNode(Generic[PST, ICT, OCT], BaseEvaluableObject):
    """A base class for generic node objects in computation graphs, which have one or more I/O
    connectors and contain a set of parameters that control their behaviors.
    """
    def __init__(self, name: str, params: PST, inputs: ICT, outputs: OCT, **kwargs):
        """Initialize the base node object.

        Args:
            name (str): Node name.
            params (PST): Container of node parameters (of generic type).
            inputs (ICT): List or mapping of input connections (of generic type).
            outputs (OCT): List or mapping of output connections (of generic type).
            kwargs (Dict[str, Any]): Keyword arguments to pass into the parent class constructor.
        """
        super().__init__(**kwargs)

        self.name = name
        self.params = params
        self.inputs = inputs
        self.outputs = outputs

        # Link the parent of node parameters to this node
        if isinstance(params, list):
            self.link_as_parent(*params)


class BaseMaterialNode(BaseNode[List[BaseParameter], MultiInputDict, MultiOutputDict]):
    """A base class for differentiable material nodes where parameters are represented by objects.
    """
    def __init__(self, name: str, res: int, params: List[BaseParameter] = [],
                 inputs: MultiInputDict = {}, outputs: MultiOutputDict = {},
                 seed: int = 0, **kwargs):
        """Initialize the base material node object (including internal node parameters).

        Args:
            name (str): Material node name.
            res (int): Output texture resolution (after log2).
            params (List[BaseParameter], optional): List of node parameters. Defaults to [].
            inputs (MultiInputDict, optional): Mapping from input connector names to corresponding
                output slots of predecessor nodes. Defaults to {}.
            outputs (MultiOutputDict, optional): Mapping from output connector names to a list of
                successor nodes. Defaults to {}.
            seed (int, optional): Random seed to node function. Defaults to 0.
            kwargs (Dict[str, Any]): Keyword arguments to pass into the parent class constructor.
        """
        super().__init__(name, params, inputs, outputs, **kwargs)

        self.res = res
        self.seed = seed

        # Internal node parameters
        self.internal_params: Dict[str, Constant] = {
            '$size': [float(1 << res), float(1 << res)],
            '$sizelog2': [float(res), float(res)],
            '$normalformat': 0,
            '$tiling': 0,
        }

    def compile(self, exposed_param_levels: Dict[str, int] = {},
                master_seed: int = 0, inherit_seed: bool = True) -> Dict[str, int]:
        """Compile function graphs inside dynamic node parameters, and acquire the value categories
        of all named variables effective to this node for static type checking.

        Args:
            exposed_param_levels (Dict[str, int], optional): Value category mapping of exposed
                parameters in a material graph. Defaults to {}.
            master_seed (int, optional): Graph-wide random seed, to which per-node random seeds
                serve as offsets in the seed value. Defaults to 0.
            inherit_seed (bool, optional): Switch for overwriting the internal random seed using
                the provided `master_seed`. Defaults to True.

        Returns:
            Dict[str, int]: Value category mapping of named variables accessible from this node.
        """
        # Add the level information of internal parameters and non-dynamic parameters
        var_levels = exposed_param_levels.copy()
        var_levels.update({key: OL.get_level(val) for key, val in self.internal_params.items()})

        # Inherit the graph-level random seed
        if inherit_seed:
            self.seed = master_seed

        # Initialize the random number generator
        rng_state = random.getstate()
        random.seed(self.seed)

        for param in (p for p in self.params if p.IS_DYNAMIC):
            param.compile(var_levels)

        # Reset the random number generator
        random.setstate(rng_state)

        return var_levels

    def _evaluate_node_params(self, exposed_params: Dict[str, ParamValue] = {}) -> \
            Tuple[Dict[str, Optional[ParamValue]], Dict[str, ParamValue]]:
        """Compute the values of node parameters (include dynamic ones). Also returns the
        collection of variables effective in this node.

        Args:
            exposed_params (Dict[str, ParamValue], optional): Name-to-value mapping for exposed
                parameters in the material graph. Defaults to {}.

        Returns:
            Dict[str, Optional[ParamValue]]: Node parameter value dictionary.
            Dict[str, ParamValue]: Named variables value dictionary.
        """
        # Initialize the dictionary that maps node parameter names to values
        node_params: Dict[str, Optional[ParamValue]] = {}

        # Evaluate dynamic parameters (be aware of inter-parameter dependency)
        var = exposed_params.copy()
        var.update(self.internal_params)

        for param in self.params:
            value = param.evaluate(var) if param.IS_DYNAMIC else param.evaluate()
            node_params[param.name] = value

            # Update the tiling variable since it can be referenced as an internal parameter
            if param.name == 'tiling':
                var['$tiling'] = value

        return node_params, var

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Union[th.Tensor, Tuple[th.Tensor, ...]]:
        """Node function wrapper. See `functional.py` for actual implementations.
        """
        ...

    def train(self):
        """Switch to training mode where all optimizable parameters require gradient.
        """
        for param in self.parameters():
            param.requires_grad_(True)

    def eval(self):
        """Switch to evaluation mode where no optimizable parameter requires gradient.
        """
        for param in self.parameters():
            param.requires_grad_(False)

    def parameters(self, detach: bool = False, flatten: bool = False) -> Iterator[th.Tensor]:
        """Return an iterator over optimizable parameter values in the material node (tensor views
        rather than copies).

        Args:
            detach (bool, optional): Whether returned tensor views are detached (i.e., don't
                require gradient). Defaults to False.
            flatten (bool, optional): Whether returned tensor views are flattened.
                Defaults to False.

        Yields:
            Iterator[Tensor]: Tensor views of optimizable node parameter values.
        """
        # Collect parameter values from optimizable parameters
        for param in (p for p in self.params if p.IS_OPTIMIZABLE):
            data: Optional[th.Tensor] = param.data
            if data is not None:
                data = data.detach() if detach else data
                data = data.view(-1) if flatten else data
                yield data

    def num_parameters(self) -> int:
        """Count the number of optimizable parameter values (floating-point numbers) in the
        material node.

        Returns:
            int: Aggregated number of optimizable parameter values (elements).
        """
        return sum(view.shape[0] for view in self.parameters(detach=True, flatten=True))

    def get_parameters_as_tensor(self) -> Optional[th.Tensor]:
        """Get the values of optimizable parameters of the material node as a 1D torch tensor.
        The tensor will be on the same device as parameter values by default.

        The function returns None if the material node doesn't have any optimizable parameters.

        Returns:
            Optional[Tensor]: Flattened concatenation of optimizable parameter values in the node,
                or None if the node doesn't have optimizable parameters.
        """
        # Collect parameter values from optimizable parameters
        param_values = list(self.parameters(detach=True, flatten=True))

        # Concatenate the parameter values into a 1D tensor
        return th.cat(param_values) if param_values else None

    def set_parameters_from_tensor(self, values: th.Tensor):
        """Set the optimizable parameters of the material graph from a 1D torch tensor.

        Args:
            values (tensor): Source parameter values (must be 1D tensor).

        Raises:
            ValueError: The input is not a tensor or doesn't have a 1D shape.
            RuntimeError: The material node does not have optimizable parameters but the function
                is called.
            RuntimeError: The size of the input tensor does not match the number of optimizable
                parameters in the node.
        """
        # Check if the input is a 1D torch tensor
        if not isinstance(values, th.Tensor) or values.ndim != 1:
            raise ValueError('The input must be a 1D torch tensor.')

        values = values.detach()

        # Obtain flattened views of optimizable parameter tensors
        param_views = list(self.parameters(detach=True, flatten=True))
        if not param_views:
            raise RuntimeError('This material nodes does not have optimizable parameters')

        # Check if the number of parameters provided match the number that this node has
        num_params = [view.shape[0] for view in param_views]
        if sum(num_params) != values.shape[0]:
            raise RuntimeError(f'The size of the input tensor ({values.shape[0]}) does not match '
                               f'the optimizable parameters ({sum(num_params)}) in this node')

        # Update the parameter values
        pos = 0
        for view, size in zip(param_views, num_params):
            view.copy_(values.narrow(0, pos, size))
            pos += size

    def set_parameters_from_config(self, config: Dict[str, Dict[str, Constant]]):
        """Set parameter values of the material graph from a nested dict-type configuration in the
        following format:
        ```
        {param_name}: # x many
          value: {param_value}
          normalize: {False/True}
          {other keyword arguments}
        ```

        Args:
            config (Dict[str, Dict[str, Constant]]): Parameter configuration as outlined above.
        """
        # Build a parameter name-to-object dictionary
        param_dict = {param.name: param for param in self.params}

        for param_name, param_config in config.items():
            param_dict[param_name].set_value(**param_config)

    def summarize(self) -> NodeSummary:
        """Generate a summary of node status, including name, I/O, and parameters.

        Returns:
            NodeSummary: A dictionary that summarizes essential information of the node, including
                name, input connections, and node parameter values.
        """
        get_variable_name: Callable[[str, str], str] = \
            lambda name, output: f'{name}_{output}' if output else name

        return {
            'name': self.name,
            'input': [get_variable_name(*val) if val is not None else None \
                      for val in self.inputs.values()],
            'param': dict(tuple(p.summarize().values()) for p in self.params)
        }

    def to_device(self, device: DeviceType = 'cpu'):
        """Move the material graph node to a specified device (e.g., CPU or GPU).

        Args:
            device (DeviceType, optional): Target device ID. Defaults to 'cpu'.
        """
        # Move data members to the target device
        for param in self.params:
            param.to_device(device)

        super().to_device(device)


class BaseFunctionNode(BaseNode[ConstantDict, InputDict, OutputList]):
    """A base class for function nodes in value processors where all parameters are constants and
    each node only has one output connector.
    """
    def __init__(self, name: str, params: ConstantDict = {}, inputs: InputDict = {},
                 outputs: OutputList = [], **kwargs):
        """Initialize the base function node object.

        Args:
            name (str): Function node name.
            params (ConstantDict, optional): Dictionary of node parameters. Defaults to {}.
            inputs (InputDict, optional): Mapping from input connector names to corresponding
                predecessor function nodes. Defaults to {}.
            outputs (OutputList, optional): List of successor function nodes. Defaults to [].
            kwargs (Dict[str, Any]): Keyword arguments to pass into the parent class constructor.
        """
        super().__init__(name, params, inputs, outputs, **kwargs)

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Union[ParamValue, Tuple[ParamValue, str]]:
        """Execute the mathematic function in this node and optionally return the infered
        expression that describes the node function.
        """
        ...

    @abstractmethod
    def evaluate_expr(self, *args, **kwargs) -> Tuple[int, str]:
        """Generate an expression that describes the node function given expressions of its
        operands.
        """
        ...


class BaseGraph(Generic[NT], BaseEvaluableObject):
    """A base class for node-based computation graphs, defined as a container of nodes with an
    optional reference to the parent (parameter, node, or None) object.
    """
    def __init__(self, nodes: List[NT], **kwargs):
        """Initialize the base graph object.

        Args:
            nodes (List[NT]): List of graph nodes (of generic types).
            kwargs (Dict[str, Any]): Keyword arguments to pass into the parent class constructor.
        """
        super().__init__(**kwargs)

        self.nodes = nodes

        # Link the parents of material/function graph nodes to this graph
        self.link_as_parent(*nodes)

    def _sort_nodes(self):
        """Topologically sort graph nodes according to data dependency.
        """
        # Build a node dictionary indexed by name
        node_dict = {node.name: node for node in self.nodes}

        # Count the in-degree of graph nodes
        in_degrees: Dict[str, int] = {}
        for node in self.nodes:
            in_degrees[node.name] = sum([v is not None for v in node.inputs.values()])

        # Topologically sort graph nodes
        node_sequence: List[NT] = []
        queue = deque(node for node in self.nodes if not in_degrees[node.name])
        while queue:
            node = queue.popleft()
            node_sequence.append(node)

            # Accommodate two possible node output formats
            if isinstance(node.outputs, dict):
                next_node_list = itertools.chain(*node.outputs.values())
            else:
                next_node_list = node.outputs

            for next_node_name in next_node_list:
                in_degrees[next_node_name] -= 1
                if not in_degrees[next_node_name]:
                    queue.append(node_dict[next_node_name])

        # Store the sorted node sequence for debugging purposes
        self.nodes[:] = node_sequence

    @abstractmethod
    def compile(self, *args, **kwargs) -> Any:
        """Compile the node graph into an instruction sequence. This step typically invokes the
        `compile` method in each graph node.
        """
        ...

    def to_device(self, device: DeviceType = 'cpu'):
        """Move the material graph to a specified device (e.g., CPU or GPU).

        Args:
            device (DeviceType, optional): Target device ID. Defaults to 'cpu'.
        """
        # Move nodes to the target device
        for node in self.nodes:
            node.to_device(device)

        super().to_device(device)


class BaseFunctionGraph(Generic[FNT], BaseGraph[FNT]):
    """A base class for function node graphs that consist of a collection of computation nodes and
    have a designated output node where the result is read from.
    """
    def __init__(self, nodes: List[FNT], output_node: FNT, **kwargs):
        """Initialize the base function node graph object.

        Args:
            nodes (List[FNT]): List of function graph nodes (of generic types).
            output_node (FNT): Output node of the graph, which must be a member of `nodes`.
            kwargs (Dict[str, Any]): Keyword arguments to pass into the parent class constructor.

        Raises:
            ValueError: The designated output node is not found in the input node list.
        """
        # Ensure that the output node is in the node list
        if output_node not in nodes:
            raise ValueError('The output node must be in the provided node list')

        super().__init__(nodes, **kwargs)

        self.output_node = output_node

    def _sort_nodes(self, reverse: bool = True):
        """Topologically sort function graph nodes according to data dependency. Different from the
        default sorting algorithm, this variant observes a DFS (or reverse DFS) order.

        Raises:
            RuntimeError: Found duplicate or missing nodes during sorting.

        Args:
            reverse (bool, optional): Switch for sorting nodes in reverse DFS order.
                Defaults to True.
        """
        # Initialize the DFS stack, an input counter array, and the output node sequence
        stack: List[FNT] = [self.output_node]
        status: List[int] = [-1]
        node_sequence: List[FNT] = []

        # Build a dictionary for node name to input nodes lookup
        node_dict: Dict[str, FNT] = {node.name: node for node in self.nodes}
        input_dict: Dict[str, List[Optional[FNT]]] = \
            {node.name: [node_dict.get(n) for n in node.inputs.values()] for node in self.nodes}

        # Avoid revisiting the same node if specified
        visited: Set[str] = set()

        # Start DFS
        while stack:

            # Get the node at the top and proceed to the next input connection we handle
            # Also mark the node as visited for deduplication
            node, counter = stack[-1], status[-1] + 1
            status[-1] = counter
            visited.add(node.name)
            if not reverse and not counter:
                node_sequence.append(node)

            # Continue to the next layer of DFS if the next input connection exists
            if counter < len(node.inputs):
                next_input = input_dict[node.name][counter]
                if next_input and next_input.name not in visited:
                    stack.append(next_input)
                    status.append(-1)

            # Otherwise, the node has been fully exploited and can be popped from the stack
            else:
                if reverse:
                    node_sequence.append(node)
                stack.pop()
                status.pop()

        # Verify that the result is correct
        if len(node_sequence) != len(self.nodes) or set(node_sequence) != set(self.nodes):
            raise RuntimeError('Some nodes are duplicated or missing. Please check.')

        self.nodes[:] = node_sequence

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> ParamValue:
        """Evaluate the function graph by executing its compiled sequence of instructions and
        return the output value.
        """
        ...
