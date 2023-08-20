from abc import ABC, abstractmethod
from collections import deque
from contextlib import contextmanager
from weakref import ref
from typing import Generic, TypeVar
from typing import Optional, Any, List, Dict, Iterator
import itertools
import logging
import random

import torch as th

from diffmat.core.types import (
    FloatArray, ParamSummary, ConstantDict, InputDict, MultiInputDict, OutputList, \
    MultiOutputDict, DeviceType
)
from diffmat.core.util import Timer, OL


# Generic type definitions
NT = TypeVar('NT', bound='BaseNode')
PDT = TypeVar('PDT')
PST = TypeVar('PST', List['BaseParameter'], ConstantDict)
ICT = TypeVar('ICT', InputDict, MultiInputDict)
OCT = TypeVar('OCT', OutputList, MultiOutputDict)


class BaseEvaluableObject(ABC):
    """A base class for all evaluable objects, including graphs, nodes, and parameters, which
    implement an `evaluate` method.
    """
    # Reference to the timer class
    timer = Timer

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
        IS_OPTIMIZABLE (bool): Whether a continuous parameter is considered optimizable.
            Defaults to False.
        IS_OPTIMIZABLE_INTEGER (bool): Whether a discrete parameter is considered optimizable.
            Defaults to False.
        IS_DYNAMIC (bool): Whether the parameter holds a dynamic value as defined using a function
            graph instead of a literal or tensor. Defaults to False.
    """
    # If the parameter is continuously optimizable
    IS_OPTIMIZABLE = False

    # If the parameter is optimizable by integer values
    IS_OPTIMIZABLE_INTEGER = False

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

    @property
    def output_level(self) -> int:
        """Obtain the category (or level) of the parameter value. A function graph translator uses
        such info to infer operand types when generating program instructions.

        Returns:
            int: Parameter value level.
        """
        return OL.get_level(self.data)

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
    def __init__(self, name: str, type: str, params: PST, inputs: ICT, outputs: OCT, **kwargs):
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
        self.type = type
        self.params = params
        self.inputs = inputs
        self.outputs = outputs

        # Link the parent of node parameters to this node
        if isinstance(params, list):
            self.link_as_parent(*params)


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
