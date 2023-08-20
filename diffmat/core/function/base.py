from abc import abstractmethod
from typing import Generic, TypeVar
from typing import Any, Optional, List, Dict, Set

from diffmat.core.base import BaseParameter, BaseNode, BaseGraph
from diffmat.core.types import ConstantDict, InputDict, OutputList, ParamValue


# Generic type definitions
FNT = TypeVar('FNT', bound='BaseFunctionNode')
PST = TypeVar('PST', List['BaseParameter'], ConstantDict)


class BaseFunctionNode(Generic[PST], BaseNode[PST, InputDict, OutputList]):
    """A base class for function nodes in value processors where all parameters are constants and
    each node only has one output connector.
    """
    def __init__(self, name: str, type: str, params: PST = {}, inputs: InputDict = {},
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
        super().__init__(name, type, params, inputs, outputs, **kwargs)

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Any:
        """Execute the mathematic function in this node and optionally return the infered
        expression that describes the node function.
        """
        ...


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
