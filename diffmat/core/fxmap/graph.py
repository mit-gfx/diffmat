from collections import deque
from typing import List, Dict

import torch as th
import numpy as np
import numpy.typing as npt

from diffmat.core.base import BaseParameter
from diffmat.core.function.base import BaseFunctionGraph
from diffmat.core.types import (
    Constant, ParamValue, FloatVector, InputDict, OutputList, FXMapNodeGenerator
)
from .base import BaseFXMapNode
from .engine import FXMapExecutor


class FXMapQuadrant(BaseFXMapNode):
    """Quadrant nodes in FX-map graphs.
    """
    def __init__(self, name: str, type: str, params: List[BaseParameter] = [],
                 inputs: InputDict = {}, outputs: OutputList = [], **kwargs):
        """Initialize the quadrant node object.
        """
        super().__init__(name, type, params=params, inputs=inputs, outputs=outputs, **kwargs)

        # Internal parameters (depth and 2^(-depth))
        self.internal_params: Dict[str, Constant] = {
            '$depth': 1.0,
            '$depthpow2': 1.0,
            '$pos': [0.5, 0.5]
        }

    def _add_offset(self, base: FloatVector, offset: FloatVector) -> FloatVector:
        """Add two floating point vectors of various types.
        """
        if isinstance(offset, list) and offset == [0.0, 0.0]:
            return base
        elif isinstance(base, th.Tensor) or isinstance(offset, th.Tensor):
            return self._t(base) + self._t(offset)
        else:
            return np.asarray(base, dtype=np.float32) + np.asarray(offset, dtype=np.float32)

    def evaluate(self, executor: FXMapExecutor, pos: npt.NDArray[np.float32],
                 branch_pos: FloatVector, depth: int = 0,
                 var: Dict[str, ParamValue] = {}) -> FXMapNodeGenerator:
        """Evaluate the quadrant node. This step submits an atomic pattern generation job to the
        FX-map executor as dictated by the node parameters, and then iteratively generates
        references to its childrens for subsequent graph traversal.
        """
        # Update the internal variable for pattern position and pattern depth
        self.internal_params.update({
            '$depth': float(depth),
            '$depthpow2': 2.0 ** -depth,
            '$pos': pos.tolist()
        })

        # Evaluate the node parameters and update the set of active variables
        node_params, var = self._evaluate_node_params(var)

        # Create a pattern generation job and submit it to the executor
        branch_pos = self._add_offset(branch_pos, node_params['branch_offset'])

        if node_params['type'] != 'none':

            # Calculate pattern position using pattern offset
            pattern_pos = self._add_offset(branch_pos, node_params['pattern_offset'])
            node_params['offset'] = pattern_pos
            del node_params['branch_offset'], node_params['pattern_offset']

            # Add pattern depth to the parameter list
            node_params['depth'] = depth

            executor.submit_job(**node_params)

        # Generate references to children if they exist
        if any(self.inputs.values()):

            # Compute the pattern positions of four children
            pos_offset_arr = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=np.float32)
            pos_offset_arr *= 2 ** (-depth - 2)
            child_pos = pos + pos_offset_arr
            child_branch_pos = self._add_offset(branch_pos, pos_offset_arr)

            for i, child_name in enumerate(self.inputs.values()):
                if child_name:
                    yield child_name, child_pos[i], child_branch_pos[i]


class FXMapSwitch(BaseFXMapNode):
    """Switch nodes in FX-map graphs.
    """
    def __init__(self, name: str, type: str, params: List[BaseParameter] = [],
                 inputs: InputDict = {}, outputs: OutputList = [], **kwargs):
        """Initialize the switch node object.
        """
        super().__init__(name, type, params=params, inputs=inputs, outputs=outputs, **kwargs)

        # Internal parameters (none)
        self.internal_params: Dict[str, Constant] = {}

    def evaluate(self, pos: npt.NDArray[np.float32], branch_pos: FloatVector,
                 var: Dict[str, ParamValue] = {}) -> FXMapNodeGenerator:
        """Evaluate the switch node. This step determines which child node to follow.
        """
        # Evaluate the node parameters and update the set of active variables
        node_params, var = self._evaluate_node_params(var)

        # Generate a reference to the selected child
        child = self.inputs['input_1' if node_params['switch'] else 'input_0']
        if child:
            yield child, pos, branch_pos


class FXMapIterate(BaseFXMapNode):
    """Iterate nodes in FX-map graphs.
    """
    def __init__(self, name: str, type: str, params: List[BaseParameter] = [],
                 inputs: InputDict = {}, outputs: OutputList = [], **kwargs):
        """Initialize the iterate node object.
        """
        super().__init__(name, type, params=params, inputs=inputs, outputs=outputs, **kwargs)

        # Internal parameters (iteration number)
        self.internal_params = {'$number': 0}

    def evaluate(self, pos: npt.NDArray[np.float32], branch_pos: FloatVector,
                 var: Dict[str, ParamValue] = {}) -> FXMapNodeGenerator:
        """Evaluate the iterate node. This step generates the left child first and then the right
        child, the latter of which is repeated for several times.
        """
        # Evaluate the node parameters and update the set of active variables
        node_params, var = self._evaluate_node_params(var)

        # First generate a reference to the left child
        # Note that the iteration number must be reset since it might be changed elsewhere
        left_child = self.inputs['input_0']
        if left_child:
            var['$number'] = 0
            yield left_child, pos, branch_pos

        # Then generate a reference to the right child, repeated for several times
        right_child = self.inputs['input_1']
        if right_child:
            for i in range(node_params['number']):
                var['$number'] = i
                yield right_child, pos, branch_pos


class FXMapGraph(BaseFunctionGraph[BaseFXMapNode]):
    """Class for differentiable FX-map graphs.
    """
    def __init__(self, nodes: List[BaseFXMapNode], output_node: BaseFXMapNode, **kwargs):
        """Initialize the FX-map graph object.
        """
        super().__init__(nodes, output_node, **kwargs)

        # Sort the nodes in DFS order for compilation
        self._sort_nodes(reverse=False)

    def max_depth(self) -> int:
        """Compute the maximum depth of the FX-map graph using topological sorting.
        """
        # Count the in-degree of graph nodes
        # Different from a material graph, the in-degree is counted on the output side
        in_degrees: Dict[str, int] = {node.name: 0 for node in self.nodes}
        for node in self.nodes:
            for next_node_name in (s for s in node.inputs.values() if s is not None):
                in_degrees[next_node_name] += 1

        # Initialize the maximum depth of every graph node
        depths: Dict[str, int] = \
            {node.name: isinstance(node, FXMapQuadrant) - 1 for node in self.nodes}

        # Look-up dictionary from node name to node object
        node_dict: Dict[str, BaseFXMapNode] = {node.name: node for node in self.nodes}

        # Topological sorting
        queue = deque([self.output_node])
        while queue:
            node = queue.popleft()
            d = depths[node.name]
            for next_node_name in (s for s in node.inputs.values() if s is not None):
                in_degrees[next_node_name] -= 1
                next_node = node_dict[next_node_name]
                if not in_degrees[next_node_name]:
                    queue.append(next_node)

                # Update the maximum depth counter of the next node
                next_depth = d + 1 if isinstance(next_node, FXMapQuadrant) else d
                depths[next_node_name] = max(depths[next_node_name], next_depth)

        return max(depths.values())

    def compile(self, var_levels: Dict[str, int] = {}):
        """Compile the FX-map graph, instantiating programs for dynamic parameters in FX-map nodes.
        """
        # Add the starting position
        var_levels['$pos'] = 1

        # Compile FX-map nodes in the order of visit (DFS)
        for node in self.nodes:
            node.compile(var_levels)

    def evaluate(self, executor: FXMapExecutor, var: Dict[str, ParamValue] = {}):
        """Traverse the FX-map graph to collect pattern generation jobs from quadrant nodes.
        """
        # Add the starting position variable
        var['$pos'] = [0.5, 0.5]

        # Initialize the traversal stack. Each stack entry is a node reference generator obtained
        # from the `evaluate` method of FX-map nodes. It points towards existing children of a
        # node and provides their position info.
        stack: List[FXMapNodeGenerator] = []

        # FX-map depth increment stack that accompanies the generator stack
        depth_inc: List[int] = []

        # Adaptor function for accessing the child reference generator an FX-map node
        def evaluate_node(node: BaseFXMapNode, pos: npt.NDArray[np.float32],
                          branch_pos: FloatVector, depth: int) -> FXMapNodeGenerator:
            if isinstance(node, FXMapQuadrant):
                return node.evaluate(executor, pos, branch_pos, depth=depth, var=var)
            else:
                return node.evaluate(pos, branch_pos, var=var)

        # The stack starts with the generator of the output node and proceeds top-down
        # Note that the initial position is (0.5, 0.5) since Substance uses [0, 1] for image
        # coordinates
        pos = np.array((0.5, 0.5), dtype=np.float32)
        branch_pos = pos
        stack.append(evaluate_node(self.output_node, pos, branch_pos, 0))
        depth_inc.append(int(isinstance(self.output_node, FXMapQuadrant)))

        # Build a dictionary for name-to-object lookup
        node_dict: Dict[str, BaseFXMapNode] = {node.name: node for node in self.nodes}

        # Traversal loop
        depth = 0

        while stack:

            # Forward the current generator state
            next_name, pos, branch_pos = next(stack[-1], ('', None, None))

            # Move to the next children or remove the current node once all children have been
            # covered. Increment depth if the next children is a quadrant node.
            if next_name:
                next_node = node_dict[next_name]
                depth += depth_inc[-1]
                stack.append(evaluate_node(next_node, pos, branch_pos, depth))
                depth_inc.append(int(isinstance(next_node, FXMapQuadrant)))
            else:
                stack.pop()
                depth_inc.pop()
                depth -= depth_inc[-1] if depth_inc else 0
