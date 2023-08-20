from abc import abstractmethod
from typing import Any, List, Tuple, Dict, Iterator

from diffmat.core.base import BaseParameter
from diffmat.core.function.base import BaseFunctionNode
from diffmat.core.types import InputDict, OutputList, ParamValue, DeviceType
from diffmat.core.util import OL


class BaseFXMapNode(BaseFunctionNode[List[BaseParameter]]):
    """A base class for FX-map graph nodes, which have parameters in the form of objects but only
    one output connector each.
    """
    def __init__(self, name: str, type: str, params: List[BaseParameter] = [],
                 inputs: InputDict = {}, outputs: OutputList = [], **kwargs):
        """Initialize the base FX-map node object.
        """
        super().__init__(name, type, params, inputs, outputs, **kwargs)

    def compile(self, var_levels: Dict[str, int] = {}) -> Dict[str, int]:
        """Compile dynamic node parameters and update the output levels of all parameters that are
        effective to this node.
        """
        # Add the level information of internal parameters and non-dynamic parameters
        var_levels.update({key: OL.get_level(val) for key, val in self.internal_params.items()})

        for param in (p for p in self.params if p.IS_DYNAMIC):
            param.compile(var_levels)

        return var_levels

    def _evaluate_node_params(self, var: Dict[str, ParamValue] = {}) -> \
            Tuple[Dict[str, ParamValue], Dict[str, ParamValue]]:
        """Compute the values of node parameters (include dynamic ones). Also updates the
        collection of variables effective in this node.
        """
        # Initialize the dictionary that maps node parameter names to values
        node_params: Dict[str, ParamValue] = {}

        # Evaluate dynamic parameters (be aware of inter-parameter dependency)
        var.update(self.internal_params)

        for param in self.params:
            value = param.evaluate(var) if param.IS_DYNAMIC else param.evaluate()
            node_params[param.name] = value

        return node_params, var

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Iterator[Any]: ...

    def to_device(self, device: DeviceType = 'cpu'):
        """Move the material graph node to a specified device (e.g., CPU or GPU).

        Args:
            device (DeviceType, optional): Target device ID. Defaults to 'cpu'.
        """
        # Move data members to the target device
        for param in self.params:
            param.to_device(device)

        super().to_device(device)
