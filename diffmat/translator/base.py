from abc import ABC, abstractmethod
from collections import deque
from operator import itemgetter
from xml.etree import ElementTree as ET
from typing import Union, List, Tuple, Set, Dict, Type, Optional, Generic, TypeVar
import logging

from diffmat.core.base import BaseEvaluableObject as BEO
from diffmat.core.base import BaseGraph, BaseNode, BaseParameter
from diffmat.translator.types import PathLike, Constant, NodeData
from diffmat.translator.util import NameAllocator
from diffmat.translator.util import get_value, get_param_type, lookup_value_type, is_image


# Type variables
NTT = TypeVar('NTT', bound='BaseNodeTranslator')
NCT = TypeVar('NCT', bound=dict)


class BaseTranslator(ABC):
    """Base translator class that defines essential members and interfaces.
    """
    def __init__(self, root: Optional[Union[PathLike, ET.Element]]):
        """Construct a base translator associated with an XML tree.

        Args:
            root (Optional[PathLike | Element]): Path to the source XML file, or a root node of
                the XML tree.
        """
        # Read the XML file if 'root' is a file path
        if root and not isinstance(root, ET.Element):
            self.root = ET.parse(root).getroot()
        else:
            self.root = root

        # Register a logger for all translator classes
        self.logger = logging.getLogger('diffmat.translator')

    @abstractmethod
    def translate(self, **obj_kwargs) -> BEO:
        """Translate the XML tree content to a target object (graph, node, or parameter).
        """
        ...

    def write_xml(self, filename: PathLike):
        """Write the XML tree to an output file.

        Args:
            filename: Output file path.
        """
        tree = ET.ElementTree(self.root)
        tree.write(filename)


class BaseParamTranslator(BaseTranslator):
    """Base translator class for material graph parameters (constant, optimizable, or dynamic).
    """
    def __init__(self, root: Optional[ET.Element], name: Optional[str] = None,
                 sbs_name: Optional[str] = None, default: Optional[Constant] = None,
                 sbs_default: Optional[Constant] = None, **param_kwargs):
        """Initialize the base parameter translator.

        Args:
            root (Optional[Element]): Root node of the XML tree. If the root is None, the parameter
                will take on its default value after translation.
            name (Optional[str], optional): Parameter name in Diffmat. Passing None indicates that
                the parameter has identical names (identifiers) in Diffmat and Substance Designer.
                Defaults to None.
            sbs_name (Optional[str], optional): Paramter name in Substance Designer. A None value
                means that the parameter is Diffmat-specific and do not translate to any parameter
                in Substance Designer. Defaults to None.
            default (Optional[Constant], optional): Default parameter value in storage. The value
                should be within [0, 1] for optimizable parameters. Defaults to None.
            sbs_default (Optional[Constant], optional): Default parameter value in Substance
                Designer. The value will be converted to Diffmat standard if `default` is not
                specified. Defaults to None.
            param_kwargs (Dict[str, Any], optional): keyword arguments that will be passed directly
                to the parameter object constructor during translation.

        Raises:
            ValueError: The parameter represents a color or grayscale image.
        """
        super().__init__(root)

        # Note that the parameter name in Substance Designer, i.e., 'sbs_name', is allowed to be
        # 'None'. In that case, the parameter is considered local only and will not perform any
        # back-translation.
        self.sbs_name = sbs_name or (root and get_value(root.find('name'))) or ''
        self.name = name or self.sbs_name
        self.default = default
        self.sbs_default = sbs_default

        # Extra keyword arguments are saved and will be passed to the parameter object
        self.param_kwargs = param_kwargs

        # Parameter type number:
        #   1, 2 - color / grayscale images
        #   4 - bool
        #   16, 32, 64, 128 - int, list of ints (length = 2, 3, 4)
        #   256, 512, 1024, 2048 - float, list of floats (length = 2, 3, 4)
        #   16384 - str
        self.type = get_param_type(root) if root else \
                    lookup_value_type(self.sbs_default) if self.sbs_default is not None else \
                    lookup_value_type(self.default) if self.default is not None else 0

        # Raise error if the parameter is an image
        if is_image(self.type):
            raise ValueError('Image parameters are invalid')

    @abstractmethod
    def translate(self, **obj_kwargs) -> Union[BaseParameter, Tuple[BaseParameter, ...]]:
        """Translates the XML tree into a parameter object.
        """
        ...


class BaseNodeTranslator(BaseTranslator, Generic[NCT]):
    """Base translator class for material or function nodes.
    """
    def __init__(self, root: ET.Element, name: str, type: str, node_config: NCT):
        """Initialize the base node translator.

        Args:
            root (Element): Root node of the XML tree.
            name (str): Node name.
            type (str): Node type or category.
            node_config (NCT): Node configuration (of generic type), such as node function, I/O
                connectors, and parameter specifications.
        """
        super().__init__(root)

        self.name = name
        self.type = type
        self.node_config = node_config.copy()

    @abstractmethod
    def _init_io_connectors(self):
        """Initialize the I/O connectors of the node.
        """
        ...

    @abstractmethod
    def translate(self, **obj_kwargs) -> BaseNode:
        """Translates the XML tree into a graph node object.
        """
        ...


class BaseGraphTranslator(BaseTranslator, Generic[NTT]):
    """Base translator class for material and function node graphs.
    """
    def __init__(self, root: Optional[Union[PathLike, ET.Element]], canonicalize: bool = False):
        """Initialize the base graph translator.

        Args:
            root (Optional[PathLike | Element]): Path to the source XML file, or a root node of
                the XML tree.
            canonicalize (bool, optional): Whether to canonicalize the graph structure by sorting
                the nodes in DFS order. Defaults to False.
        """
        super().__init__(root)

        self.node_name_allocator = NameAllocator()
        self.node_translators: List[NTT] = []

        # Extract and analyze material graph structure to remove redundant nodes
        self.graph: Dict[int, NodeData] = {}
        self._init_graph()
        self._prune_graph()

        # Sort the graph nodes in a canonical order
        if canonicalize:
            self._sort_graph()

        # Initialize node translators and node connections
        self._init_node_translators()
        self._init_graph_connectivity()

    @abstractmethod
    def _init_graph(self):
        """Initialize an internal data structure that records the graph structure.
        """
        ...

    def _prune_graph(self):
        """Prune the graph data structure of redundant nodes that do not contribute to output maps.
        """
        graph = self.graph

        # Maintain the colleciton of nodes that contribute to graph outputs
        visited = set(uid for uid, n in graph.items() if n['is_output'])
        queue = deque(graph[uid] for uid in visited)

        # Run backward BFS to search for other relevant nodes
        while queue:
            node = queue.popleft()
            for _, *in_arr in node['in']:
                in_ref: int = in_arr[0]
                if in_ref not in visited:
                    visited.add(in_ref)
                    queue.append(graph[in_ref])

        # Update the graph data structure to remove unvisited nodes
        self.graph = {uid: graph[uid] for uid in visited}

    def _sort_graph(self):
        """Sort the graph nodes in a canonical DFS order.
        """
        graph, key_func = self.graph, itemgetter(0)
        sorted_uids: List[int] = []
        visited: Set[int] = set()

        # DFS function at each node
        def dfs(uid: int):
            for in_data in sorted(graph[uid]['in'], key=key_func):
                next_uid = in_data[1]
                if next_uid not in visited:
                    dfs(next_uid)
            visited.add(uid)
            sorted_uids.append(uid)

        # Run DFS and re-order graph nodes
        output_uids = [uid for uid, n in graph.items() if n['is_output']]

        for uid in sorted(output_uids):
            dfs(uid)
        self.graph = {uid: graph[uid] for uid in sorted_uids}

    @abstractmethod
    def _init_node_translators(self):
        """Initialize node translators from XML data.
        """
        ...

    @abstractmethod
    def _init_graph_connectivity(self):
        """Initialize graph connectivity using the internal graph data structure.
        """
        ...

    @abstractmethod
    def translate(self, **obj_kwargs) -> BaseGraph:
        """Translate the XML tree into a graph object.
        """
        ...


# Type alias
TranslatorType = Type[BaseTranslator]
