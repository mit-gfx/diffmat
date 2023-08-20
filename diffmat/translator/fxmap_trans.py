from xml.etree import ElementTree as ET
from typing import Optional, List, Tuple, Dict, Type

from diffmat.core.base import BaseParameter
from diffmat.core.fxmap import FXMapQuadrant, FXMapSwitch, FXMapIterate, FXMapGraph
from diffmat.core.fxmap.base import BaseFXMapNode
from diffmat.core.material import FXMap
from diffmat.translator import param_trans
from diffmat.translator.base import BaseParamTranslator, BaseNodeTranslator, BaseGraphTranslator
from diffmat.translator.node_trans import MaterialNodeTranslator
from diffmat.translator.types import NodeConfig
from diffmat.translator.util import (
    FACTORY_LUT, get_value, load_node_config, find_connections, gen_input_dict
)

# Dictionary from FX-map node types to classes
CLASS_FACTORY: Dict[str, Type[BaseFXMapNode]] = {
    'paramset': FXMapQuadrant,
    'markov2': FXMapSwitch,
    'addnode': FXMapIterate
}


class FXMapNodeTranslator(BaseNodeTranslator[NodeConfig]):
    """Translator of XML into an FX-map graph node.
    """
    def __init__(self, root: ET.Element, data: ET.Element, name: str, type: str,
                 node_config: NodeConfig):
        """Initialize the FX-map graph node translator. Different from material nodes, the node
        subtree and the parameter subtree are separate.

        Args:
            root (Element): Root XML element of the FX-map graph node.
            data (Element): Root XML element of node parameters.
            name (str): FX-map graph node name.
            type (str): FX-map graph node type.
            node_config (NodeConfig): Node configuration dictionary.
        """
        super().__init__(root, name, type, node_config)

        # Initialize parameters
        self.data = data
        self.param_translators: List[BaseParamTranslator] = []
        self._init_param_translators()

        # Intialize I/O connectors
        self.inputs: Dict[str, Optional[str]] = {}
        self.outputs: List[str] = []
        self._init_io_connectors()

    def _init_param_translators(self):
        """Create parameter translators according to node configuration info.
        """
        self.param_translators.clear()

        # Skip if the node configuration doesn't specify any parameter info
        if not self.node_config.get('param'):
            return

        # Build lookup dictionary for parameter XML elements
        param_et_dict: Dict[str, ET.Element] = {}

        for param_et in self.data.iterfind('parameters/parameter'):
            param_et_dict[param_et.find('name').get('v')] = param_et

        # Read parameter translator configurations and construct translator objects
        for config in self.node_config['param']:

            # Get translator type and look up the factory for class name
            trans_type = config.get('type', 'default')
            trans_class: Type[BaseParamTranslator] = \
                getattr(param_trans, FACTORY_LUT['param_trans'][trans_type])

            # Retrieve associated parameter XML element
            root = param_et_dict.get(config['sbs_name'])

            # Create the parameter translator
            kwargs = config.copy()
            del kwargs['type']

            self.param_translators.append(trans_class(root, **kwargs))

    def _init_io_connectors(self):
        """Initialize input and output connectors from node configuraiton info.
        """
        # Initialize the dictionary of input connectors
        if self.node_config.get('input'):
            self.inputs = {n: None for n in self.node_config['input'].values()}

    def translate(self, **obj_kwargs) -> BaseFXMapNode:
        """Translate XML into a differentiable FX-map graph node object.

        Args:
            obj_kwargs (Dict[str, Any], optional): Keyword arguments that are passed to both the
                FX-map graph node object and the `translate` method of parameter translators
                (e.g., device ID).

        Returns:
            BaseFXMapNode: Translated FX-map graph node object.
        """
        # Instantiate parameters from parameter translators
        params: List[BaseParameter] = [pt.translate(**obj_kwargs) for pt in self.param_translators]

        # Generate a node object based on the type information
        node_class = CLASS_FACTORY[self.type]

        return node_class(self.name, self.type, params=params, inputs=self.inputs,
                          outputs=self.outputs, **obj_kwargs)


class FXMapGraphTranslator(BaseGraphTranslator[FXMapNodeTranslator]):
    """Translator of XML into a differentiable FX-map graph.
    """
    def __init__(self, root: ET.Element):
        """Initialize the FX-map graph translator.

        Args:
            root (Element): Root XML element of the FX-map graph.
        """
        # Index to the output node in the node (translator) list
        self.output_node_index: Optional[int] = None

        # Invoke the generic graph translation routine
        super().__init__(root)

    def _init_graph(self):
        """Build a graph data structure from the XML tree, which is a dictionary from node UIDs to
        basic node information (type, connectivity).

        This step ignores node functionalities or parameters.
        """
        self.graph.clear()

        # Identify output node UID
        output_uid = int(get_value(self.root.find('rootnode')))

        # Scan all graph nodes
        for node_et in self.root.iter('paramsGraphNode'):

            # Retrieve UID and check if the node is output
            node_uid = int(node_et.find('uid').get('v'))
            node_is_output = node_uid == output_uid

            # Get inward connections. Each connection is represented by a pair
            # (input name, target node output UID)
            node_in: List[Tuple[str, int]] = []

            for node_in_et in find_connections(node_et):
                node_in_name = node_in_et.find('identifier').get('v')
                node_in_ref = int(get_value(node_in_et.find('connRef')))
                node_in.append((node_in_name, node_in_ref))

            # Record node information
            node_data = {'is_output': node_is_output, 'in': node_in}

            self.graph[node_uid] = node_data

    def _init_node_translators(self):
        """Create FX-map graph node translators from node records in the XML tree.
        """
        self.node_translators.clear()
        self.node_name_allocator.reset()

        # Build lookup dictionary for graph node XML elements
        node_et_dict: Dict[int, ET.Element] = {}

        for node_et in self.root.iter('paramsGraphNode'):
            node_et_dict[int(node_et.find('uid').get('v'))] = node_et

        # Build lookup dictionary for graph node data (i.e., node parameter) XML elements
        data_et_dict: Dict[int, ET.Element] = {}

        for data_et in self.root.iter('paramsGraphData'):
            data_et_dict[int(data_et.find('uid').get('v'))] = data_et

        # Scan graph nodes
        for node_uid, node_data in self.graph.items():
            node_et = node_et_dict[node_uid]

            # Identify node function type
            node_type = node_et.find('type').get('v')
            if node_type not in CLASS_FACTORY:
                raise NotImplementedError(f'Unsupported FX-map node type: {node_type}')

            # Find the associated node data (parameter info)
            data_uid = int(node_et.find('data').get('v'))
            data_et = data_et_dict[data_uid]

            # Load node configuration from file
            node_config = load_node_config(node_type, mode='fxmap')

            # Construct a function node translator
            node_name = self.node_name_allocator.get_name(node_type)

            trans = FXMapNodeTranslator(
                node_et, data_et, node_name, node_type, node_config)
            self.node_translators.append(trans)

            # Record the output node index
            if node_data['is_output']:
                self.output_node_index = len(self.node_translators) - 1

    def _init_graph_connectivity(self):
        """Initialize graph connectivity by filling in input and output node connections.
        """
        # Build a mapping from function node UIDs to translators
        trans_dict = {int(t.root.find('uid').get('v')): t for t in self.node_translators}

        # Process I/O connections for each node
        for uid, trans in trans_dict.items():
            node_in: List[Tuple[str, int]] = self.graph[uid]['in']

            # For each input connection, add an input record to this node translator and add an
            # output record to the referenced translator
            for name, ref in node_in:

                # Translate the input connector name to diffmat
                input_name = trans.node_config['input'][name]

                # Obtain the referenced translator
                ref_trans = trans_dict[ref]

                # Add the connection
                trans.inputs[input_name] = ref_trans.name
                ref_trans.outputs.append(trans.name)

    def translate(self, **obj_kwargs) -> FXMapGraph:
        """Translate XML into a differentiable function graph (value processor) object.

        Args:
            obj_kwargs (Dict[str, Any], optional): Keyword arguments that are passed to both the
                FX-map graph object and the `translate` method of node translators
                (e.g., device ID).

        Returns:
            FXMapGraph: Translated FX-map graph object.
        """
        # Invoke node generators to produce a list of function graph nodes
        nodes = [trans.translate(**obj_kwargs) for trans in self.node_translators]

        return FXMapGraph(nodes, nodes[self.output_node_index], **obj_kwargs)


class FXMapTranslator(MaterialNodeTranslator):
    """Translator of XML into a differentiable FX-map filter node.
    """
    def __init__(self, root: ET.Element, name: str, type: str, res: int, node_config: NodeConfig,
                 **kwargs):
        """Initialize the FX-map node translator.

        Args:
            root (Element): Root XML element of the FX-map node.
            name (str): FX-map node name.
            type (str): FX-map node type.
            res (int): Resolution of the FX-map node.
            node_config (NodeConfig): Node configuration dictionary.
            kwargs (Dict[str, Any], optional): Additional keyword arguments that are passed to the
                base class constructor.
        """
        super().__init__(root, name, type, res, node_config, **kwargs)

    def _init_node_function(self):
        """Create the FX-map graph translator for this FX-map filter node.

        FX-map is another special case inside material nodes. Its node function is driven by the
        internal FX-map graph and completed by an FX-map executor.
        """
        # Locate the XML subtree of the FX-map graph
        graph_et = self.root.find('.//paramsGraph')

        # Create the FX-map graph translator
        if graph_et is not None:
            self.node_func_translator = FXMapGraphTranslator(graph_et)
        else:
            self.node_func_translator = None

    def _init_io_connectors(self):
        """Initialize input and output connectors.

        Input connectors are detected from XML and a translated connector name is assigned to each
        slot.
        """
        # Generate the input slot configuration dictionary
        self.node_config['input'] = {
            'background': '', **gen_input_dict(self.root),
            'background': 'img_bg',
        }

        # Create I/O connectors using the default routine
        super()._init_io_connectors()

    def translate(self, seed: int = 0, **obj_kwargs) -> FXMap:
        """Translate XML into a pixel processor node object.

        Args:
            seed (int, optional): Random seed for the FX-map node. Defaults to 0.
            obj_kwargs (Dict[str, Any], optional): Keyword arguments that are passed to both the
                FX-map node object and the `translate` method of node parameter translators
                (e.g., device ID).

        Returns:
            FXMap: Translated FX-map node object.
        """
        # Instantiate parameters from parameter translators
        params: List[BaseParameter] = [pt.translate(**obj_kwargs) for pt in self.param_translators]

        # Translate the function graph
        func = None
        if self.node_func_translator:
            func = self.node_func_translator.translate(**obj_kwargs)

        return FXMap(self.name, self.type, self.res, func, params=params, seed=seed,
                     **self._node_kwargs(), **obj_kwargs)
