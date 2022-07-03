from xml.etree import ElementTree as ET
from typing import Union, Optional, List, Tuple, Dict, Type

from .base import BaseNodeTranslator, BaseGraphTranslator
from .types import Constant, FunctionConfig
from .util import load_node_config, find_connections
from .util import get_param_type, get_value, get_param_value, to_constant
from .util import FUNCTION_CATEGORY_LUT, FACTORY_LUT
from ..core.function import FunctionNode, GetFunctionNode, RandFunctionNode
from ..core.function import FunctionGraph


class FunctionNodeTranslator(BaseNodeTranslator[FunctionConfig]):
    """Translator of XML to a differentiable function graph node.
    """
    def __init__(self, root: ET.Element, name: str, type: str, node_config: FunctionConfig):
        """Initialize the function graph node translator.

        Args:
            root (Element): Root node of the XML tree.
            name (str): Function node name.
            type (str): Function node operation type.
            node_config (FunctionConfig): Function node configuration info, such as function
                expression templates, input connectors, and parameter specifications.
        """
        super().__init__(root, name, type, node_config)

        # Initialize parameters
        self.params: Dict[str, Constant] = {}
        self._init_params()

        # Initialize I/O connectors
        self.inputs: Dict[str, Optional[str]] = {}
        self.outputs: List[str] = []
        self._init_io_connectors()

    def _init_params(self):
        """Create the parameter value dictionary according to node configuration information. This
        step is equivalent to an inline parameter translator.
        """
        self.params.clear()

        # Skip if the current node doesn't have parameters
        if not self.node_config.get('param'):
            return

        # Build parameter XML element lookup dictionary
        param_et_dict: Dict[str, ET.Element] = {}

        for param_et in self.root.iterfind('funcDatas/funcData'):
            param_name = param_et.find('name').get('v')
            param_et_dict[param_name] = param_et

        # Execute parameter translation rules in node configuration
        for param_trans in self.node_config['param']:

            # Get the parameter name in Substance designer using the 'sbs_name' field
            sbs_name: Union[str, List[str]] = param_trans['sbs_name']
            if isinstance(sbs_name, list):
                sbs_name = next((s for s in sbs_name if s in param_et_dict), '')

            # Retrieve parameter value from the XML subtree or use a default one
            param_et = param_et_dict.get(sbs_name)
            if param_et is not None:
                param_type = get_param_type(param_et)
                param_value = to_constant(get_param_value(param_et), param_type)
            else:
                param_value: Union[Constant, Dict[str, Constant]] = param_trans['default']
                if isinstance(param_value, dict):
                    param_value = param_value[self.type]

            # Register the parameter using the diffmat name
            param_name = param_trans['name']
            self.params[param_name] = param_value

    def _init_io_connectors(self):
        """Initialize input and output connectors from node configuraiton info.
        """
        if self.node_config.get('input'):
            name: str
            for name in self.node_config['input']:
                self.inputs[name] = None

    def translate(self, **obj_kwargs) -> FunctionNode:
        """Translate XML into a differentiable function graph node object.

        Args:
            obj_kwargs (Dict[str, Any], optional): Keyword arguments for the constructor of the
                translated function node object.

        Returns:
            FunctionNode: The translated function node object.
        """
        # Other keyword arguments to the node
        node_kwargs = {key: val for key, val in self.node_config.items() \
                       if key not in ('param', 'input')}

        return FunctionNode(self.name, params=self.params, inputs=self.inputs,
                            outputs=self.outputs, **node_kwargs, **obj_kwargs)


class DivFunctionNT(FunctionNodeTranslator):
    """Translator of XML to a 'div' function node. This is a special case since the operator
    depends on processed value type (float or integer).
    """
    def __init__(self, root: ET.Element, name: str, type: str, node_config: FunctionConfig):
        """Initialize the 'div' function node translator.

        Args:
            root (Element): Root node of the XML tree.
            name (str): Function node name.
            type (str): Function node operation type.
            node_config (FunctionConfig): Function node configuration info. See `node_config` in
                the constructor of `FunctionNodeTranslator`.
        """
        super().__init__(root, name, type, node_config)

    def _init_params(self):
        """After initializing parameters from node configuration. Determine the operator choice
        from the node's output type number.
        """
        # Deduce the operator as floor division if the output type number represents an integer
        # scalar or vector; true division otherwise
        output_type_num = int(get_value(self.root.find('type')))
        self.params['op'] = '//' if output_type_num < 256 else '/'


class GetFunctionNT(FunctionNodeTranslator):
    """Translator of XML to a 'get' function node.
    """
    def __init__(self, root: ET.Element, name: str, type: str, node_config: FunctionConfig):
        """Initialize the 'get' function node translator.

        Args:
            root (Element): Root node of the XML tree.
            name (str): Function node name.
            type (str): Function node operation type.
            node_config (FunctionConfig): Function node configuration info. See `node_config` in
                the constructor of `FunctionNodeTranslator`.
        """
        super().__init__(root, name, type, node_config)

    def _init_params(self):
        """Create the parameter value dictionary which contains the name of the variable that this
        node retrieves.
        """
        self.params.clear()

        # Search for the target variable name in XML subtree
        param_et = self.root.find(f"funcDatas/funcData/name[@v='{self.type}']/..")
        self.params['name'] = get_param_value(param_et)

    def translate(self, **obj_kwargs) -> GetFunctionNode:
        """Translate XML into a 'get' function node object.

        Args:
            obj_kwargs (Dict[str, Any], optional): Keyword arguments for the constructor of the
                translated function node object.

        Returns:
            GetFunctionNT: The translated function node object.
        """
        return GetFunctionNode(self.name, params=self.params, inputs=self.inputs,
                               outputs=self.outputs, **obj_kwargs)


class RandFunctionNT(FunctionNodeTranslator):
    """Translator of XML to a 'rand' function node.
    """
    def __init__(self, root: ET.Element, name: str, type: str, node_config: FunctionConfig):
        """Initialize the 'rand' function node translator.

        Args:
            root (Element): Root node of the XML tree.
            name (str): Function node name.
            type (str): Function node operation type.
            node_config (FunctionConfig): Function node configuration info. See `node_config` in
                the constructor of `FunctionNodeTranslator`.
        """
        super().__init__(root, name, type, node_config)

    def translate(self, **obj_kwargs) -> RandFunctionNode:
        """Translate XML into a 'rand' function node object.

        Args:
            obj_kwargs (Dict[str, Any], optional): Keyword arguments for the constructor of the
                translated function node object.

        Returns:
            RandFunctionNT: The translated function node object.
        """
        return RandFunctionNode(self.name, inputs=self.inputs, outputs=self.outputs, **obj_kwargs)


class FunctionGraphTranslator(BaseGraphTranslator[FunctionNodeTranslator]):
    """Translator of XML to a differentiable value processor (or a dynamic parameter value).
    """
    def __init__(self, root: ET.Element, name: str):
        """Initialize the function graph translator.

        Args:
            root (Element): Root node of the XML tree.
            name (str): Function graph name, usually identical to the name of the parameter defined
                by this graph.
        """
        self.name = name

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
        for node_et in self.root.iter('paramNode'):

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
        """Create function node translators from node records in the XML tree.

        Raises:
            NotImplementedError: Unsupported function node type.
        """
        self.node_translators.clear()
        self.node_name_allocator.reset()

        # Build lookup dictionary for graph node XML elements
        node_et_dict: Dict[int, ET.Element] = {}

        for node_et in self.root.iter('paramNode'):
            node_et_dict[int(node_et.find('uid').get('v'))] = node_et

        # Scan graph nodes
        for node_uid, node_data in self.graph.items():
            node_et = node_et_dict[node_uid]

            # Identify node function type
            node_type = node_et.find('function').get('v')

            # Raise type error if the node is not supported
            if node_type not in FUNCTION_CATEGORY_LUT:
                raise NotImplementedError(f'Unsupported function node type: {node_type}')

            # Query the look-up table for node category and load node configuration
            node_category = FUNCTION_CATEGORY_LUT[node_type]

            if node_category == 'default':
                node_config = load_node_config(node_type, mode='function')
            elif node_category != 'get':
                node_config = load_node_config(node_category, mode='function')
            else:
                node_config = {}

            # Construct a function node translator
            node_name = self.node_name_allocator.get_name(node_type)

            trans_class: Type[FunctionNodeTranslator] = \
                globals()[FACTORY_LUT['function_trans'][node_category]]
            trans = trans_class(node_et, node_name, node_type, node_config)
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
            for input_name, ref in node_in:

                # Obtain the referenced translator
                ref_trans = trans_dict[ref]

                # Add the connection
                trans.inputs[input_name] = ref_trans.name
                ref_trans.outputs.append(trans.name)

    def translate(self, **obj_kwargs) -> FunctionGraph:
        """Translate XML into a differentiable function graph (value processor) object.

        Args:
            obj_kwargs (Dict[str, Any], optional): Keyword arguments that will be passed to the
                translated function graph object and the `translate` method of function node
                translators in this graph.

        Returns:
            FunctionGraph: The translated graph object.
        """
        # Invoke node generators to produce a list of function graph nodes
        nodes = [trans.translate(**obj_kwargs) for trans in self.node_translators]

        return FunctionGraph(nodes, nodes[self.output_node_index], self.name, **obj_kwargs)
