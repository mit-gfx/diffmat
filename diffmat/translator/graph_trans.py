from xml.etree import ElementTree as ET
from typing import Union, Optional, List, Tuple, Dict, Type

from .base import BaseGraphTranslator, BaseNodeTranslator, BaseParamTranslator
from .external_input import ExtInputGenerator
from .node_trans import MaterialNodeTranslator, ExternalInputNT
from .param_trans import ConstantParamTranslator, ParamTranslator
from .types import PathLike, DeviceType
from .util import is_image, is_optimizable, get_value, get_param_value
from .util import find_connections, load_node_config
from .util import NODE_CATEGORY_LUT, FACTORY_LUT
from ..core.base import BaseParameter
from ..core.render import Renderer
from ..core.graph import MaterialGraph


# Class factory dictionary
CLASS_FACTORY: Dict[str, Type[BaseNodeTranslator]] = {
    'MaterialNodeTranslator': MaterialNodeTranslator,
    'ExternalInputNT': ExternalInputNT,
}


class MaterialGraphTranslator(BaseGraphTranslator[MaterialNodeTranslator]):
    """Translator of XML to a differentiable material graph.
    """
    def __init__(self, root: Union[PathLike, ET.Element], res: int, external_noise: bool = True,
                 toolkit_path: Optional[PathLike] = None):
        """Initialize the material graph translator using a source XML file or ElementTree root
        node.

        Args:
            root (PathLike | Element): Path to the source XML file, or a root node of the XML tree.
            res (int): Output texture resolution (after log2).
            external_noise (bool, optional): When set to True, noises and parameters are generated
                externally using Substance Automation Toolkit. Otherwise, they are generated from
                Diffmat implementations. Defaults to True.
            toolkit_path (Optional[PathLike], optional): Path to the executables of Substance
                Automation Toolkit. Passing None prompts the translator to use a OS-specific
                default location (see `external_input.py`). Defaults to None.
        """
        self.res = res
        self.external_noise = external_noise

        # Invoke the generic graph translation routine
        super().__init__(root)

        # Initialize exposed parameter translators from XML root
        self.exposed_param_translators: List[BaseParamTranslator] = []
        self._init_exposed_param_translators()

        # Get the graph name
        graph_name: Optional[str] = self.root.find('content/graph/identifier').get('v')
        if graph_name is None:
            self.graph_name = 'substance_graph'
        else:
            self.graph_name = graph_name.lower().replace(' ', '_')

        # Create an external input generator
        self.input_generator = ExtInputGenerator(self.root, res, toolkit_path=toolkit_path)

    def _init_graph(self):
        """Build a graph data structure from the XML tree, which is a dictionary from node UIDs to
        basic node information (type, connectivity).

        This step ignores node functionalities or parameters.

        Raises:
            RuntimeError: The material graph does not have any supported output SVBRDF channel.
        """
        self.graph.clear()

        # Scan supported graph outputs (basecolor, normal, roughness, metallic)
        graph_outputs: Dict[int, str] = {}

        for output_et in self.root.iter('graphoutput'):
            output_uid = int(output_et.find('uid').get('v'))
            output_group = get_value(output_et.find('group'))
            output_usage = get_value(output_et.find('usages/usage/name'))

            if output_group == 'Material' and output_usage.lower() in Renderer.CHANNELS:
                graph_outputs[output_uid] = output_usage.lower()

        if not graph_outputs:
            raise RuntimeError(f'The graph does not have any output channel in '
                               f'{Renderer.CHANNELS.keys()}')

        # Scan all graph nodes
        for node_et in self.root.iter('compNode'):

            # Retrieve UID and identify if the node is a supported graph output
            node_uid = int(node_et.find('uid').get('v'))
            node_imp_et = node_et.find('compImplementation')[0]
            node_is_output = False
            if node_imp_et.tag == 'compOutputBridge':
                output_uid = int(node_imp_et.find('output').get('v'))
                node_is_output = output_uid in graph_outputs
                node_output_usage = graph_outputs.get(output_uid, '')

            # Get outward connection slots (not all of them are actually connected)
            # The slots are organized using a dictionary from output slot UID to name
            node_out_uid_path = 'compOutputs/compOutput/uid'
            node_out_name_path = 'outputBridgings/outputBridging/identifier'
            node_out_uids = [int(e.get('v')) for e in node_et.iterfind(node_out_uid_path)]
            node_out_names = [e.get('v') for e in node_imp_et.iterfind(node_out_name_path)] or ['']
            node_out = dict(zip(node_out_uids, node_out_names))

            # Get inward connections. Each connection is represented by a triplet
            # (input name, target node UID, target node output UID)
            node_in: List[Tuple[str, int, int]] = []

            for node_in_et in find_connections(node_et):
                node_in_name = node_in_et.find('identifier').get('v')
                node_in_ref = int(get_value(node_in_et.find('connRef')))
                node_in_ref_output = int(get_value(node_in_et.find('connRefOutput'), '-1'))
                node_in.append((node_in_name, node_in_ref, node_in_ref_output))

            # Record node information
            node_data = {'is_output': node_is_output, 'out': node_out, 'in': node_in}
            if node_is_output:
                node_data['usage'] = node_output_usage

            self.graph[node_uid] = node_data

    def _init_node_translators(self):
        """Create node translators from material graph node records in the XML tree.

        Raises:
            RuntimeError: 'Input' atomic nodes are not allowed in the material graph.
            NotImplementedError: Unsupported material node type.
        """
        self.node_translators.clear()
        self.node_name_allocator.reset()

        # Build lookup dictionary for graph node XML elements
        node_et_dict: Dict[int, ET.Element] = {}

        for node_et in self.root.iter('compNode'):
            node_et_dict[int(node_et.find('uid').get('v'))] = node_et

        # Scan graph nodes
        for node_uid, node_data in self.graph.items():
            node_et = node_et_dict[node_uid]

            # Identify node type (input, output, filter, or generator)
            node_imp_et = node_et.find('compImplementation')[0]

            ## Input node - abort since the content is unknown
            if node_imp_et.tag == 'compInputBridge':
                raise RuntimeError('Input nodes are not allowed in differentiable material graphs')

            ## Output node - get output usage and create an output node translator
            elif node_imp_et.tag == 'compOutputBridge':
                node_type = 'output'

            ## Atomic filter or generator node
            elif node_imp_et.tag == 'compFilter':
                node_type = node_imp_et.find('filter').get('v')

            ## Non-atomic filter or generator node
            else:
                path = get_value(node_imp_et.find('path'))
                node_type = path[path.rfind('/') + 1: path.rfind('?')]

            # Raise error if the node is not supported and cannot be handled by SAT
            # Otherwise, query the look-up table for node category
            if node_type not in NODE_CATEGORY_LUT and node_data['in']:
                raise NotImplementedError(f'Unsupported node type: {node_type}')
            else:
                node_category = NODE_CATEGORY_LUT.get(node_type, 'generator')

            # Generate node configuration contingent on node category
            # For non-external nodes, load the configuration file of the associated node type from
            # the configuration folder
            if node_category in ('default', 'output'):
                node_config = load_node_config(node_type)

            # An ambiguous node category is resolved and replaced by 'default' or 'generator'
            # according to the number of input connections
            if node_category in ('dual', 'pixelprocessor', 'fxmap'):
                if node_data['in']:
                    node_category = 'default'
                    node_config = load_node_config(node_type)
                else:
                    node_category = 'generator'

            # For 'generator' nodes, resolve the node category into 'default' or 'external', which
            # determines whether a node will depend on external inputs
            if node_category == 'generator':
                if self.external_noise:
                    node_category = 'external'
                else:
                    node_category = 'default'
                    node_config = load_node_config(node_type)

            # For 'external' input nodes, override the default configuration using a proxy that
            # only specifies output connectors
            if node_category == 'external':
                node_config = {'output': {n: n.lower() for n in node_data['out'].values()}}

            # Construct a node translator based on node category
            node_name = node_data['usage'] if node_category == 'output' else \
                        self.node_name_allocator.get_name(node_type)

            trans_class: Type[MaterialNodeTranslator] = \
                CLASS_FACTORY[FACTORY_LUT['node_trans'][node_category]]
            trans = trans_class(node_et, node_name, node_type, self.res, node_config)
            self.node_translators.append(trans)

    def _init_exposed_param_translators(self):
        """Create exposed parameter translators from input parameter records in the XML tree.
        """
        self.exposed_param_translators.clear()

        # Iterate over all exposed parameter records
        for param in self.root.iter('paraminput'):

            # Create a parameter translator when the parameter is not an image;
            # otherwise, warn the user
            param_type = int(get_value(param.find('type')))
            if is_image(param_type):
                self.logger.warn('Images detected in exposed parameters. Ignored.')
                continue

            # Identify the parameter name and its default value
            param_trans_kwargs = {
                'sbs_name': param.find('identifier').get('v'),
                'sbs_default': get_param_value(param),
            }

            # For optimizable parameters, extract the scales
            if is_optimizable(param_type):
                trans_class = ParamTranslator
                scale_min_str = get_value(param.find(".//options/option/name[@v='min']/.."), '0')
                scale_max_str = get_value(param.find(".//options/option/name[@v='max']/.."), '1')
                param_trans_kwargs['scale'] = (float(scale_min_str), float(scale_max_str))

            else:
                trans_class = ConstantParamTranslator

            # Create and register the parameter translator
            param_trans = trans_class(param, **param_trans_kwargs)
            self.exposed_param_translators.append(param_trans)

    def _init_graph_connectivity(self):
        """Initialize graph connectivity by filling in input and output node connections.
        """
        # Build a mapping from node UIDs to translators
        trans_dict = {int(t.root.find('uid').get('v')): t for t in self.node_translators}

        # Process I/O connections for each node
        for uid, trans in trans_dict.items():
            node_in: List[Tuple[str, int, int]] = self.graph[uid]['in']

            # For each input connection, add an input record to this node translator and add an
            # output record to the referenced translator
            for name, ref, conn_ref in node_in:

                # Translate the input connector name to diffmat
                input_name = trans.node_config['input'][name]

                # Obtain the referenced translator
                ref_trans = trans_dict[ref]

                # Translate the output connector name of the referenced translator to diffmat
                ref_output_name = '' if conn_ref < 0 else self.graph[ref]['out'][conn_ref]
                if ref_trans.node_config['output']:
                    ref_output_name = ref_trans.node_config['output'][ref_output_name]

                # Add the connection
                trans.inputs[input_name] = (ref_trans.name, ref_output_name)
                ref_trans.outputs[ref_output_name].append(trans.name)

    def translate(self, seed: int = -1, use_alpha: bool = True,
                  normal_format: str = 'dx', external_input_folder: PathLike = '.',
                  device: DeviceType = 'cpu') -> MaterialGraph:
        """Translate XML into a differentiable material graph object.

        Args:
            seed (int, optional): Graph-wide random seed effective to all material nodes. Each
                material node has an individual random seed that serves as an additional offset to
                this global random seed. Defaults to -1.
            use_alpha (bool, optional): Enable alpha channel processing in the translated graph.
                Defaults to True.
            normal_format (str, optional): Normal format that the translated graph uses when
                rendering its output texture (DirectX 'dx' or OpenGL 'gl'). Defaults to 'dx'.
            external_input_folder (PathLike, optional): Target directory for storing all externally
                generated texture maps in the material graph, including noises, patterns, and
                linked/embedded images. Defaults to '.'.
            device (DeviceType, optional): The device where the material graph is placed (e.g.,
                CPU or GPU), per PyTorch device naming conventions. Defaults to 'cpu'.

        Raises:
            ValueError: Unknown normal format.

        Returns:
            MaterialGraph: Translated material graph object.
        """
        # Invoke node generators to produce a list of graph nodes (with established connectivity)
        kwargs = {'device': device, 'seed': max(seed, 0)}
        nodes = [trans.translate(**kwargs) for trans in self.node_translators]

        # Generate external input images
        external_input_trans = [trans for trans in self.node_translators
                                if isinstance(trans, ExternalInputNT)]
        external_inputs = \
            self.input_generator.process(
                external_input_trans, seed=seed, result_folder=external_input_folder,
                device=device)

        # Process exposed parameters
        exposed_params: List[BaseParameter] = \
            [trans.translate(device=device) for trans in self.exposed_param_translators]

        # Rendering parameters
        if normal_format not in ('dx', 'gl'):
            raise ValueError("Supported normal formats are 'dx' (DirectX) and 'gl' (OpenGL)")
        render_params = {'normal_format': normal_format}

        return MaterialGraph(nodes, self.graph_name, self.res,
                             external_inputs=external_inputs, exposed_params=exposed_params,
                             render_params=render_params, use_alpha=use_alpha, **kwargs)
