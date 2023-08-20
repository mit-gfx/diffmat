from xml.etree import ElementTree as ET
from typing import Dict, List, Tuple, Sequence, Optional, Type, Union

import torch as th

from diffmat.core.base import BaseParameter
from diffmat.core.material import functional, noise
from diffmat.core.material import MaterialNode, ExternalInputNode
from diffmat.core.operator import resize_image_color
from diffmat.core.util import check_arg_choice
from diffmat.translator import param_trans
from diffmat.translator.base import BaseNodeTranslator, BaseParamTranslator
from diffmat.translator.types import NodeFunction, NodeConfig, Constant
from diffmat.translator.util import FACTORY_LUT, gen_input_dict


class MaterialNodeTranslator(BaseNodeTranslator[NodeConfig]):
    """Translator of an XML subtree to a differentiable material graph node.
    """
    def __init__(self, root: ET.Element, name: str, type: str, res: int, node_config: NodeConfig,
                 integer_option: str = 'integer'):
        """Initialize the material graph node translator using a source XML subtree and
        configuration information (read from external files).

        Args:
            root (Element): Root node of the XML tree.
            name (str): Material node name.
            type (str): Material node operation type, equivalent to the name of the node function
                in most cases.
            res (int): Output texture resolution (after log2).
            node_config (NodeConfig): Material node configuration info, such as node function name,
                I/O connectors, and parameter specifications.
        """
        # Validate the integer option parameter
        check_arg_choice(integer_option, ['integer', 'constant'], 'integer_option')

        super().__init__(root, name, type, node_config)

        self.res = res
        self.integer_option = integer_option

        # Initialize node function
        self.node_func: NodeFunction = functional.passthrough
        self._init_node_function()

        # Initialize parameter translators
        self.param_translators: List[BaseParamTranslator] = []
        self._init_param_translators()

        # Initialize I/O connectors
        self.inputs: Dict[str, Optional[Tuple[str, str]]] = {}
        self.outputs: Dict[str, List[str]] = {}
        self._init_io_connectors()

        # Helper function for keyword arguments in node constructors
        self._node_kwargs = lambda: {
            'inputs': self.inputs,
            'outputs': self.outputs,
            'allow_ablation': self.node_config.get('allow_ablation', False),
            'is_generator': self.node_config.get('is_generator', False)
        }

    def _init_node_function(self):
        """Load node function according to node configuration info.

        Raises:
            RuntimeError: Node function is not found in `functional.py`.
        """
        # Get node function from core.functional or core.noise
        if 'func' in self.node_config:
            func_type = self.node_config['func']
            if hasattr(functional, func_type):
                self.node_func: NodeFunction = getattr(functional, func_type)
            elif hasattr(noise, func_type):
                self.node_func: NodeFunction = getattr(noise, func_type)
            else:
                raise RuntimeError(f'Node function not found: {func_type}')

    def _init_param_translators(self):
        """Create parameter translators according to node configuration info.
        """
        integer_option = self.integer_option
        self.param_translators.clear()

        # Skip if the node configuration doesn't specify any parameter info
        if not self.node_config.get('param'):
            return

        # Get node implementation element
        node_imp_et = self.root.find('compImplementation')[0]

        # Build lookup dictionary for parameter XML elements
        param_et_dict: Dict[str, ET.Element] = {}

        for param_et in node_imp_et.iterfind('parameters/parameter'):
            param_et_dict[param_et.find('name').get('v')] = param_et
        for param_et in node_imp_et.iterfind('paramsArrays/paramsArray'):
            param_et_dict[param_et.find('name').get('v')] = param_et

        # Read parameter translator configurations and construct translator objects
        for config in self.node_config['param']:

            # Get translator type and look up the factory for class name
            trans_type = config.get('type', 'default')

            # Retrieve associated parameter XML element
            root = param_et_dict.get(config['sbs_name'])

            # Delete irrelevant entries in parameter config
            kwargs: Dict[str, Constant] = config.copy()
            del kwargs['type']

            # Re-classify integer parameters and remove irrelevant entries according to user option
            if trans_type == 'integer':
                trans_type = integer_option
                kwargs.pop('quantize', None)
                if trans_type == 'constant':
                    kwargs.pop('scale', None)

            # Create the parameter translator
            trans_class: Type[BaseParamTranslator] = \
                getattr(param_trans, FACTORY_LUT['param_trans'][trans_type])
            self.param_translators.append(trans_class(root, **kwargs))

    def _init_io_connectors(self):
        """Initialize input and output connectors from node configuration info.
        """
        # Initialize the dictionary of input connectors
        if self.node_config.get('input'):
            self.inputs = {n: None for n in self.node_config['input'].values()}

        # Initialize the dictionary of output connectors
        if self.node_config.get('output'):
            self.outputs = {n: [] for n in self.node_config['output'].values()}
        else:
            self.outputs = {'': []}

    def translate(self, seed: int = 0, **obj_kwargs) -> MaterialNode:
        """Translate XML into a differentiable material graph node object.

        Args:
            seed (int, optional): Random seed of the material node. Defaults to 0.
            obj_kwargs (Dict[str, Any], optional): Keyword arguments that are passed to both the
                material node object and the `translate` method of parameter translators
                (e.g., device ID).

        Returns:
            MaterialNode: Translated material node object.
        """
        # Instantiate parameters from parameter translators
        params: List[BaseParameter] = []
        for pt in self.param_translators:
            param: Union[BaseParameter, Sequence[BaseParameter]] = pt.translate(**obj_kwargs)
            params.append(param) if isinstance(param, BaseParameter) else params.extend(param)

        # Create the material node 
        return MaterialNode(self.name, self.type, self.res, self.node_func, params=params,
                            seed=seed, **self._node_kwargs(), **obj_kwargs)


class ExternalInputNT(MaterialNodeTranslator):
    """Translator for material graph nodes that read from external inputs.
    """
    def __init__(self, root: ET.Element, name: str, type: str, res: int, node_config: NodeConfig,
                 **kwargs):
        """Initialize the material graph node translator from a source XML subtree and
        configuration information (actually unused for external input nodes).

        Args:
            root (Element): Root node of the XML tree.
            name (str): External input node name.
            type (str): External input node type, equivalent to the name of the node function
                in most cases.
            res (int): Output texture resolution (after log2).
            node_config (NodeConfig): Unused.
        """
        super().__init__(root, name, type, res, node_config, **kwargs)

    def _init_node_function(self):
        """Set the node function as reading from the external input dictionary.
        """
        # Define the node function which retrieves output maps from external inputs
        def node_func(external_input_dict: Dict[str, th.Tensor], use_alpha: bool = False) -> \
            Union[th.Tensor, Tuple[Optional[th.Tensor]]]:

            # Read all output channels of this node
            # Adjust any color image inputs to account for the use of alpha channel
            output_tensors: List[th.Tensor] = []
            for output_name in self.node_config['output'].values():
                output_name = f"{self.name}{'_' if output_name else ''}{output_name}"
                img = external_input_dict.get(output_name)
                if img is not None and img.shape[1] > 1:
                    img = resize_image_color(img, 3 + use_alpha)
                output_tensors.append(img)

            return tuple(output_tensors) if len(output_tensors) > 1 else output_tensors[0]

        self.node_func = node_func

    def translate(self, seed: int = 0, **obj_kwargs) -> ExternalInputNode:
        """Translate XML into a external input node object.

        Args:
            seed (int, optional): Random seed of the external input node. Defaults to 0.
            obj_kwargs (Dict[str, Any], optional): Keyword arguments that are passed to the
                instantiated external input node object.

        Returns:
            MaterialNode: Translated external input node object.
        """
        return ExternalInputNode(self.name, self.type, self.res, self.node_func, seed=seed,
                                 **self._node_kwargs(), **obj_kwargs)


class DummyNodeTranslator(MaterialNodeTranslator):
    """Translator that substitutes material graph nodes with 'dummy' pass-through nodes.
    """
    def __init__(self, root: ET.Element, name: str, type: str, res: int, node_config: NodeConfig,
                 **kwargs):
        """Initialize the dummy material graph node translator from a source XML subtree and
        relevant configuration information. Node config info is still used to construct the I/O
        connectors.

        Args:
            root (Element): Root node of the XML tree.
            name (str): dummy node name.
            type (str): dummy node operation type, equivalent to the name of the node function
                in most cases.
            res (int): Output texture resolution (after log2).
            node_config (NodeConfig): Original node configuration info, only I/O connectors are
                concerned.
        """
        # Delete parameter info from node configuration
        node_config = node_config.copy()
        node_config.pop('param', None)

        super().__init__(root, name, type, res, node_config, **kwargs)

    def _init_node_function(self):
        """Set the node function as passthrough with a pre-specified number of outputs.
        """
        output_config = self.node_config.get('output') or {'': ''}
        self.node_func = functional.passthrough_template(len(output_config))

    def _init_io_connectors(self):
        """Initialize input and output connectors from node configuration info.
        """
        # Generate custom input slot configuration for Pixel Processor and FX-Map nodes
        if self.type == 'pixelprocessor':
            self.node_config['input'] = gen_input_dict(self.root)
        elif self.type == 'fxmaps':
            self.node_config['input'] = {'background': 'img_bg', **gen_input_dict(self.root)}

        # Create I/O connectors using the default routine
        super()._init_io_connectors()
