from xml.etree import ElementTree as ET
from typing import Dict, List, Tuple, Sequence, Optional, Type, Union

import torch as th

from . import param_trans
from .base import BaseNodeTranslator, BaseParamTranslator
from .types import NodeFunction, NodeConfig
from .util import FACTORY_LUT
from ..core import functional
from ..core.functional import resize_image_color
from ..core.base import BaseParameter
from ..core.node import MaterialNode, ExternalInputNode


class MaterialNodeTranslator(BaseNodeTranslator[NodeConfig]):
    """Translator of an XML subtree to a differentiable material graph node.
    """
    def __init__(self, root: ET.Element, name: str, type: str, res: int, node_config: NodeConfig):
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
        super().__init__(root, name, type, node_config)

        # Node output resolution (after log2)
        self.res = res

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
            else:
                raise RuntimeError(f'Node function not found: {func_type}')

    def _init_param_translators(self):
        """Create parameter translators according to node configuration info.
        """
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
            trans_class: Type[BaseParamTranslator] = \
                getattr(param_trans, FACTORY_LUT['param_trans'][trans_type])

            # Retrieve associated parameter XML element
            root = param_et_dict.get(config['sbs_name'])

            # Create the parameter translator
            kwargs = config.copy()
            del kwargs['type']

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

        return MaterialNode(self.name, self.res, self.node_func, params=params, inputs=self.inputs,
                            outputs=self.outputs, seed=seed, **obj_kwargs)


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
        return ExternalInputNode(self.name, self.res, self.node_func, inputs=self.inputs,
                                 outputs=self.outputs, seed=seed, **obj_kwargs)
