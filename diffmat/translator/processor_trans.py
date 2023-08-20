from xml.etree import ElementTree as ET
from typing import List

from diffmat.core.base import BaseParameter
from diffmat.core.material import PixelProcessor
from diffmat.translator.node_trans import MaterialNodeTranslator
from diffmat.translator.function_trans import FunctionGraphTranslator
from diffmat.translator.types import NodeConfig
from diffmat.translator.util import gen_input_dict


class PixelProcessorTranslator(MaterialNodeTranslator):
    """Translator of XML to a pixel processor node.
    """
    def __init__(self, root: ET.Element, name: str, type: str, res: int, node_config: NodeConfig,
                 **kwargs):
        """Initialize the pixel processor node translator.

        Args:
            root (Element): Root element of the XML tree.
            name (str): Pixel Processor node name.
            type (str): Pixel Processor node type.
            res (int): Resolution of the node.
            node_config (NodeConfig): Node configuration.
            kwargs (Dict[str, Any], optional): Additional keyword arguments to be passed to the
                base class constructor. Defaults to {}.
        """
        super().__init__(root, name, type, res, node_config, **kwargs)

    def _init_node_function(self):
        """Create a funcion graph translator for the per-pixel function.
        
        Unlike most material nodes, the node function of a pixel processor is defined by a function
        graph (value processor) so its implementation can not be imported from 'functional'.
        """
        # Locate the XML subtree of the per-pixel funciton
        func_et = self.root.find(".//parameter/name[@v='perpixel']/..//dynamicValue")

        # Create the function graph translator
        if func_et is not None:
            self.node_func_translator = \
                FunctionGraphTranslator(func_et, 'perpixel', per_pixel=True)
        else:
            self.node_func_translator = None

    def _init_io_connectors(self):
        """Initialize input and output connectors.

        Input connectors are detected from XML and a translated connector name is assigned to each
        slot.
        """
        # Generate the input slot configuration dictionary
        self.node_config['input'] = gen_input_dict(self.root)

        # Create I/O connectors using the default routine
        super()._init_io_connectors()

    def translate(self, seed: int = 0, **obj_kwargs) -> PixelProcessor:
        """Translate XML into a pixel processor node object.

        Args:
            seed (int, optional): Random seed for the Pixel Processor node. Defaults to 0.
            obj_kwargs (Dict[str, Any], optional): Keyword arguments that are passed to both the
                Pixel Processor node object and the `translate` method of node parameter translators
                (e.g., device ID).

        Returns:
            PixelProcessor: Pixel Processor node object.
        """
        # Instantiate parameters from parameter translators
        params: List[BaseParameter] = [pt.translate(**obj_kwargs) for pt in self.param_translators]

        # Translate the function graph
        func = None
        if self.node_func_translator:
            func = self.node_func_translator.translate(**obj_kwargs)

        return PixelProcessor(self.name, self.type, self.res, func, params=params, seed=seed,
                              **self._node_kwargs(), **obj_kwargs)
