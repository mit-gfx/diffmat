from xml.etree import ElementTree as ET
from pathlib import PurePath, Path
from typing import Dict, List, Union, Optional, Iterator, Any
import itertools

import yaml

from diffmat.translator import types as tp
from diffmat.translator.types import Constant, PathLike, NodeConfig, FunctionConfig


# ----------------------------------- #
#          Parameter related          #
# ----------------------------------- #

def is_image(type: int) -> bool:
    """Check if a parameter type refers to color or grayscale images.

    Args:
        type (int): Value type specifier. See 'Type numbers' in `diffmat/translator/types.py`.

    Returns:
        bool: Whether the type defines a color or grayscale image.
    """
    return type in (tp.COLOR, tp.GRAYSCALE)


def is_optimizable(type: int) -> bool:
    """Check if a parameter type indicates continuous optimization capability (i.e., it must
    contain float values exclusively).

    Args:
        type (int): Value type specifier. See 'Type numbers' in `diffmat/translator/types.py`.

    Returns:
        bool: Whether the type represents an optimizable continuous parameter (i.e., a float or a
            floating-point vector).
    """
    return type in (tp.FLOAT, tp.FLOAT2, tp.FLOAT3, tp.FLOAT4)


def is_integer_optimizable(type: int) -> bool:
    """Check if a parameter type indicates integer optimization capability (i.e., it must
    contain integer values exclusively).

    Args:
        type (int): Value type specifier. See 'Type numbers' in `diffmat/translator/types.py`.

    Returns:
        bool: Whether the type represents an optimizable integer parameter (i.e., an integer or a
            vector of integers).
    """
    return type in (tp.INT, tp.INT2, tp.INT3, tp.INT4)


def get_value(node: Optional[ET.Element], default: str = '') -> str:
    """Return the 'v' value of an XML node; if the node is None, return a default string.

    Args:
        node (Optional[Element]): XML tree node.
        default (str, optional): Default return value when the node is None or does not have a
            'v' attribute. Defaults to ''.

    Returns:
        str: Value of the 'v' attribute, or the default string.
    """
    if node is None:
        value = default
    elif node.get('v') is not None:
        value = node.get('v')
    elif node.find('value') is not None:
        value = node.find('value').get('v')
    else:
        value = default
    return value


def get_param_type(node: ET.Element) -> int:
    """Obtain the type info inside an XML parameter record.

    Args:
        node (Element): XML subtree root of the node parameter.

    Returns:
        int: Node parameter type specifier.
    """
    # Ignore parameter arrays as they don't have a type
    if node.tag == 'paramsArray':
        return tp.OPTIONAL

    # Locate the parameter value XML subtree
    value_et = (node.find('defaultValue') or node.find('paramValue') or \
                node.find('constantValue'))[0]

    # Ignore dynamic values as their types are determined by functions
    value_tag = value_et.tag
    if value_tag == 'dynamicValue':
        return tp.DYNAMIC

    # Recognize value type from tag name
    return PARAM_TYPE_LUT[value_tag]


def get_param_value(node: ET.Element, check_dynamic: bool = False) -> \
        Optional[Union[str, List[Dict[str, str]]]]:
    """Obtain the value string inside an XML parameter record, optionally reporting an error for
    dynamic parameter values.

    Args:
        node (Element): XML subtree root of the node parameter.
        check_dynamic (bool, optional): When set to True, raise a `ValueError` when the node
            parameter holds a dynamic value (i.e., defined by a function graph).

    Raises:
        ValueError: The node parameter is dynamic when `check_dynamic` is set.

    Returns:
        Optional[str | List[Dict[str, str]]]: Parameter value in string format. For parameter
            arrays, return a list of dictionaries that record individual parameter array cell info.
    """
    # For dynamic values
    if node.find('.//dynamicValue') is not None:
        if check_dynamic:
            raise ValueError("Please use dynamic parameter translator for dynamic values.")
        return None

    # Locate the parameter value XML subtree and obtain parameter value in string
    if node.tag != 'paramsArray':
        value_et = (node.find('defaultValue') or node.find('paramValue') or \
                    node.find('constantValue'))[0]
        value = get_value(value_et)

    # Obtain parameter value from an array of cells (each cell may hold one or more parameters) so
    # this function is called recursively
    else:
        value: List[Dict[str, str]] = []
        for cell_et in node.iterfind('paramsArrayCells/paramsArrayCell'):
            cell: Dict[str, str] = {}
            for param_et in cell_et.iter('parameter'):
                param_name = param_et.find('name').get('v')
                param_value = get_param_value(param_et, check_dynamic=True)
                cell[param_name] = param_value
            value.append(cell)

    return value


def lookup_value_type(value: Constant) -> int:
    """Analyze the type number of a constant value.
    
    Args:
        value (Constant): Constant value.

    Raises:
        ValueError: The input value is of an unknown type.

    Returns:
        int: Type specifier of the input value. See 'Type numbers' in
            `diffmat/translator/types.py`.
    """
    type: Optional[int] = None

    if isinstance(value, bool):
        type = tp.BOOL
    elif isinstance(value, int):
        type = tp.INT
    elif isinstance(value, float):
        type = tp.FLOAT
    elif isinstance(value, str):
        type = tp.STR
    elif isinstance(value, list) and len(value) in (2, 3, 4):
        if isinstance(value[0], int):
            type = (tp.INT2, tp.INT3, tp.INT4)[len(value) - 2]
        elif isinstance(value[0], float):
            type = (tp.FLOAT2, tp.FLOAT3, tp.FLOAT4)[len(value) - 2]
        elif isinstance(value[0], list) and isinstance(value[0][0], float):
            type = tp.OPTIONAL

    if type is None:
        raise ValueError(f"Unrecognized type for constant value '{value}'")
    return type


def to_constant(value_str: str, type: int) -> Constant:
    """Convert a parameter value string to a numerical constant.

    Args:
        value_str (str): Parameter value in string format.
        type (int): Parameter value type specifier.

    Raises:
        ValueError: Unknown parameter type specifier.

    Returns:
        Constant: Parameter value in numerics.
    """
    def int32(x: str) -> int:
        int_x, p32 = int(x), 2 ** 31
        return int_x % p32 - int_x // p32 * p32

    if type == tp.BOOL:
        value = bool(int(value_str))
    elif type == tp.INT:
        value = int32(value_str)
    elif type in (tp.INT2, tp.INT3, tp.INT4):
        value = [int32(c) for c in value_str.strip().split()]
    elif type == tp.FLOAT:
        value = float(value_str)
    elif type in (tp.FLOAT2, tp.FLOAT3, tp.FLOAT4):
        value = [float(c) for c in value_str.strip().split()]
    elif type == tp.STR:
        value = value_str
    else:
        raise ValueError(f'Unrecognized parameter type: {type}')

    return value


def to_str(value: Constant, type: int) -> str:
    """Convert a constant parameter value to string.

    Args:
        value (Constant): Parameter value in numerical format.
        type (int): Parameter value type specifier.

    Raises:
        ValueError: Unknown parameter type specifier.

    Returns:
        str: Parameter value in string format.
    """
    if type == tp.BOOL:
        value_str = str(int(value))
    elif type == tp.INT:
        value_str = str(value)
    elif type in (tp.INT2, tp.INT3, tp.INT4):
        value_str = ' '.join([str(v) for v in value])
    elif type == tp.FLOAT:
        value_str = f'{value:.9f}'
    elif type in (tp.FLOAT2, tp.FLOAT3, tp.FLOAT4):
        value_str = ' '.join([f'{v:.9f}' for v in value])
    elif type == tp.STR:
        value_str = value
    else:
        raise ValueError(f'Unrecognized parameter type: {type}')

    return value_str


# ------------------------------ #
#          Node related          #
# ------------------------------ #

class NameAllocator:
    """Node name allocator.
    """
    def __init__(self):
        """Initialize the allocator counter.
        """
        self.counter: Dict[str, int] = {}

    def get_name(self, node_type: str) -> str:
        """Allocate a name for a translated node.

        Args:
            node_type (str): Source node type.

        Returns:
            str: Allocated node name.
        """
        # Create a new record for the given node type
        # The node name is designated to be the same as its type
        if node_type not in self.counter:
            self.counter[node_type] = 1
            name = f'{node_type}_0'

        # The allocated node name is f'{node_type}_{counter_val}'
        # Increment the counter for the given node type
        else:
            name = f'{node_type}_{self.counter[node_type]}'
            self.counter[node_type] += 1

        return name

    def reset(self):
        """Reset the allocator counter.
        """
        self.counter.clear()


def load_config(filename: PathLike) -> Any:
    """Read a configuration file in YAML format.

    Args:
        filename (PathLike): Path to the source YAML file.

    Returns:
        Any: YAML file content.
    """
    with open(filename, 'r') as f:
        return yaml.safe_load(f)


def load_node_config(node_type: str, mode: str = 'node') -> \
        Union[NodeConfig, FunctionConfig]:
    """Read a node configuration file.

    Args:
        node_type (str): Material/function node type. All supported node types are listed in
            `diffmat/config/node_list.yml`.
        mode (str, optional): Determines which folder to load configuration from
            ('node', 'function', or 'generator'). Defaults to 'node'.

    Raises:
        FileNotFoundError: Configuration file is not found for the given node type.

    Returns:
        NodeConfig | FunctionConfig: Material/function node configuration.
    """
    # Search for loaded configurations first
    node_config_label = f'{mode}_{node_type}'
    if node_config_label in LOADED_CONFIGS:
        return LOADED_CONFIGS[node_config_label]

    # Directory names
    dir_name = f'{mode}s' if mode in ('node', 'function', 'generator') else mode

    # Check if the configuration file exists
    node_config_path = Path(CONFIG_DIR / dir_name / f'{node_type}.yml')
    if not node_config_path.exists():
        raise FileNotFoundError(f'Configuration file not found for {mode} type: '
                                f'{node_type}')

    # Load from the config file and inject internal node parameters info
    config: NodeConfig = load_config(node_config_path)
    if mode in ('node', 'generator'):
        param_config = NODE_INTERAL_PARAMS['param'].copy()
        param_config.extend(config.get('param') or [])
        config['param'] = param_config

    # Remember the loaded config for later references 
    LOADED_CONFIGS[node_config_label] = config

    return config


def gen_category_lut(config: Dict[str, Union[str, List[str]]]) -> Dict[str, str]:
    """Invert the node category dictionary into a look-up table.

    Args:
        config (Dict[str, str | List[str]]): Node category dictionary that lists supported node
            types in each category.

    Returns:
        Dict[str, str]: An inverted dictionary (reversed mapping) for node category lookup.
    """
    lut: Dict[str, str] = {}
    for key, val in config.items():
        if isinstance(val, str):
            lut[val] = key
        else:
            lut.update({val: key for val in val})
    return lut


def has_connections(node: ET.Element) -> bool:
    """Examine if a graph node has input connections. This function is for backward compatibility
    since older graphs use 'connexions' as the tag name instead of 'connections'.

    Args:
        node (Element): XML subtree root of the material node.

    Returns:
        bool: Whether XML data contains input connection info.
    """
    return node.find('connections') is not None or node.find('connexions') is not None


def find_connections(node: ET.Element) -> Iterator[ET.Element]:
    """Return an iterator over the input connections to a graph node. This function is for backward
    compatibility since older graphs use 'connexions' as the tag name instead of 'connections'.

    Args:
        node (Element): XML subtree root of the material node.

    Yields:
        Iterator[Element]: An iterator over the input connection entries of the material node.
    """
    return itertools.chain(node.iterfind('connections/connection'),
                           node.iterfind('connexions/connexion'))


def gen_input_dict(node: ET.Element, sort: bool = True) -> Dict[str, str]:
    """Generate the input slot configuration of a graph node by extracting its sequence of
    connections. Optionally sort the input connections by slot names.
    """
    # Collect input connector names
    input_names = [conn.find('identifier').get('v') for conn in find_connections(node)]
    input_names = sorted(sorted(input_names), key=len) if sort else input_names

    # Allocate input connectors and assign translated names
    return {name: name.replace(':', '_') for name in input_names}


# Global configuration folder
CONFIG_DIR = PurePath(__file__).parents[1] / 'config'

# Dictionary of loaded configurations
LOADED_CONFIGS: Dict[str, NodeConfig] = {}

# Internal material node parameters
NODE_INTERAL_PARAMS: NodeConfig = load_config(CONFIG_DIR / 'nodes' / 'internal.yml')

# Factory look-up table
FACTORY_LUT: Dict[str, Dict[str, str]] = load_config(CONFIG_DIR / 'factory.yml')

# Node and function category look-up table
NODE_CATEGORY_LUT: Dict[str, str] = \
    gen_category_lut(load_config(CONFIG_DIR / 'node_list.yml'))
FUNCTION_CATEGORY_LUT: Dict[str, str] = \
    gen_category_lut(load_config(CONFIG_DIR / 'function_list.yml'))

# Parameter type look-up table
PARAM_TYPE_LUT: Dict[str, int] = load_config(CONFIG_DIR / 'param_types.yml')
