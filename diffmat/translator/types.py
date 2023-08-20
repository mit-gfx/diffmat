from typing import Union, List, Tuple, Dict, Optional

from diffmat.core.types import \
    Constant, ParamValue, IntParamValue, NodeFunction, PathLike, DeviceType


# Type aliases
NodeData = Dict[str, Union[bool, str,
    Dict[int, str], List[Union[Tuple[str, int, int], Tuple[str, int]]]
]]
NodeConfig = Dict[str, Optional[Union[str, Dict[str, str], List[Dict[str, Constant]]]]]
FunctionConfig = Dict[str, Optional[Union[
    List[str], List[bool], List[Dict[str, Union[Constant, List[str], Dict[str, Constant]]]]
]]]


# Type numbers
DYNAMIC = -1
OPTIONAL = 0
COLOR = 1
GRAYSCALE = 2
BOOL = 4
INT = 16
INT2 = 32
INT3 = 64
INT4 = 128
FLOAT = 256
FLOAT2 = 512
FLOAT3 = 1024
FLOAT4 = 2048
STR = 16384
