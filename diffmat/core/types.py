from typing import Union, Optional, Tuple, List, Dict, Callable
import os

import torch as th
import numpy as np
import numpy.typing as npt


# Type aliases
Constant = Union[float, int, bool, str, List[int], List[float]]
ParamValue = Union[Constant, th.Tensor]
FloatValue = Union[float, th.Tensor]
FloatVector = Union[List[float], npt.NDArray[np.float32], th.Tensor]
FloatArray = Union[List[float], List[List[float]], float, th.Tensor, npt.NDArray[np.float32]]
OffsetType = Union[Tuple[float, float], th.Tensor]
Instruction = Dict[str, Union[str, List[Optional[str]]]]
NodeFunction = Callable[..., Union[th.Tensor, Tuple[th.Tensor, ...]]]
PatternFunction = \
    Callable[[th.Tensor, Union[float, th.Tensor]], Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]]
FunctionTemplates = List[Callable[..., str]]
PathLike = Union[str, bytes, os.PathLike]
ParamSummary = Dict[str, Union[str, Constant]]
NodeSummary = Dict[str, Union[str, List[Optional[str]], Dict[str, Constant]]]
GraphSummary = Dict[str, Union[str, NodeSummary, Dict[str, Constant]]]
DeviceType = Union[str, th.device]

# For generic typing
ConstantDict = Dict[str, Constant]
InputDict = Dict[str, Optional[str]]
MultiInputDict = Dict[str, Optional[Tuple[str, str]]]
OutputList = List[str]
MultiOutputDict = Dict[str, List[str]]
