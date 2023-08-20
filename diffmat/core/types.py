from typing import Union, Optional, Tuple, List, Dict, Callable, Iterator
import os

import torch as th
import numpy as np
import numpy.typing as npt


# Type aliases
Constant = Union[float, int, bool, str, List[int], List[float]]
ParamValue = Union[Constant, th.Tensor]
IntParamValue = Union[int, List[int]]
FloatValue = Union[float, th.Tensor]
FloatVector = Union[List[float], npt.NDArray[np.floating], th.Tensor]
IntVector = Union[List[int], npt.NDArray[np.integer], th.Tensor]
FloatArray = Union[float, List[float], List[List[float]], npt.NDArray[np.floating], th.Tensor]
IntArray = Union[int, List[int], npt.NDArray[np.integer], th.Tensor]
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
ParamConfig = Dict[str, Dict[str, Dict[str, ParamValue]]]
FXMapJobArray = \
    Dict[str, Union[List[int], List[str], List[FloatValue], List[FloatArray], FloatVector,
                    IntVector]]
FXMapNodeGenerator = Iterator[Tuple[str, npt.NDArray[np.float32], FloatVector]]
JobArrayFunction = \
    Callable[[int, th.Tensor, th.Tensor, th.Tensor], Tuple[Optional[FXMapJobArray], th.Tensor]]
DeviceType = Union[str, th.device]

# For generic typing
ConstantDict = Dict[str, Constant]
InputDict = Dict[str, Optional[str]]
MultiInputDict = Dict[str, Optional[Tuple[str, str]]]
OutputList = List[str]
MultiOutputDict = Dict[str, List[str]]
