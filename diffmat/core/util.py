from typing import Union, Tuple, List, Dict, Iterable
import logging
import time

import torch as th
import numpy as np

from diffmat.core.types import Constant, ParamValue, FloatArray, IntArray


# Debug switch for enabling/disabling debugging-related contents
DEBUG = False

# Optional filter value constants
FILTER_OFF = -1
FILTER_NO = 0
FILTER_YES = 1


# Tensor conversion related helper functions
def to_tensor(a: Union[FloatArray, IntArray]) -> th.Tensor:
    return th.as_tensor(a, dtype=th.float32)

def to_numpy(a: Union[FloatArray, IntArray]) -> np.ndarray:
    return a.detach().cpu().numpy() if isinstance(a, th.Tensor) else np.asarray(a)

def to_const(a: Union[FloatArray, IntArray]) -> Constant:
    return a.detach().cpu().tolist() if isinstance(a, th.Tensor) else \
           a.tolist() if isinstance(a, np.ndarray) else a

def to_tensor_and_const(a: Union[FloatArray, IntArray]) -> Tuple[th.Tensor, Constant]:
    return to_tensor(a), to_const(a)

def to_tensor_and_numpy(a: Union[FloatArray, IntArray]) -> Tuple[th.Tensor, np.ndarray]:
    return to_tensor(a), to_numpy(a)


def check_output_dict(output_dict: Dict[str, th.Tensor]):
    """Verify an output image dictionary can be saved, namely all images are of proper dimensions.

    Args:
        output_dict (Dict[str, Tensor]): Dictionary of source images.

    Raises:
        TypeError: An image is not a PyTorch tensor.
        ValueError: An image is not a 2D/3D/4D tensor or doesn't have a correct number of channels
            (not in 1, 3, or 4).
        ValueError: The batch dimension (`dim=0`) of an image is not 1.
    """
    for name, img in output_dict.items():
        if not isinstance(img, th.Tensor):
            raise TypeError(f"Image '{name}' is not a PyTorch tensor")
        if img.ndim not in (2, 3, 4) or img.ndim >= 3 and img.shape[-3] not in (1, 3, 4):
            raise ValueError(f"Invalid shape from image '{name}': {list(img.shape)}")
        if img.ndim == 4 and img.shape[0] != 1:
            raise ValueError(f"The batch dimension of image '{name}' should be 1 but got "
                             f"{img.shape[0]}")


def check_arg_choice(arg: Constant, choices: Iterable[Constant], arg_name: str = 'arg'):
    """Verify an argument comes from a group of valid options.

    Args:
        arg (Constant): Source argument.
        choices (Iterable[Constant]): A sequence of valid argument choices.
        arg_name (str, optional): Source argument name. Defaults to 'arg'.

    Raises:
        ValueError: The source argument does not hold any value in the provided choices.
    """
    if arg not in choices:
        raise ValueError(f"Valid options for argument '{arg_name}' are {choices}, but got {arg}")


class Timer:
    """A context manager that times the code section inside.
    """
    # Default logger associated with the timer
    logger = logging.getLogger('diffmat.core')

    def __init__(self, header: str = '', log_level: str = 'info', unit: str = 'ms',
                 gpu_mode: bool = True):
        """Initialize the timer.

        Args:
            header (str): On-screen header of the timer (e.g., describing its content).
            log_level (str, optional): Message level of the timer ('info' or 'debug').
                Defaults to 'info'.
            unit (str, optional): Time unit ('ms' or 's'). Defaults to 'ms'.
            gpu_mode (bool, optional): Whether to use more accurate timing for GPU.
                Defaults to True.
        """
        # Check input validity
        check_arg_choice(log_level, ['info', 'debug'], arg_name='log_level')
        check_arg_choice(unit, ['ms', 's'], arg_name='unit')

        self.header = header
        self.log_level = log_level
        self.unit = unit
        self.gpu_mode = gpu_mode

        # Initialize duration time
        self.t_duration = -1.0

    def __enter__(self) -> 'Timer':
        """Start the timer upon context entry.
        """
        # GPU timing
        if self.gpu_mode:
            self.event_start = th.cuda.Event(enable_timing=True)
            self.event_end = th.cuda.Event(enable_timing=True)
            self.event_start.record()

        # CPU timing
        else:
            self.t_start = time.time()

        return self

    def __exit__(self, *_) -> bool:
        """Stop the timer and do not handle any exception.
        """
        unit, header, log_level = self.unit, self.header, self.log_level
        unit_ms = unit == 'ms'

        # GPU timing
        if self.gpu_mode:
            self.event_end.record()
            th.cuda.synchronize()
            t_duration = self.event_start.elapsed_time(self.event_end)
            t_duration = t_duration * 1e-3 if not unit_ms else t_duration

        # CPU timing
        else:
            self.t_end = time.time()
            t_duration = self.t_end - self.t_start
            t_duration = t_duration * 1e3 if unit_ms else t_duration

        self.t_duration = t_duration

        # Print measured time
        if header:
            getattr(self.logger, log_level)(f'{header}: {t_duration:.3f} {unit}')

        return False

    @property
    def elapsed(self) -> float:
        """Get the measured time consumption in seconds.

        Returns:
            float: Time consumption in seconds.
        """
        # The timer must be started before calling this method
        if self.t_duration < 0:
            raise RuntimeError('The timer has not been started')

        return self.t_duration


class OperandLevel:
    """Definition of operand categories (or levels) for dynamic code generation.

    Please refer to `diffmat/config/functions/add.yml` for a detailed description of the static
    type inference system.

    Static members:
        SCALAR (int): Flag of scalar values (float or int).
        VECTOR (int): Flag of vector values (list of floats or ints).
        TENSOR_1D (int): Flag of 1D PyTorch tensors (scalars or vectors; require gradient).
        TENSOR_3D_4D (int): Flag of 3D and 4D PyTorch tensors (per-pixel scalars or vectors; may
            require gradient).
        PROMOTION_FUNC_TEMPLATES (List[str]): Expression templates for promoting a value from
            one value category to the next.
    """
    SCALAR = 0          # Scalar: float | int
    VECTOR = 1          # Vector: List[float] | List[int]
    TENSOR_1D = 2       # 1D tensor: th.Tensor (ndim = 1)
    TENSOR_3D_4D = 3    # 3D/4D tensor: th.Tensor (ndim = 3, 4) for pixel processor only

    # Function templates for operand level promotion, i.e., convert an operand to a higher level
    PROMOTION_FUNC_TEMPLATES: List[str] = [
        '[{}]',                 # Scalar to vector
        '_t({})',               # Vector to 1D tensor
        '{}.view(1, 1, -1)',    # 1D tensor to 3D tensor
    ]

    @classmethod
    def get_level(cls, value: ParamValue) -> int:
        """Determine the value category (level) of an operand given its value. Return -1 if the
        operand value does not belong to any level.

        Note that a 0D tensor (scalar) also counts into 1D since an extra dimension will be added
        at runtime (see method `FunctionGraph.evaluate`).

        Args:
            value (ParamValue): Source operand value.

        Returns:
            int: Value category of the operand.
        """
        if isinstance(value, (float, int)):
            level = cls.SCALAR
        elif isinstance(value, list) and value and isinstance(value[0], (float, int)):
            level = cls.VECTOR
        elif isinstance(value, th.Tensor) and value.ndim in (0, 1):
            level = cls.TENSOR_1D
        elif isinstance(value, th.Tensor) and value.ndim in (3, 4):
            level = cls.TENSOR_3D_4D
        else:
            level = -1

        return level

    @classmethod
    def promote(cls, value: ParamValue, level: int,
                promotion_mask: List[bool] = [True, True, True],
                device: str = 'cpu') -> ParamValue:
        """Promote an operand to a target level. This function has no effect when the operand is
        already at or above the target level.

        Args:
            value (ParamValue): Source operand value.
            level (int): Target value category.
            promotion_mask (List[bool], optional): Switches for every level of promotion. Please
                refer to `diffmat/config/functions/add.yml` for a detailed explanation.
                Defaults to [True, True, True].
            device (str, optional): Device placement for the promoted operand (tensors only).
                Defaults to 'cpu'.

        Returns:
            ParamValue: The promoted operand.
        """
        value_level = cls.get_level(value)

        # Progressive promotion
        for i in range(max(0, value_level), min(3, level)):
            if not promotion_mask[i]:
                break
            elif value_level <= i and level > i:
                if i == 0:
                    value = [value]
                elif i == 1:
                    value = th.tensor(value, device=device)
                else:
                    value = th.atleast_3d(value)

        return value

    @classmethod
    def promote_expr(cls, expr: str, expr_level: int, target_level: int,
                     promotion_mask: List[bool] = [True, True, True]) -> str:
        """Promote an operand (in the form of string expression) to a target level. This function
        has no effect when the operand is already at or above the target level.

        Args:
            expr (str): Source operand expression.
            expr_level (int): Value category of the source operand.
            target_level (int): Target value category.
            promotion_mask (List[bool], optional): See `promotion_mask` in the `promote` method.
                Defaults to [True, True, True].

        Returns:
            str: Expression of the promoted operand.
        """
        templates = cls.PROMOTION_FUNC_TEMPLATES

        # Progressive promotion by applying templates
        for i in range(max(0, expr_level), min(3, target_level)):
            if not promotion_mask[i]:
                break
            elif expr_level <= i and target_level > i:
                expr = templates[i].format(expr)

        return expr


# Type aliases
OL = OperandLevel
