from typing import Union, Tuple, List, Dict, Iterable, TypeVar, Callable, Any
import inspect
import functools

import torch as th

from .types import Constant, ParamValue


# Define node function type variable
FT = TypeVar('FT', bound=Callable[..., Any])

# Debug switch for enabling/disabling debugging-related contents
DEBUG = True


def input_check(num_inputs: int, channel_specs: str = '', reduction: str = 'all',
                reduction_range: int = 0, class_method: bool = False) -> Callable[[FT], FT]:
    """Node function decorator that checks whether a certain number of inputs are 4D torch tensors.

    The `channel_specs` option defines whether inputs are interpreted as color or gray
    images. Images of the same type are optionally required to have matching numbers of channels
    (along dimension 1). See the 

    The `reduction` option specifies if all/any of those inputs must be a tensor.

    Args:
        num_inputs (int): Number of input tensors to inspect. Note that it could be more than
            the number of positional arguments in a node function. In that case, the first several
            keyword arguments will be inspected as well.
        channel_specs (str, optional): Specifies the type of each input tensor, i.e., whether it
            should represent a color or a grayscale image. The following individual characters are
            allowed in the string:
                `.`: No constraint (either color or grayscale).
                `c`: Color image. All color images must have matching numbers of channels (size of
                    tensor along `dim=1`)
                `g`: Grayscale image.
                `-`: Either color or grayscale, but all images must have matching numbers of
                    channels.
            The channel specification string should not be longer than `num_inputs`. If the string
            is shorter, missing characters are considered as `.`'s.
            Defaults to ''.
        reduction (str, optional): Specifies if 'all' or 'any' of the first several inputs to the
            node function must be a tensor. Defaults to 'all'.
        reduction_range (int, optional): Number of inputs in the reduction range. Defaults to 0.
        class_method (bool, optional): Whether the wrapped function is a class method (that
            receives an additional leading `self` or `cls` argument). Defaults to False.

    Raises:
        ValueError: Unknown reduction mode.
        ValueError: Channel specification string is longer than the number of checked inputs.
        ValueError: Unknown channel specifiction option (invalid character).
        TypeError: 'all' reduction check fails as one input tensor is left empty.
        TypeError: 'any' reduction check fails as no input tensor is found.

    Returns:
        Decorator: The node function decorator.
    """
    # Verify valid reduction modes
    if reduction not in ('all', 'any'):
        raise ValueError(f'Unrecognized reduction mode: {reduction}')

    # Verify color channel specifications
    valid_channel_specs = 'cg.-'
    if len(channel_specs) > num_inputs:
        raise ValueError(f"Channel specification string '{channel_specs}' is longer than the "
                         f"number of inputs to check ({num_inputs})")
    for c in channel_specs:
        if c not in valid_channel_specs:
            raise ValueError(f'Unknown channel specification option: {c}')

    def decorator(func: FT) -> FT:
        @functools.wraps(func)
        def wrapper(*args: ParamValue, **kwargs: ParamValue) -> Union[th.Tensor, Tuple[th.Tensor]]:

            # Create the mapping from arguments to values and extract the first 'num_inputs' values
            bound_args = inspect.signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()
            start_idx = int(class_method > 0)
            args_to_check = list(bound_args.arguments.items())[start_idx: start_idx + num_inputs]

            # Check whether non-empty inputs are 4D torch tensors
            func_name = func.__name__
            check_4d_tensors(args_to_check, func_name)

            # Check whether non-empty inputs have identical shapes (last two dimensions only)
            tensor_mask = [isinstance(pair[1], th.Tensor) for pair in args_to_check]
            imgs_to_check: List[Tuple[str, th.Tensor]] = \
                [pair for pair, bit in zip(args_to_check, tensor_mask) if bit]
            shapes_to_check = [(name, list(img.shape[2:])) for name, img in imgs_to_check]
            check_identical_values(
                shapes_to_check, func_name, val_description='spatial dimensions')

            # Validate against color channel specifications
            specs = ''.join([flag for flag, bit in zip(channel_specs, tensor_mask) if bit])

            if specs:
                ## The number of channels of each input image must match the corresponding flag in
                ## spec string
                check_image_types(imgs_to_check, func_name, types=specs)

                ## Images of 'c' or '-' flags must have identical numbers of channels
                for flag in 'c-':
                    iterator = zip(imgs_to_check, specs)
                    channels_to_check = \
                        [(name, img.shape[1]) for (name, img), c in iterator if c == flag]
                    check_identical_values(
                        channels_to_check, func_name, val_description='numbers of channels')

            # Perform reduction in the specified range and report errors if any
            if reduction_range > 0:
                flags = [val is not None for _, val in args_to_check[:reduction_range]]

                ## 'all' means all arguments in the checking range must be 4D tensors
                if reduction == 'all' and not all(flags):
                    failed_name = args_to_check[flags.index(False)][0]
                    raise TypeError(f"Empty input '{failed_name}' to function '{func_name}'")

                ## 'any' means at least one argument in the checking range must be a 4D tensor
                elif reduction == 'any' and not any(flags):
                    raise TypeError(f"Function '{func_name}' did not receive input among the "
                                    f"first {num_inputs} input arguments")

            # Run the node function
            return func(*args, **kwargs)

        return wrapper
    return decorator


def input_check_all_positional(class_method: bool = False, channel_specs: str = '') -> \
        Callable[[FT], FT]:
    """A special decorator that checks whether all positional arguments of a function are 4D torch
    tensors.

    Args:
        class_method (bool, optional): Whether the wrapped function is a class method (that
            receives an additional leading `self` or `cls` argument). Defaults to False.
        channel_specs (str, optional): See `channel_specs` in the `input_check` function.
            Defaults to ''.

    Raises:
        ValueError: Unknown channel specification option (invalid character).

    Returns:
        Decorator: The node function decorator.
    """
    # Verify color channel specifications
    valid_channel_specs = 'cg.-'
    if len(channel_specs) >= 2 or channel_specs not in valid_channel_specs:
        raise ValueError(f'Unknown channel specification option: {channel_specs}')

    def decorator(func: FT) -> FT:
        @functools.wraps(func)
        def wrapper(*args: th.Tensor, **kwargs: ParamValue) -> Union[th.Tensor, Tuple[th.Tensor]]:

            # Create the mapping from arguments to values and extract the variadic positional
            # argument
            bound_args = inspect.signature(func).bind(*args, **kwargs)
            start_idx = int(class_method > 0)
            img_args = bound_args.args[start_idx:]
            args_to_check = [(f'img_{i}', img) for i, img in enumerate(img_args)]

            # Check whether the positional arguments are 4D torch tensors
            func_name = func.__name__
            check_4d_tensors(args_to_check, func_name)

            # Check whether non-empty inputs have identical shapes (last two dimensions only)
            imgs_to_check: List[Tuple[str, th.Tensor]] = \
                [pair for pair in args_to_check if isinstance(pair[1], th.Tensor)]
            shapes_to_check = [(name, list(val.shape[2:])) for name, val in imgs_to_check]
            check_identical_values(
                shapes_to_check, func_name, val_description='spatial dimensions')

            # Validate against color channel specifications
            flag = channel_specs

            if flag:
                ## The number of channels of each non-empty input image must match the spec flag
                check_image_types(imgs_to_check, func_name, types=flag, apply_to_all=True)

                ## For 'c' or '-' flags, input images must have identical numbers of channels
                if flag in 'c-':
                    channels_to_check = [(name, val.shape[1]) for name, val in imgs_to_check]
                    check_identical_values(
                        channels_to_check, func_name, val_description='numbers of channels')

            # Run the node function
            return func(*args, **kwargs)

        return wrapper
    return decorator


def color_input_check(img: th.Tensor, var_name: str):
    """Verify a tensor input represents a color image.

    Args:
        img (Tensor): Source tensor input.
        var_name (str): Argument name of the tensor.

    Raises:
        ValueError: The input tensor does not represent a color image.
    """
    if not isinstance(img, th.Tensor) or img.ndim != 4 or img.shape[1] not in (3, 4):
        raise ValueError(f"Node function input '{var_name}' must be a color image but have "
                         f"{img.shape[1]} channels")


def grayscale_input_check(img: th.Tensor, var_name: str):
    """Verify a tensor input represents a grayscale image.

    Args:
        img (Tensor): Source tensor input.
        var_name (str): Argument name of the tensor.

    Raises:
        ValueError: The input tensor does not represent a grayscale image.
    """
    if not isinstance(img, th.Tensor) or img.ndim != 4 or img.shape[1] != 1:
        raise ValueError(f"Node function input '{var_name}' must be a grayscale image but "
                         f"have {img.shape[1]} channels")


def check_4d_tensors(vals_to_check: List[Tuple[str, Any]], func_name: str):
    """Verify a series of named input arguments are either 4D PyTorch tensors or none. Raises
    an error when an exception is found.

    Args:
        vals_to_check (List[Tuple[str, Any]]): A dictionary of input arguments by argument name,
            flattened into a list of key-value pairs.
        func_name (str): Name of the node function where the checking happens.

    Raises:
        TypeError: An input is non-empty but not a PyTorch tensor.
        ValueError: An input is a PyTorch tensor but without a 4D shape.
    """
    for name, val in vals_to_check:
        if val is not None:
            if not isinstance(val, th.Tensor):
                raise TypeError(f"Input '{name}' to function '{func_name}' must be a "
                                f"PyTorch tensor but got '{type(val).__name__}'")
            elif val.ndim != 4:
                raise ValueError(f"Input '{name}' to function '{func_name}' must be a "
                                 f"4D PyTorch tensor but got shape {list(val.shape)}")


def check_image_types(vals_to_check: List[Tuple[str, th.Tensor]], func_name: str,
                      types: str = '', apply_to_all: bool = False):
    """Verify a group of images conform to given type ('c'olor or 'g'rayscale) specifications.
    Raises an error if there is any type violation.

    Args:
        vals_to_check (List[Tuple[str, Any]]): A dictionary of input images by argument name,
            flattened into a list of key-value pairs.
        func_name (str): Name of the node function where the checking happens.
        types (str, optional): Channel specification string. See `channel_specs` in the
            `input_check` function. Defaults to ''.
        apply_to_all (bool, optional): If set to True, only the first character of the `types`
            string will be used and the option will apply to all images. Otherwise, each input
            image will match a character in the `types` string consecutively. Defaults to False.

    Raises:
        ValueError: Image type check fails, i.e., an input image doesn't match its type specifier.
    """
    # Generator that produces data-type pairs
    if apply_to_all:
        flag = types[0] if len(types) else '.'
        generator = ((pair, flag) for pair in vals_to_check)
    else:
        generator = zip(vals_to_check, types)

    for (name, val), flag in generator:
        if flag == 'c' and val.shape[1] not in (3, 4) or \
            flag == 'g' and val.shape[1] != 1:
            image_type = 'color' if flag == 'c' else 'grayscale'
            raise ValueError(f"Input '{name}' to function '{func_name}' must be a "
                                f"{image_type} image but have {val.shape[1]} channels")


def check_identical_values(vals_to_check: List[Tuple[str, Any]], func_name: str,
                           val_description: str = 'values'):
    """Verify a series of named input arguments are identical. Raises an error if a mismatch
    is found.

    Args:
        vals_to_check (List[Tuple[str, Any]]): A dictionary of input arguments by argument name,
            flattened into a list of key-value pairs.
        func_name (str): Name of the node function where the checking happens.
        val_description (str, optional): A brief text description of the checked value's meaning.
            Only plurrals are grammatically correct. Defaults to 'values'.

    Raises:
        ValueError: Found two differing input arguments.
    """
    if len(vals_to_check) > 1:
        arg_name, val = vals_to_check[0]
        fail_pair = next((pair for pair in vals_to_check[1:] if pair[1] != val), None)
        if fail_pair:
            raise ValueError(f"Input '{arg_name}' and '{fail_pair[0]}' to function '{func_name}' "
                             f"have mismatched {val_description} ({val} and {fail_pair[1]})")


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


class OperandLevel:
    """Definition of operand categories (or levels) for dynamic code generation.

    Please refer to `diffmat/config/functions/add.yml` for a detailed description of the static
    type inference system.

    Static members:
        SCALAR (int): Flag of scalar values (float or int).
        VECTOR (int): Flag of vector values (list of floats or ints).
        TENSOR_1D (int): Flag of 1D PyTorch tensors (scalars or vectors; require gradient).
        PROMOTION_FUNC_TEMPLATES (List[str]): Expression templates for promoting a value from
            one value category to the next.
    """
    SCALAR = 0          # Scalar: float | int
    VECTOR = 1          # Vector: List[float] | List[int]
    TENSOR_1D = 2       # 1D tensor: th.Tensor (ndim = 1)

    # Function templates for operand level promotion, i.e., convert an operand to a higher level
    PROMOTION_FUNC_TEMPLATES: List[str] = [
        '[{}]',                 # Scalar to vector
        '_t({})',               # Vector to 1D tensor
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
        else:
            level = -1

        return level

    @classmethod
    def promote(cls, value: ParamValue, level: int, promotion_mask: List[bool] = [True, True],
                device: str = 'cpu') -> ParamValue:
        """Promote an operand to a target level. This function has no effect when the operand is
        already at or above the target level.

        Args:
            value (ParamValue): Source operand value.
            level (int): Target value category.
            promotion_mask (List[bool], optional): Switches for every level of promotion. Please
                refer to `diffmat/config/functions/add.yml` for a detailed explanation.
                Defaults to [True, True].
            device (str, optional): Device placement for the promoted operand (tensors only).
                Defaults to 'cpu'.

        Returns:
            ParamValue: The promoted operand.
        """
        value_level = cls.get_level(value)

        # Progressive promotion
        for i in range(max(0, value_level), min(2, level)):
            if not promotion_mask[i]:
                break
            elif value_level <= i and level > i:
                if i == 0:
                    value = [value]
                elif i == 1:
                    value = th.tensor(value, device=device)

        return value

    @classmethod
    def promote_expr(cls, expr: str, expr_level: int, target_level: int,
                     promotion_mask: List[bool] = [True, True]) -> str:
        """Promote an operand (in the form of string expression) to a target level. This function
        has no effect when the operand is already at or above the target level.

        Args:
            expr (str): Source operand expression.
            expr_level (int): Value category of the source operand.
            target_level (int): Target value category.
            promotion_mask (List[bool], optional): See `promotion_mask` in the `promote` method.
                Defaults to [True, True].

        Returns:
            str: Expression of the promoted operand.
        """
        templates = cls.PROMOTION_FUNC_TEMPLATES

        # Progressive promotion by applying templates
        for i in range(max(0, expr_level), min(2, target_level)):
            if not promotion_mask[i]:
                break
            elif expr_level <= i and target_level > i:
                expr = templates[i].format(expr)

        return expr


# Type aliases
OL = OperandLevel
