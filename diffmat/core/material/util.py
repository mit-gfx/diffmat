from typing import (
    Tuple, List, Dict, Sequence, Iterator, Iterable, Callable, Union, Optional, Any, TypeVar
)
import inspect
import functools
import time

import torch as th

from diffmat.core.types import ParamValue, IntParamValue
from diffmat.core.base import BaseParameter
from diffmat.core.util import FILTER_YES, to_const


# Define node function type variable
FT = TypeVar('FT', bound=Callable[..., Any])


def input_check(num_inputs: int, channel_specs: str = '', reduction: str = 'all',
                reduction_range: int = 0, class_method: bool = False) -> \
                    Callable[[FT], FT]:
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
                check_image_types(imgs_to_check, func_name, types=specs)

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
    for c in channel_specs:
        if c not in valid_channel_specs:
            raise ValueError(f'Unknown channel specification option: {c}')

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
            tensor_mask = [isinstance(pair[1], th.Tensor) for pair in args_to_check]
            imgs_to_check: List[Tuple[str, th.Tensor]] = \
                [pair for pair, bit in zip(args_to_check, tensor_mask) if bit]
            shapes_to_check = [(name, list(val.shape[2:])) for name, val in imgs_to_check]
            check_identical_values(
                shapes_to_check, func_name, val_description='spatial dimensions')

            # Validate against color channel specifications
            specs = ''.join([flag for flag, bit in zip(channel_specs, tensor_mask) if bit])

            if specs:
                ## Append with the last flag if the specs are shorter than the image list
                ## Trim the specs if they are longer than the image list
                len_specs, num_imgs = len(specs), len(imgs_to_check)
                if len_specs < num_imgs:
                    specs += channel_specs[-1] * (num_imgs - len_specs)
                elif len_specs > num_imgs:
                    specs = specs[:num_imgs]

                ## The number of channels of each non-empty input image must match the spec flag
                check_image_types(imgs_to_check, func_name, types=specs)

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


def check_image_types(imgs_to_check: List[Tuple[str, th.Tensor]], func_name: str,
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
        generator = ((pair, flag) for pair in imgs_to_check)
    else:
        generator = zip(imgs_to_check, types)

    for (name, val), flag in generator:
        if flag == 'c' and val.shape[1] not in (3, 4) or \
            flag == 'g' and val.shape[1] != 1:
            image_type = 'color' if flag == 'c' else 'grayscale'
            raise ValueError(f"Input '{name}' to function '{func_name}' must be a "
                                f"{image_type} image but have {val.shape[1]} channels")

    ## Images of 'c' or '-' flags must have identical numbers of channels
    if apply_to_all:
        if types[0] in 'c-':
            channels_to_check = [(name, img.shape[1]) for name, img in imgs_to_check]
            check_identical_values(
                channels_to_check, func_name, val_description='numbers of channels')
    else:
        for flag in 'c-':
            channels_to_check = \
                [(name, img.shape[1]) for (name, img), c in generator if c == flag]
            check_identical_values(
                channels_to_check, func_name, val_description='numbers of channels')


def get_parameters(src: Iterable[BaseParameter], filter_requires_grad: int = FILTER_YES,
                   detach: bool = False, flatten: bool = False) -> Iterator[th.Tensor]:
    """Iterate over a list of parameters and return the tensor views of optimizable parameters.
    """
    # Return tensor views of filtered parameters
    for param in (p for p in src if p.IS_OPTIMIZABLE):
        data: Optional[th.Tensor] = param.data
        if data is not None and \
            (filter_requires_grad < 0 or filter_requires_grad == data.requires_grad):
            data = data.detach() if detach else data
            data = data.view(-1) if flatten else data
            yield data


def get_parameters_as_config(src: Iterable[BaseParameter], constant: bool = False) -> \
        Dict[str, Dict[str, ParamValue]]:
    """Return the values of a parameter sequence as a dict-type configuration in the
    following format:
    ```
    {param_name}: # x many
        value: {param_value}
        normalize: {False/True} # optional for optimizable parameters
    """
    _c = lambda v: to_const(v) if constant else v
    return {p.name: {'value': _c(p.evaluate()), 'normalize': True}
            for p in src if p.IS_OPTIMIZABLE and p.data is not None}


def set_parameters_from_config(src: Iterable[BaseParameter],
                               config: Dict[str, Dict[str, ParamValue]]):
    """Set the values of a parameter sequence from a nested dict-type configuration in the
    following format:
    ```
    {param_name}: # x many
        value: {param_value}
        normalize: {False/True} # optional for optimizable parameters
    ```

    Args:
        config (Dict[str, Dict[str, ParamValue]]): Parameter configuration as outlined above.
    """
    # Build a parameter name-to-object dictionary
    param_dict = {p.name: p for p in src if p.IS_OPTIMIZABLE and p.data is not None}

    for param_name, param_config in config.items():
        param_dict[param_name].set_value(
            param_config['value'], normalize=param_config.get('normalize', False))


def get_integer_parameters(src: Iterable[BaseParameter]) -> Iterator[IntParamValue]:
    """Iterate over a list of parameters and return the values of optimizable integer
    parameters.
    """
    for param in (p for p in src if p.IS_OPTIMIZABLE_INTEGER):
        yield param.evaluate()


def num_integer_parameters(src: Iterable[BaseParameter]) -> int:
    """Count the number of optimizable integer parameters in the material graph.
    """
    return sum(1 if isinstance(val, int) else len(val) for val in get_integer_parameters(src))


def set_integer_parameters_from_list(src: Iterable[BaseParameter], values: List[int]):
    """Set the optimizable integer parameter values among a parameter sequence from an integer
    list.
    """
    # Make sure the input parameter list matches the numer of integer parameters
    num_params = num_integer_parameters(src)
    if len(values) != num_params:
        raise ValueError(f'The length of the input list ({len(values)}) does not match '
                         f'the optimizable parameters ({num_params}) in the parameter sequence')

    # Extract parameters from the input list
    pos = 0

    for param in (p for p in src if p.IS_OPTIMIZABLE_INTEGER):
        val_length = 1 if isinstance(param.data, int) else len(param.data)
        param.set_value(values[pos:pos+val_length] if val_length > 1 else values[pos])
        pos += val_length


def get_integer_parameters_as_config(src: Iterable[BaseParameter]) -> \
        Dict[str, Dict[str, IntParamValue]]:
    """Return optimizable integer parameter values of the material node as a dict-type
    configuration in the following format:
    ```
    {param_name}: # x many
        value: {param_value}
        low: {param_low_bound}
        high: {param_high_bound}
    ```
    """
    return {p.name: {'value': p.evaluate(), 'low': p.scale[0], 'high': p.scale[1]}
            for p in src if p.IS_OPTIMIZABLE_INTEGER}


def set_integer_parameters_from_config(src: Iterable[BaseParameter],
                                       config: Dict[str, Dict[str, IntParamValue]]):
    """Set optimizable integer parameter values of the material node from a nested dict-type
    configuration in the following format:
    ```
    {param_name}: # x many
        value: {param_value}
    ```
    """
    # Build a parameter name-to-object dictionary
    param_dict = {p.name: p for p in src if p.IS_OPTIMIZABLE_INTEGER}

    for param_name, param_config in config.items():
        param_dict[param_name].set_value(param_config['value'])


def timed_func(func: Callable[..., Any], args: Sequence[Any], kwargs: Dict[str, Any],
               gpu_mode: bool = True) -> Tuple[Any, float]:
    """Time the execution of an arbitrary function using device-specific methods.

    Args:
        func (Callable[..., Any]): Function to be executed.
        args (Sequence[Any]): Positional arguments to the function.
        kwargs (Dict[str, Any]): Keyword arguments to the function.
        gpu_mode (bool, optional): Whether to use more accurate timing for functions running
            on GPUs. Defaults to True.

    Returns:
        Any: The return value of the function.
        float: The elapsed time in seconds.
    """
    # GPU timing
    if gpu_mode:
        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)

        start.record()
        ret = func(*args, **kwargs)
        end.record()

        th.cuda.synchronize()
        t_duration = start.elapsed_time(end) * 1e-3

    # CPU timing
    else:
        t_start = time.time()
        ret = func(*args, **kwargs)
        t_duration = time.time() - t_start

    return ret, t_duration
