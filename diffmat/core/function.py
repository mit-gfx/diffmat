from functools import partial
from types import CodeType
from typing import List, Tuple, Dict, Optional, Union, Callable, Any
import inspect
import random
import math
import textwrap

import torch as th

from .base import BaseFunctionNode, BaseFunctionGraph
from .types import ParamValue, ConstantDict, InputDict, OutputList
from .util import OL


class FunctionNode(BaseFunctionNode):
    """Class for nodes inside a differentiable function graph.
    """
    def __init__(self, name: str, func: List[str], params: ConstantDict = {},
                 inputs: InputDict = {}, outputs: OutputList = [],
                 output_level: List[int] = [0, 1, 2],
                 promotion_mask: List[bool] = [False, False], **kwargs):
        """Initialize the function node.

        Args:
            name (str): Function node name.
            func (List[str]): Function expression templates at different levels. For a detailed
                definition of these templates, please refer to `diffmat/config/functions/add.yml`.
            params (ConstantDict, optional): Function node parameters. Defaults to {}.
            inputs (InputDict, optional): Mapping from input connectors to predecessor nodes that
                they connect to. Defaults to {}.
            outputs (OutputList, optional): List of succesor nodes, namely those who receive input
                from this node. Defaults to [].
            output_level (List[int], optional): Output value category (or level) corresponding to
                function expressions instantiated from templates in the `func` parameter. Please
                see `add.yml` for a more detailed explanation. Defaults to [0, 1, 2].
            promotion_mask (List[bool], optional): Operand level promotion mask. Please see
                `add.yml` for a more detailed explanation. Defaults to [False, False].
            kwargs (Dict[str, Any], optional): Keyword arguments to pass into the parent class
                constructor.
        """
        super().__init__(name, params=params, inputs=inputs, outputs=outputs, **kwargs)

        # Node function is defined as a string template. It must be instantiated using input
        # variable names, compiled, and then evaluated
        # Here, the template is wrapped inside a partial formatting function with node parameters
        # already applied. Other inputs will be provided by the user in runtime
        func: List[Callable[..., str]] = [partial(f.format, **params) for f in func]

        self.func = func
        self.output_level = output_level
        self.promotion_mask = promotion_mask

        # Cache compiled node expression
        self._init_func_signature()

    def _init_func_signature(self):
        """Manually create a signature of the node function for runtime input check.
        """
        # Construct parameters for the node function signature, which includes:
        #   - input variables (required)
        #   - expressions of input variables (optional)
        #   - other parameters accessible by 'get' operators
        param = inspect.Parameter
        flag = param.POSITIONAL_OR_KEYWORD
        parameters = [param(name, flag) for name in self.inputs]
        parameters.extend([param(f'{name}_expr', flag, default='') for name in self.inputs])

        self.sig = inspect.Signature(parameters)

    def evaluate(self, *args: ParamValue, return_expr: bool = False,
                 **kwargs: ParamValue) -> Union[ParamValue, Tuple[ParamValue, str]]:
        """Evaluate the node function given input operands, while deducing an expression form of
        the result based on optional input expressions of the operands.

        Args:
            args (Sequence[ParamValue]): Operand values or expressions.
            return_expr (bool, optional): Switch for returning an instantiated function expression.
                Defaults to False.
            kwargs (Dict[str, ParamValue], optional): Operand values or expressions.

        Returns:
            ParamValue: Function node output.
            str (optional): Function node expression.
        """
        # Bind the arguments of this function to the manually created node function signature
        # Bound arguments are used as a namespace to lookup input variables
        bound_args = self.sig.bind(*args, **kwargs).arguments

        # Read operands and determine max operand level
        operands = [bound_args[name] for name in self.inputs]
        operand_levels = [OL.get_level(val) for val in operands]
        max_level = max(operand_levels) if operand_levels else 0

        # Promote each operand to the max level
        promote: Callable[[str, int], str] = partial(
            OL.promote_expr, target_level=max_level, promotion_mask=self.promotion_mask)
        operand_exprs = [promote(*pair) for pair in zip(self.inputs, operand_levels)]

        # Define a tensor creation function by detecting the current device
        device = next((val.device for val in operands if isinstance(val, th.Tensor)), 'cpu')
        bound_args['_t'] = partial(th.as_tensor, device=device)

        # Apply promoted operands to the node function template to get a complete node expression
        node_func_format = self.func[max_level]
        node_expr = node_func_format(**operand_exprs)
        result: ParamValue = eval(node_expr, None, bound_args)

        if not return_expr:
            return result

        # Promote operand expressions to the max level and apply them to the node function template
        # This generates a statement in a code block where operands might be passed in as variables
        # or expressions
        operand_exprs = {name: promote(bound_args[f'{name}_expr'], level) \
                         for name, level in zip(self.inputs, operand_levels)}
        node_expr = node_func_format(**operand_exprs)

        return result, node_expr

    def evaluate_expr(self, *args: Union[int, str], **kwargs: Union[int, str]) -> Tuple[int, str]:
        """A static version of the 'evaluate' method. Infer the output expression of the node
        function from the expressions of input operands and their value categories (levels).

        Args:
            args (Sequence[int | str]): Value categories (levels) and expressions of operands.
            kwargs (Dict[str, int | str], optional): Value categories (levels) and expressions of
                operands.

        Returns:
            int: Function node output value category.
            str: Function node expression.
        """
        # Bind the arguments of this function to the manually created node function signature
        bound_args = self.sig.bind(*args, **kwargs).arguments

        # Read operand levels and determine max operand level
        operand_levels = [bound_args[name] for name in self.inputs]
        max_level = max(operand_levels) if operand_levels else 0

        # Retrieve the function template at the max operand level
        node_func_format = self.func[max_level]
        node_output_level = self.output_level[max_level]

        # Promote operand expressions to the max level and apply them to the node function template
        promote: Callable[[str, int], str] = partial(
            OL.promote_expr, target_level=max_level, promotion_mask=self.promotion_mask)
        operand_exprs = {name: promote(bound_args[f'{name}_expr'], level) \
                         for name, level in zip(self.inputs, operand_levels)}
        node_expr = node_func_format(**operand_exprs)

        return node_output_level, node_expr


class GetFunctionNode(BaseFunctionNode):
    """Class for 'get' function nodes in a differentiable function graph.
    """
    def __init__(self, name: str, params: ConstantDict = {}, inputs: InputDict = {},
                 outputs: OutputList = [], **kwargs):
        """Initialize the 'get' function node.

        Args:
            name (str): Function node name.
            params (ConstantDict, optional): Function node parameters, including the name of the
                retrieved variable. Defaults to {}.
            inputs (InputDict, optional): Mapping from input connectors to predecessor nodes that
                they connect to. Defaults to {}.
            outputs (OutputList, optional): List of successor nodes, namely those who receive input
                from this node. Defaults to [].
            kwargs (Dict[str, Any], optional): Keyword arguments to pass into the parent class
                constructor.
        """
        super().__init__(name, params=params, inputs=inputs, outputs=outputs, **kwargs)

    def evaluate(self, var: Dict[str, ParamValue],
                 return_expr: bool = False) -> Union[ParamValue, Tuple[ParamValue, str]]:
        """Evaluate the function node by querying a variable dictionary.

        Args:
            var (Dict[str, ParamValue]): Collection of named variables visible to the function
                graph (i.e., accessible using Get nodes).
            return_expr (bool, optional): Return an expression of the Get node. Defaults to False.

        Returns:
            ParamValue: Retrieved variable value.
            str (optional): Node expression.
        """
        # Obtain the variable value
        var_name: str = self.params['name']
        var_value: ParamValue = var[var_name]

        if return_expr:
            return var_value, f"var['{var_name}']"
        else:
            return var_value

    def evaluate_expr(self, var: Dict[str, int]) -> Tuple[int, str]:
        """Static evaluation of the function node. Return output level and expression.

        Args:
            var (Dict[str, int]): The value categories (levels) of named variables visible
                to the function graph (i.e., accessible using Get nodes).

        Returns:
            int: Value category of the retrieved variable.
            str: Node expression.
        """
        # Obtain the variable level
        var_name: str = self.params['name']
        var_level: int = var[var_name]

        return var_level, f"var['{var_name}']"


class RandFunctionNode(BaseFunctionNode):
    """Class for 'rand' function node in a differentiable function graph.
    """
    def __init__(self, name: str, inputs: InputDict = {}, outputs: OutputList = [], **kwargs):
        """Initialize the 'rand' function node.

        Args:
            name (str): Function node name.
            inputs (InputDict, optional): Mapping from input connectors to predecessor nodes that
                they connect to. Defaults to {}.
            outputs (OutputList, optional): List of successor nodes, namely those who receive input
                from this node. Defaults to [].
            kwargs (Dict[str, Any], optional): Keyword arguments to pass into the parent class
                constructor.
        """
        super().__init__(name, inputs=inputs, outputs=outputs, **kwargs)

    def evaluate(self, a: Optional[Union[float, th.Tensor]] = None,
                 return_expr: bool = False) -> Union[ParamValue, Tuple[ParamValue, str]]:
        """Evaluate the function node using an optional input that specifies random range.

        Args:
            a (Optional[float | Tensor], optional): Random number scale multiplier.
                Defaults to None (not applied).
            return_expr (bool, optional): Return an expression of the Get node. Defaults to False.

        Returns:
            ParamValue: Random scalar.
            str (optional): Node expression.
        """
        value = random.random()
        if not return_expr:
            return value

        # 'rand' node receives an optional input to determine random range
        rand_expr = 'rand()'
        expr = f'{rand_expr} * {a}' if a is not None else rand_expr
        return value, expr

    def evaluate_expr(self, a: Optional[int] = None, a_expr: Optional[str] = None) -> \
            Tuple[int, str]:
        """Static evaluation of the function node. Only used for determining output level.

        Args:
            a (Optional[int], optional): Value category (level) of the input random number scale
                multiplier. Defaults to None (not applied).
            a_expr (Optional[str], optional): Expression of the random number scale multiplier.
                Defaults to None.

        Returns:
            int: Value category of the generated random number(s).
            str: Node expression.
        """
        # Output is always a scalar
        level = 0

        # 'rand' node receives an optional input to determine random range
        rand_expr = 'rand()'
        expr = f'{rand_expr} * {a_expr}' if a is not None else rand_expr
        return level, expr


class FunctionGraph(BaseFunctionGraph[BaseFunctionNode]):
    """Differentiable function graph (value processor) class.
    """
    def __init__(self, nodes: List[BaseFunctionNode], output_node: BaseFunctionNode, name: str,
                 **kwargs):
        """Initialize the function graph.

        Args:
            nodes (List[BaseFunctionNode]): List of function nodes in the graph.
            output_node (BaseFunctionNode): Designated output node of the function graph. This node
                must be a member of `nodes`, otherwise an error will be thrown.
            name (str): Function graph name, usually equal to the associated parameter name.
            kwargs (Dict[str, Any], optional): Keyword arguments to pass into the parent class
                constructor.
        """
        super().__init__(nodes, output_node, **kwargs)

        self.name = name

        # Topologically sort the function nodes
        self._sort_nodes()

        # Store precompiled program
        self.program_str: Optional[str] = None
        self.program: Optional[CodeType] = None

    def compile(self, var: Dict[str, int]) -> int:
        """Compile the function graph into a sequence of statements and obtain the Python code.

        This step performs static type checking and returns the output level.

        Args:
            var (Dict[str, int]): Value categories (levels) of all named variables visible to this
                graph, e.g., exposed parameters and internal parameter of a material node.

        Raises:
            RuntimeError: Function graph compilation failed since the output expression of a
                function node is invalid.

        Returns:
            int: Value category of the graph output.
        """
        # Initialize the list of statements and the variable level memory
        statements: List[str] = []
        memory_exprs: Dict[str, str] = {node.name: f'x{i}' for i, node in enumerate(self.nodes)}
        memory_levels = var.copy()

        # Iterate over the node sequence to generate statements and run type checking
        for node in self.nodes:

            # Get the variable name of this node
            expr = memory_exprs[node.name]

            # Collect input levels and variable names
            input_levels = [memory_levels[name] for name in node.inputs.values()]
            input_exprs = [memory_exprs[name] for name in node.inputs.values()]

            # Obtain node output level and generate node expression
            # Only pass the 'var' dictionary to the node if it conducts 'get' or 'set' operations
            var_args = [var] if isinstance(node, GetFunctionNode) else []
            level, node_expr = node.evaluate_expr(*var_args, *input_levels, *input_exprs)
            if level < 0:
                raise RuntimeError(f"Function graph compilation failed. The output of expression "
                                   f"'{node_expr}' is invalid.")

            # Generate node statement and update memory
            statements.append(f'{expr} = {node_expr}')
            memory_levels[node.name] = level

        # Add the output statement
        output_expr = memory_exprs[self.output_node.name]
        output_level = memory_levels[self.output_node.name]
        statements.append(f'result = {output_expr}')

        # Compile the list of statements into Python code
        self.program_str = '\n'.join(statements)
        self.program: CodeType = compile(self.program_str, '<string>', 'exec')

        return output_level

    def evaluate(self, var: Dict[str, ParamValue] = {}) -> ParamValue:
        """Evaluate the function graph with external (global or node-specific) variables.

        Args:
            var (Dict[str, ParamValue], optional): Collection of named variables visible to the
                function graph, including graph-wide exposed parameters and internal material node
                parameters. Defaults to {}.

        Raises:
            RuntimeError: The function graph has not been compiled.

        Returns:
            ParamValue: Output value (or texture map) of the function graph.
        """
        # Check if the material graph has been compiled
        if not self.program:
            raise RuntimeError("The function graph has not been compiled. Please invoke the "
                               "'compile' method first.")

        # The namespace for function graph execution
        #   - '_t' and '_at' are tensor creation functions on the local device
        #   - 'rand' is a random number generation function
        local_ns: Dict[str, Any] = {
            'math': math,
            'torch': th,
            'var': var,
            '_t': self._t,
            '_at': self._at,
            'rand': random.random,
        }

        # Execute the compiled code and extract the result
        try:
            exec(self.program, local_ns)
        except Exception as e:
            self.log_debug_info(local_ns, has_error=True)
            raise e

        result: ParamValue = local_ns['result']

        # Convert 1D len=1 tensor back to scalar
        if isinstance(result, th.Tensor) and result.ndim == 1:
            result = result.squeeze()

        return result

    def to_device(self, device: Union[str, th.device] = 'cpu'):
        """Move the function graph to a specified device (e.g., CPU or GPU).

        Args:
            device (DeviceType, optional): Target device ID. Defaults to 'cpu'.
        """
        # Move nodes to the target device and set the device attribute
        super().to_device(device)

    def log_debug_info(self, memory: Dict[str, Any], has_error: bool = False):
        """Print compiled program and runtime memory info (for debugging) of a function graph.

        Args:
            memory (Dict[str, Any]): Runtime memory of the function graph as a mapping from
                variable names to values.
            has_error (bool, optional): Whether this method is called in the wake of a runtime
                error during function graph evaluation. Defaults to False.
        """
        # Print the compiled program
        indented_program = textwrap.indent(self.program_str, '  ')
        if has_error:
            self.logger.critical(
                f'Error encountered while executing the function graph. Please refer to the '
                f'program content as follows:\n{indented_program}')
        else:
            self.logger.debug(
                f"Compiled program of function graph '{self.name}':\n{indented_program}")

        # Print the memory info
        def f(x: ParamValue) -> str:
            if isinstance(x, th.Tensor) and x.ndim > 1:
                return f'tensor of shape {list(x.shape)}'
            else:
                return str(x)

        memory_str = '\n'.join(
            [f'{key}: {f(val)}' for key, val in memory.items() if key.startswith('x')])
        indented_memory_str = textwrap.indent(memory_str, '  ')
        self.logger.debug(
            f"Runtime memory of function graph '{self.name}':\n{indented_memory_str}")
