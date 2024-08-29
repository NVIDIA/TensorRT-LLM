from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

from lark import Lark, Token, Tree

if TYPE_CHECKING:
    from tensorrt_llm.tools.plugin_gen.core import Argument

parser = Lark(r"""
value: SIGNED_NUMBER
      | name
      | expr
      | "(" expr ")"

expr: value "+" value -> add
    | value "-" value -> sub
    | value "*" value -> mul
    | value "/" value -> div
    | value "///" value -> cdiv
    | value

shaped_tensor: name "[" value ("," value)* ("," "*")? "]" -> tensor
      | name "[" "*" "]" -> wildcard_tensor

tensors: shaped_tensor ("," shaped_tensor)*

deduce_shape: tensors "->" tensors

deduce_dim_size_arg: tensors ":" expr "->" name

name: CNAME
?start: deduce_shape | deduce_dim_size_arg

%import common.SIGNED_NUMBER
%import common.WS
%import common.CNAME
%ignore WS
""".strip())


class TargetType(Enum):
    CONCRETE = 0  # to produce size_t
    SYMBOLIC = 1  # to produce IDimensionExpr


# Here we introduce a set of ASTs to represent the target's expression.
# The Ast nodes from lark is not convenient to use.
class _AST:
    pass


@dataclass
class NumberAST(_AST):
    value: int
    target_type: TargetType = TargetType.CONCRETE


@dataclass
class BinaryAST(_AST):
    op: str
    left: _AST
    right: _AST
    target_type: TargetType = TargetType.CONCRETE


@dataclass
class ShapeAST:
    dims: List[_AST]


@dataclass
class DimAST(_AST):
    name: str


@dataclass
class ShapedTensorAST(_AST):
    arg_name: str
    shape: ShapeAST


@dataclass
class DeduceShapeRule(_AST):
    left: List[ShapedTensorAST]
    right: List[ShapedTensorAST]


@dataclass
class DeduceDimSizeArgRule(_AST):
    left: List[ShapedTensorAST]
    expr: _AST
    right: str


class ToAst:

    def __call__(self,
                 tree: Tree) -> Union[DeduceShapeRule, DeduceDimSizeArgRule]:
        if tree.data == "deduce_shape":
            assert len(tree.children) == 2
            return self.visit_DeduceShape(tree.children[0], tree.children[1])
        elif tree.data == "deduce_dim_size_arg":
            assert len(tree.children) == 3
            return self.visit_DeduceDimSizeArg(tree.children[0],
                                               tree.children[1],
                                               tree.children[2])
        raise NotImplementedError()

    def visit_DeduceShape(self, left: Tree, right: Tree) -> DeduceShapeRule:
        assert left.data == "tensors"
        assert right.data == "tensors"

        lefts = self.visit_tensors(left, TargetType.SYMBOLIC)
        rights = self.visit_tensors(right, TargetType.SYMBOLIC)
        return DeduceShapeRule(lefts, rights)

    def visit_DeduceDimSizeArg(self, left: Tree, expr: Tree,
                               right: Tree) -> DeduceDimSizeArgRule:
        lefts = self.visit_tensors(left, TargetType.CONCRETE)
        _expr = self.visit_expr(expr, TargetType.CONCRETE)
        rights = self.visit_name(right)
        return DeduceDimSizeArgRule(lefts, _expr, rights)

    def visit_tensors(self, tree: Tree,
                      target_type: TargetType) -> List[ShapedTensorAST]:
        assert tree.data == "tensors", repr(tree)
        return [
            self.visit_tensor(child, target_type) for child in tree.children
        ]

    def visit_tensor(self, tree: Tree,
                     target_type: TargetType) -> ShapedTensorAST:
        if tree.data == "tensor":
            arg_name = self.visit_name(tree.children[0])
            dims = [
                self.visit_expr(child, target_type)
                for child in tree.children[1:]
            ]
            return ShapedTensorAST(arg_name, ShapeAST(dims))

        assert tree.data == "wildcard_tensor", repr(tree)
        arg_name = self.visit_name(tree.children[0])
        return ShapedTensorAST(arg_name, ShapeAST([DimAST("*")]))

    def visit_number(self, v: str) -> _AST:
        return NumberAST(int(v))

    def visit_expr(self, tree: Tree, target_type: TargetType) -> _AST:
        '''
        for expression of dims, like `m * 2 + 1`
        '''

        def visit(tree: Union[Tree, Token]) -> _AST:
            if isinstance(tree, Token):
                if tree.type == "SIGNED_NUMBER":
                    return NumberAST(int(tree.value), target_type)
                elif tree.type == "CNAME":
                    return DimAST(tree.value)
                raise ValueError("Unexpected token: %s" % tree)

            elif isinstance(tree.data, Token):  # RULE; CNAME
                tree_type = tree.data.value
                if tree_type == 'name':
                    return DimAST(tree.children[0].value)
                elif tree_type == 'value':
                    return visit(tree.children[0])
                elif tree_type == 'expr':
                    return visit(tree.children[0])
                elif tree.data == "SIGNED_NUMBER":
                    return NumberAST(int(tree.children[0].data))
                else:
                    raise ValueError(f"Unexpected tree: {repr(tree)}")

            # (add, sub, mul) have operator overloading for IDimensionExpr
            # no need to do anything special
            elif tree.data == "add":
                assert len(tree.children) == 2
                return BinaryAST("+", visit(tree.children[0]),
                                 visit(tree.children[1]))
            elif tree.data == "sub":
                assert len(tree.children) == 2
                return BinaryAST("-", visit(tree.children[0]),
                                 visit(tree.children[1]))
            elif tree.data == "mul":
                assert len(tree.children) == 2
                return BinaryAST("*", visit(tree.children[0]),
                                 visit(tree.children[1]))
            elif tree.data == "div":
                assert len(tree.children) == 2
                return BinaryAST("/", visit(tree.children[0]),
                                 visit(tree.children[1]), target_type)
            elif tree.data == "cdiv":
                assert len(tree.children) == 2
                return BinaryAST("///", visit(tree.children[0]),
                                 visit(tree.children[1]), target_type)
            else:
                raise ValueError(f"Unexpected tree: {repr(tree)}")

        return visit(tree)

    def visit_name(self, tree: Tree) -> str:
        assert isinstance(tree.data, Token) and tree.data.value == "name"
        return tree.children[0].value


@dataclass
class Dim:
    arg: "Argument"
    dim_off: int


@dataclass
class CppCodeTranspiler:
    # The mapping from a arg_name in the expression to the corresponding Argument.
    name_to_arg: Dict[str, "Argument"]

    # The mapping from a dim_name in the expression to the corresponding Dim in an Argument.
    name_to_dim: Dict[str, Dim] = field(default_factory=dict, init=False)

    def __call__(self, exprs: List[str]) -> Tuple[List[str], Dict[str, str]]:
        asts = [self.to_ast(expr) for expr in exprs]
        return self.codegen(asts)

    def to_ast(self, expr: str) -> _AST:
        self.cur_expr = expr
        ast = parser.parse(expr)
        ast = ToAst()(ast)
        return ast

    def codegen(self, asts: List[_AST]) -> Tuple[List[str], Dict[str, str]]:
        '''
        Parse an expression group and generate the corresponding C++ code.

        The syntax of an expression is like below:

        - `name[expr, expr, ...] -> name[expr, expr, ...]`
        - `name[expr, expr, ...]:expr -> dim_arg`
        '''
        shape_infer_code = []
        dim_size_infer_code = {}

        for ast in asts:
            if isinstance(ast, DeduceShapeRule):
                self.dim_cpp_repr = lambda arg_idx, dim_idx: f"inputDims[{arg_idx}].d[{dim_idx}]"
                shape_infer_code.extend(self.emit_DeduceShapeRule(ast))
            elif isinstance(ast, DeduceDimSizeArgRule):
                self.dim_cpp_repr = lambda arg_idx, dim_idx: f"inputDesc[{arg_idx}].dims.d[{dim_idx}]"
                dim_size_infer_code[ast.right] = self.emit_DeduceDimSizeArgRule(
                    ast)
            else:
                raise ValueError("Unexpected ast: %s" % repr(ast))

        return shape_infer_code, dim_size_infer_code

    @staticmethod
    def is_cur_identical_dims(item: ShapedTensorAST):
        return len(item.shape.dims) == 1 and isinstance(
            item.shape.dims[0], DimAST) and item.shape.dims[0].name == "*"

    def collect_dims_from_left(self, lefts: List[ShapedTensorAST]):
        self.name_to_dim.clear()

        is_left_identical_dims = self.is_cur_identical_dims(lefts[0])
        # process left, and record the named dimensions
        for left in lefts:
            arg_name = left.arg_name
            argument = self.name_to_arg[arg_name]
            for off, dim in enumerate(left.shape.dims):
                assert isinstance(
                    dim, DimAST
                ), f"Wrong syntax in '{self.cur_expr}', for deduce_shape rule, each named dimension should be a name rather than an expression"
                self.name_to_dim[dim.name] = Dim(argument, off)
        return is_left_identical_dims

    def emit_DeduceShapeRule(self, rule: DeduceShapeRule) -> List[str]:
        from tensorrt_llm.tools.plugin_gen.core import code

        is_cur_identical_dims = lambda item: len(
            item.shape.dims) == 1 and isinstance(item.shape.dims[
                0], DimAST) and item.shape.dims[0].name == "*"

        is_left_identical_dims = self.collect_dims_from_left(rule.left)

        first_left_tensor = rule.left[0]
        first_left_tensor_arg = self.name_to_arg[first_left_tensor.arg_name]

        ret = []
        # process right, and generate the code for each dimensions

        # TODO: support more wildcard cases, currently only A[*] -> B[*], C[*] is supported
        is_right_identical_dims = False
        for off, item in enumerate(rule.right):
            is_cur_identical_dims = self.is_cur_identical_dims(item)
            if is_right_identical_dims and not is_cur_identical_dims:
                assert is_cur_identical_dims, "Wrong syntax in '%s', for deduce_shape rule, once the left side be X[*], the should all be X[*] format too" % self.cur_expr
            is_right_identical_dims = is_cur_identical_dims

        assert is_left_identical_dims == is_right_identical_dims, "Wrong syntax in '%s', for deduce_shape rule, the left and right side should be both X[*] or not" % self.cur_expr

        for off, tensor in enumerate(rule.right):
            out_arg = self.name_to_arg[tensor.arg_name]
            ret.append(code(f"if (outputIndex == {out_arg.offset}) {{"))

            if is_right_identical_dims:
                ret.append(
                    code(
                        f"  outputDims = inputDims[{first_left_tensor_arg.offset}];"
                    ))
            else:
                ret.append(
                    code(f"  outputDims.nbDims = {len(tensor.shape.dims)};"))
                for dim_off, dim in enumerate(tensor.shape.dims):
                    ret.append(
                        code(
                            f"  outputDims.d[{dim_off}] = {self.emit_expr(dim)};"
                        ))

            ret.append(code(f"}}"))

        return ret

    def emit_DeduceDimSizeArgRule(self, rule: DeduceDimSizeArgRule) -> str:
        self.collect_dims_from_left(rule.left)
        return self.emit_expr(rule.expr)

    def emit_expr(self, expr: _AST) -> str:
        if isinstance(expr, NumberAST):
            if expr.target_type == TargetType.SYMBOLIC:
                return f"exprBuilder.constant({expr.value})"
            else:
                return str(expr.value)
        elif isinstance(expr, DimAST):
            return self.emit_dim(expr)
        elif isinstance(expr, BinaryAST):
            return self.emit_binary(expr)
        raise ValueError("Unexpected expr: %s" % expr)

    def emit_dim(self, dim: DimAST) -> str:
        dim_: Dim = self.name_to_dim[dim.name]
        repr = self.dim_cpp_repr(dim_.arg.offset, dim_.dim_off)
        return repr

    def emit_binary(self, binary: BinaryAST) -> str:
        left = self.emit_expr(binary.left)
        right = self.emit_expr(binary.right)
        if binary.op == "/" and binary.target_type == TargetType.SYMBOLIC:
            return f"exprBuilder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV, *{left}, *{right})"
        elif binary.op == "///":
            if binary.target_type == TargetType.SYMBOLIC:
                return f"exprBuilder.operation(nvinfer1::DimensionOperation::kCEIL_DIV, *{left}, *{right})"
            else:
                return f"(({left} + {right} - 1) / {right})"
        else:
            return f"({left} {binary.op} {right})"
