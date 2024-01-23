import inspect
import weakref
from copy import copy
from dataclasses import dataclass, field
from functools import wraps
from typing import (Any, Callable, ClassVar, Dict, List, Optional, Set, Tuple,
                    TypeVar)

import tensorrt as trt

from .logger import logger
from .network import Network


class Layer:
    '''
    Layer is a wrapper for TensorRT's ILayer with several python-friendly helper functions.
    '''

    def __init__(self, network: Network, trt_layer: trt.ILayer):
        self._network = weakref.ref(network)
        self.trt_layer = trt_layer

        assert isinstance(self.network, Network)
        assert isinstance(self.trt_layer, trt.ILayer)

    @property
    def network(self):
        return self._network()

    def get_inputs(self, *indices: int):
        '''
        Get the input tensors of the layer.

        Parameters:
            idx: the indices of the input tensor, will return all inputs if left empty

        Returns:
            List[Tensor]
        '''
        from .functional import Tensor
        indices = indices if indices else range(self.trt_layer.num_inputs)

        ret = []
        for i in indices:
            assert i < self.trt_layer.num_inputs, f"Invalid input index {i} for layer {self.trt_layer.name}"

            tensor = self.trt_layer.get_input(i)
            tensor = Tensor(trt_tensor=tensor,
                            network=self.network,
                            is_network_input=False)
            ret.append(tensor)
        return ret

    def get_outputs(self, *indices: int):
        '''
        Get the output tensor of the layer.

        Parameters:
            idx: the index of the output tensor

        Returns:
            List[Tensor]
        '''
        from .functional import Tensor

        indices = indices if indices else range(self.trt_layer.num_outputs)

        ret = []
        for i in indices:
            assert i < self.trt_layer.num_outputs, f"Invalid output index {i} for layer {self.trt_layer.name}"

            tensor = self.trt_layer.get_output(i)
            tensor = Tensor(trt_tensor=tensor,
                            network=self.network,
                            is_network_input=False)
            ret.append(tensor)
        return ret

    def is_removed(self):
        return self.network.is_removed_layer(self)

    def mark_as_removed(self):
        '''
        Mark the layer as removed, this will remove the layer from the network.
        '''
        # NOTE, since INetwork python API doesn't provide a way to remove a layer, we actually mark the layer as removed in the network.
        self.network.mark_removed_layer(self)

        # remove the FLayerInfo if exists
        FLayerInfoMemo.instance().remove(self.name)

    def __eq__(self, other: "Layer") -> bool:
        if isinstance(other, Layer):
            return self.trt_layer == other.trt_layer
        if isinstance(other, trt.tensorrt.ILayer):
            return self.trt_layer == other
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.trt_layer, name)

    # Refer to https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/Layers.html?highlight=elementwise#layers for a complete
    # list of TRT layers.
    TRT_LAYER_TYPE_TO_LAYER = {
        trt.LayerType.CONVOLUTION: trt.IConvolutionLayer,
        trt.LayerType.ACTIVATION: trt.IActivationLayer,
        trt.LayerType.POOLING: trt.IPoolingLayer,
        trt.LayerType.LRN: trt.ILRNLayer,
        trt.LayerType.SCALE: trt.IScaleLayer,
        trt.LayerType.SOFTMAX: trt.ISoftMaxLayer,
        trt.LayerType.DECONVOLUTION: trt.IDeconvolutionLayer,
        trt.LayerType.CONCATENATION: trt.IConcatenationLayer,
        trt.LayerType.ELEMENTWISE: trt.IElementWiseLayer,
        trt.LayerType.UNARY: trt.IUnaryLayer,
        trt.LayerType.PADDING: trt.IPaddingLayer,
        trt.LayerType.SHUFFLE: trt.IShuffleLayer,
        trt.LayerType.REDUCE: trt.IReduceLayer,
        trt.LayerType.TOPK: trt.ITopKLayer,
        trt.LayerType.GATHER: trt.IGatherLayer,
        trt.LayerType.MATRIX_MULTIPLY: trt.IMatrixMultiplyLayer,
        trt.LayerType.RAGGED_SOFTMAX: trt.IRaggedSoftMaxLayer,
        trt.LayerType.CONSTANT: trt.IConstantLayer,
        trt.LayerType.IDENTITY: trt.IIdentityLayer,
        trt.LayerType.PLUGIN_V2: trt.IPluginV2Layer,
        trt.LayerType.SLICE: trt.ISliceLayer,
        trt.LayerType.SHAPE: trt.IShapeLayer,
        trt.LayerType.PARAMETRIC_RELU: trt.IParametricReLULayer,
        trt.LayerType.RESIZE: trt.IResizeLayer,
        trt.LayerType.TRIP_LIMIT: trt.ITripLimitLayer,
        trt.LayerType.RECURRENCE: trt.IRecurrenceLayer,
        trt.LayerType.ITERATOR: trt.IIteratorLayer,
        trt.LayerType.LOOP_OUTPUT: trt.ILoopOutputLayer,
        trt.LayerType.SELECT: trt.ISelectLayer,
        trt.LayerType.FILL: trt.IFillLayer,
        trt.LayerType.QUANTIZE: trt.IQuantizeLayer,
        trt.LayerType.DEQUANTIZE: trt.IDequantizeLayer,
        trt.LayerType.CONDITION: trt.IConditionLayer,
        trt.LayerType.CONDITIONAL_INPUT: trt.IIfConditionalInputLayer,
        trt.LayerType.CONDITIONAL_OUTPUT: trt.IIfConditionalOutputLayer,
        trt.LayerType.ASSERTION: trt.IAssertionLayer,
        trt.LayerType.SCATTER: trt.IScatterLayer,
        trt.LayerType.EINSUM: trt.IEinsumLayer,
        trt.LayerType.GRID_SAMPLE: trt.IGridSampleLayer,
        trt.LayerType.ONE_HOT: trt.IOneHotLayer,
        trt.LayerType.NON_ZERO: trt.INonZeroLayer,
        trt.LayerType.NMS: trt.INMSLayer,
        trt.LayerType.REVERSE_SEQUENCE: trt.IReverseSequenceLayer,
        trt.LayerType.NORMALIZATION: trt.INormalizationLayer,
        trt.LayerType.CAST: trt.ICastLayer,
    }

    def as_layer(self) -> Any:
        '''
        Convert to a actual TensorRT layer object, such as IPluginV2Layer or IConvolutionLayer so
        that we can access the actual layer information.
        '''
        if self.type in self.TRT_LAYER_TYPE_TO_LAYER:
            # bypass TRT's bug of retrieving a specific ILayer type in TensorRT
            self.trt_layer.__class__ = self.TRT_LAYER_TYPE_TO_LAYER[self.type]
            return self.trt_layer
        raise NotImplementedError(f"Unknown layer type: {self.type}")

    def __hash__(self):
        return id(self.trt_layer)


@dataclass
class _Pattern:
    name: str
    # args helps to pass in/out some information
    args: Dict[str, Any] = field(default_factory=dict, init=False)

    def log_info(self, msg: str):
        logger.info(f"Pattern {self.name}: {msg}")

    def log_error(self, msg: str):
        logger.error(f"Pattern {self.name}: {msg}")

    def log_warn(self, msg: str):
        logger.warning(f"Pattern {self.name}: {msg}")


class PatternRewriter(_Pattern):
    '''
    A pattern rewriter is a class that can match a pattern in the graph and rewrite the matched pattern.

    There are two ways to implement a pattern rewriter, either override match() and rewrite() separately, or override match_and_rewrite().
    '''

    def __init__(self,
                 name: str,
                 root_layer: Optional[Set[trt.LayerType]] = None,
                 seperate_match_rewrite=False):
        '''
        Parameters:
            name: the name of the rewrite pattern
            root_layer: the root layer types to start the pattern matching, if not provided, the pattern will traverse all the layers in the graph.
            seperate_match_rewrite: if set to True, the pattern should override match() and rewrite() separately, otherwise, the pattern should override match_and_rewrite()
        '''
        super().__init__(name)
        self.root_layer = root_layer
        self._seperate_match_rewrite = seperate_match_rewrite

    def match(self, layer: Layer) -> bool:
        raise NotImplementedError()

    def rewrite(self, layer: Layer) -> None:
        raise NotImplementedError()

    def match_and_rewrite(self, layer: Layer) -> bool:
        raise NotImplementedError()


class PatternAnalyzer(_Pattern):

    def __init__(self, name: str,
                 root_layer: Optional[Set[trt.LayerType]]) -> None:
        super().__init__(name)
        self.root_layer = root_layer

    def match(self, layer: Layer) -> bool:
        raise NotImplementedError()

    def analyze(self, subgraph: List[Layer]) -> None:
        raise NotImplementedError()


class _PatternManager:
    PatternType = TypeVar('PatternType')

    def __init__(self):
        # records of (benefit, pattern, id)
        self.patterns: Dict[str, Tuple[int, _PatternManager.PatternType]] = {}

    def add(self,
            label: str,
            pattern: "_PatternManager.PatternType",
            benefit: int = 0):
        assert label not in self.patterns, f"Pattern {label} already exists"
        self.patterns[label] = (benefit, pattern)

    def get(self, label: str) -> "_PatternManager.PatternType":
        return self.patterns[label][1]


class RewritePatternManager(_PatternManager):

    def rewrite(self, net: Network, args=None):
        modified = True
        # TODO: we can optimize this by asking TRT to expose a graph iterator consistent even after the graph is modified
        while modified:
            modified = False
            # Since the graph iterator is hold by the underlying INetwork, we can only rebuild the graph cache and match the nodes again.
            for layer in net.get_layers():
                if layer.is_removed():
                    continue
                for (profit, pattern) in sorted(self.patterns.values(),
                                                key=lambda x: x[0]):
                    pattern.args = args

                    if pattern.root_layer is not None and layer.type not in pattern.root_layer:
                        continue
                    if pattern._seperate_match_rewrite:
                        if pattern.match(layer):
                            pattern.rewrite(layer)
                            modified = True
                    else:
                        if pattern.match_and_rewrite(layer):
                            modified = True

    @staticmethod
    def instance():
        return _global_rewrite_pattern_manager


class AnalysisPatternManager(_PatternManager):

    def analyze(self, graph: Network, args=None):
        for layer in graph.get_layers():
            if layer.name in graph.removed_layers:
                continue
            for (benefit, pattern) in sorted(self.patterns.values(),
                                             key=lambda x: x[0]):
                pattern.args = args

                if pattern.root_layer is not None and layer.type not in pattern.root_layer:
                    continue
                if pattern.match(layer):
                    subgraph = pattern.match(layer)
                    pattern.analyze(subgraph)

    @staticmethod
    def instance():
        return _global_analysis_pattern_manager


@dataclass
class FLayerInfo:
    '''
    The FLayerInfo is used to track the functional layers in the INetwork, and it is used to help the graph rewriting.

    The lifetime of a FLayer is the same as the corresponding plugin instance in the INetwork. Once the
    plugin instance is removed by the graph rewriting, the FLayer will be removed as well.

    WHY this is needed?
    In the current implementation, for functional methods, once it is called in Python, it will lower to a plugin instance in the INetwork. However,
    the plugin interface is black box with customized logic, we cannot retrieve necessary information from it, this is quite different from ILayer,
    which provides a set of APIs to retrieve the information. Therefore, we need to record the high level information in the FLayerInfo, and keep
    it consistent during the graph rewriting.
    '''
    layer_kind: str  # the method name in the functional.py
    # Record the raw inputs of the functional layer to be used in the graph rewrite
    # NOTE: the raw inputs contains both the constants and Tensors, the Tensors will be also updated by graph rewriting
    # APIs such as `replace_all_uses_with`
    raw_inputs: Dict[str, Any]

    raw_outputs: List[Any] = field(default_factory=list, init=False)

    # the corresponding ILayer name
    layer_name: str = field(init=False, default="")

    # the signature of the functional layer
    signature: Any = field(init=False, default=None)

    def __post_init__(self):
        from .functional import Tensor
        assert self.layer_kind

        def replace_with_symbols(arg) -> Any:
            if arg is None:
                return None
            if isinstance(arg, Tensor):
                return Tensor
            if isinstance(arg, (list, tuple)):
                return [replace_with_symbols(x) for x in arg]
            if isinstance(arg, dict):
                return {k: replace_with_symbols(v) for k, v in arg.items()}

            return arg

        def amend_tensor(arg) -> Any:
            if arg is None:
                return None
            if isinstance(arg, Tensor):
                arg.network = self.network
            if isinstance(arg, (list, tuple)):
                [replace_with_symbols(x) for x in arg]
            if isinstance(arg, dict):
                {k: replace_with_symbols(v) for k, v in arg.items()}

            return arg

        self.signature = self.layer_kind, {
            name: replace_with_symbols(value)
            for name, value in self.raw_inputs.items()
        }

        amend_tensor(self.raw_inputs)

    def set_outputs(self, outputs: List[Any]):
        self.raw_outputs = outputs

    def get_input(self, name: str) -> Any:
        return self.raw_inputs[name]

    def clone_inputs(self):
        '''
        Get a shallow copy of the inputs.
        '''
        return copy(self.raw_inputs)

    def replace_input_with(self, src, dst):
        '''
        Replace the input `src` with the input `dst` in the raw_inputs.

        src: Tensor
        dst: Tensor
        '''
        from .functional import Tensor

        def replace(arg: Any):
            if isinstance(arg, Tensor):
                if arg.trt_tensor is src.trt_tensor:
                    return dst
                return arg
            elif isinstance(arg, (list, tuple)):
                return [replace(x) for x in arg]
            elif isinstance(arg, dict):
                return {k: replace(v) for k, v in arg.items()}
            return arg

        replace(self.raw_inputs)

    def replace_outputs_uses_with(self, net: Network, new_outs: List[Any]):
        '''
        Replace the output users with the new outputs.

        new_outs: List[Tensor], the new outputs to replace with
        '''
        from .functional import Tensor
        assert len(self.raw_outputs) == len(new_outs)
        for old_out, new_out in zip(self.raw_outputs, new_outs):
            assert type(old_out) == type(
                new_out
            ), f"rewrite error, the output type {type(old_out)} is different from the new output type {type(new_out)} not match the original output type {type(old_out)}"

        def _swap_tensor_info(new, deprecated):
            name = deprecated.trt_tensor.name
            deprecated.trt_tensor.name = name + '_deprecated'
            from .functional import cast

            new = cast(new, deprecated.dtype)
            new.trt_tensor.name = name

        def _reset_network_output_tensors(network, out, new_out):
            net_outputs = list()
            num_outputs = network._trt_network.num_outputs
            need_to_mark = False
            for i in range(num_outputs):
                net_outputs.append(network._trt_network.get_output(i))
                if out.trt_tensor is net_outputs[i]:
                    need_to_mark = True
            if need_to_mark is False:
                return
            for output in net_outputs:
                network.trt_network.unmark_output(output)
            for i in range(num_outputs):
                if net_outputs[i] is out.trt_tensor:
                    network.trt_network.mark_output(new_out.trt_tensor)
                    new_out.trt_tensor.dtype = out.trt_tensor.dtype
                else:
                    network.trt_network.mark_output(net_outputs[i])

        def replace_all_uses_with(out, new_out):
            if isinstance(out, Tensor):
                assert isinstance(new_out, Tensor)
                out.replace_all_uses_with(new_out)
                _swap_tensor_info(new_out, out)
                _reset_network_output_tensors(net, out, new_out)
            elif isinstance(out, list):
                assert isinstance(new_out, list)
                for x, y in zip(out, new_out):
                    replace_all_uses_with(x, y)
            elif isinstance(out, dict):
                assert isinstance(new_out, dict)
                for k, v in out.items():
                    replace_all_uses_with(v, new_out[k])
            elif isinstance(out, tuple):
                assert isinstance(new_out, tuple)
                for x, y in zip(out, new_out):
                    replace_all_uses_with(x, y)

        replace_all_uses_with(self.raw_outputs, new_outs)

    def __hash__(self) -> int:
        return hash(self.signature)

    def __repr__(self) -> str:
        return '<FLayer {}>'.format(self.signature)

    @staticmethod
    def _get_spec(arg):
        '''
        Get the spec that could impact on the Module's topology in the `forward` method.
        '''
        from .functional import Tensor

        # For scalars, we track their value since they are constant
        if arg is None:
            return None
        elif isinstance(arg, (bool, int, str)):
            return arg
        # For tensors, currently we only track their type, since they are variables
        elif isinstance(arg, Tensor):
            return Tensor
        elif isinstance(arg, (list, tuple)):
            return [FLayerInfo._get_spec(x) for x in arg]
        # NOTE Free to add more types here is broken, carefully note that, from the engine building angle, all the constants
        # should be captured while for the network variables, their types as placeholders are enough
        else:
            raise TypeError(f"unsupported input type detected: {type(arg)}")


@dataclass
class FLayerInfoMemo:
    '''
    FLayerInfoMemo holds the FLayer of all the necessary functional layers.
    '''
    data: Dict[str, FLayerInfo] = field(default_factory=dict, init=False)

    cur_flayer: ClassVar[Optional[FLayerInfo]] = None

    def add(self, layer_name: str, layer: FLayerInfo) -> None:
        assert layer_name not in self.data, f"FLayer {layer_name} already exists in FLayerMemo"
        self.data[layer_name] = layer

    def create(self, fn: Callable, *args, **kwargs) -> FLayerInfo:
        '''
        Add a FLayer to the memo.
        '''
        return FLayerInfo(fn.__name__,
                          self.get_function_arg_dict(fn, *args, **kwargs))

    def get(self, layer_name: str) -> Optional[FLayerInfo]:
        return self.data.get(layer_name, None)

    def remove(self, layer_name: str) -> None:
        if layer_name in self.data:
            del self.data[layer_name]

    @staticmethod
    def instance() -> "FLayerInfoMemo":
        '''
        A singleton instance of FLayerMemo.
        '''
        from ._common import default_net
        return default_net().flayer_memo

    @staticmethod
    def get_function_arg_dict(f: Callable, *args, **kwargs):
        '''
        Get the input argument dict of a function.
        '''
        sig = inspect.signature(f)

        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        return {k: v for k, v in bound_args.arguments.items() if k != 'self'}


class FLayerScope:
    '''
    FLayerScope is used to capture the plugin within a functional method.
    '''

    def __init__(self, fn, *args, **kwargs):
        self.layer = FLayerInfoMemo.instance().create(fn, *args, **kwargs)

    def __enter__(self):
        assert FLayerInfoMemo.cur_flayer is None, "FLayerMemo is not reentrant"
        # There is no FLayer hierarchy, since the functional layers are not nested
        FLayerInfoMemo.cur_flayer = self.layer

    def __exit__(self, exc_type, exc_val, exc_tb):
        FLayerInfoMemo.cur_flayer = None
        if exc_type is None:
            assert self.layer.layer_name != "", f"FLayer {self.layer.layer_kind} without a plugin name detected"
            FLayerInfoMemo.instance().add(self.layer.layer_name, self.layer)


def record_signature(f):
    '''
    Helps to decorate a functional method and record its metadata with a FLayerInfo.
    '''

    @wraps(f)
    def wrapper(*args, **kwargs):
        with FLayerScope(f, *args, **kwargs):
            outs = f(*args, **kwargs)
            FLayerInfoMemo.cur_flayer.set_outputs(outs)
            return outs

    return wrapper


# singletons
_global_rewrite_pattern_manager = RewritePatternManager()
_global_analysis_pattern_manager = AnalysisPatternManager()


class FuseAttentionWithBiasPass(PatternRewriter):

    def __init__(self):
        super().__init__(name="fuse_attention_with_bias",
                         seperate_match_rewrite=False)

    @staticmethod
    def is_attention_plugin(layer: Layer) -> bool:
        if layer.as_layer().type != trt.LayerType.PLUGIN_V2:
            return False
        p = layer.as_layer().plugin
        conds = [
            p.plugin_namespace == 'tensorrt_llm',
            p.plugin_type == 'GPTAttention', p.num_outputs == 2
        ]
        return all(conds)

    @staticmethod
    def is_elementwise_sum(layer: Layer) -> bool:
        l = layer.as_layer()
        if l.type != trt.LayerType.ELEMENTWISE:
            return False
        return l.op == trt.ElementWiseOperation.SUM

    @staticmethod
    def get_eltwise_inputs(layer: Layer):
        const_inputs = []
        mutable_inputs = []

        from .functional import Tensor

        def const_foldable(tensor: Tensor, depth=0) -> bool:
            max_depth = 10
            layer = tensor.get_parent()
            if layer is None or depth > max_depth:
                return False
            if layer.type == trt.LayerType.CONSTANT and len(
                    layer.get_inputs()) == 0:
                return True
            for _ in layer.get_inputs():
                if not const_foldable(_, depth + 1): return False
            return True

        for input in layer.get_inputs():
            if const_foldable(input):
                const_inputs.append(input)
            else:
                mutable_inputs.append(input)
        return const_inputs, mutable_inputs

    def match_and_rewrite(self, layer: Layer) -> bool:
        from tensorrt_llm.network import net_guard
        with net_guard(layer.network):
            if not self.is_attention_plugin(layer):
                return False
            plugin_flayer = FLayerInfoMemo.instance().get(layer.name)
            input = plugin_flayer.raw_inputs['qkv']
            if input is None or isinstance(
                    input, list) or len(list(input.get_users())) != 1:
                return False
            parent_layer = input.get_parent()
            if not self.is_elementwise_sum(parent_layer):
                return False
            eltwise_const_inputs, eltwise_mutable_inputs = self.get_eltwise_inputs(
                parent_layer)
            if len(eltwise_const_inputs) != 1 or len(
                    eltwise_mutable_inputs) != 1:
                return False
            if plugin_flayer.raw_inputs['qkv_bias'] is not None:
                return False
            plugin_flayer.raw_inputs['qkv'] = eltwise_mutable_inputs[0]
            plugin_flayer.raw_inputs['qkv_bias'] = eltwise_const_inputs[0]
            from .functional import gpt_attention
            new_outputs = gpt_attention(**plugin_flayer.raw_inputs)
            plugin_flayer.replace_outputs_uses_with(layer.network, new_outputs)
        return True


def optimize(net):
    patterns = RewritePatternManager()
    patterns.add(
        label='fuse_attention_with_bias',
        pattern=FuseAttentionWithBiasPass(),
    )
    patterns.rewrite(net)
