# MLIR Elementwise Fusion — Sequence Diagram

## Overview

The `mlir_elementwise_fusion` transform converts an FX graph into MLIR,
decomposes high-level ops, discovers fusible subgraphs, generates Triton
kernels, replaces the subgraphs in the MLIR IR, and converts back to FX.

## Sequence Diagram

```mermaid
sequenceDiagram
    participant Pipeline as AutoDeploy Pipeline
    participant Transform as MLIRElementwiseFusion
    participant FX2MLIR as FXToMLIRConverter
    participant Decompose as run_decomposition
    participant Discovery as discover_fusible_subgraphs
    participant Codegen as generate_kernel_from_subgraph
    participant Replace as replace_subgraph_with_fused_op
    participant MLIR2FX as MLIRToFXConverter
    participant TorchLib as torch.library.custom_op

    Pipeline->>Transform: _apply(gm, ...)

    Note over Transform: Step 1: FX → MLIR
    Transform->>FX2MLIR: FXToMLIRConverter(gm).convert()
    activate FX2MLIR

    loop For each FX node
        FX2MLIR->>FX2MLIR: _convert_node(node, block)
        Note over FX2MLIR: aten.add → ad.add (scalar → ad.splat + ad.add)<br/>aten.mul/sub → ad.mul/sub<br/>aten.neg/silu/gelu/relu/tanh/sigmoid → ad.neg/…<br/>aten.exp/softplus/rsqrt/sqrt → ad.exp/…<br/>aten.pow → ad.pow, aten.mean → ad.reduce_mean<br/>rmsnorm variants → ad.rmsnorm<br/>gated rmsnorm → ad.gated_rmsnorm<br/>aten.to.dtype → ad.to_dtype<br/>unmapped ops → ad.opaque
    end

    FX2MLIR-->>Transform: mlir_module, metadata

    deactivate FX2MLIR

    Note over Transform: Step 2: Decompose
    Transform->>Decompose: run_decomposition(mlir_module)
    activate Decompose

    loop For each decomposable op (ad.rmsnorm, ad.gated_rmsnorm)
        Decompose->>Decompose: _DecompPattern.match_and_rewrite(op)
        Note over Decompose: ad.rmsnorm(x, w, eps) →<br/>ad.mul(x, x)<br/>ad.reduce_mean(sq, dim=-1)<br/>ad.splat(eps)<br/>ad.add(var, eps_t)<br/>ad.rsqrt(var_eps)<br/>ad.mul(x, inv)<br/>ad.mul(normed, w)<br/><br/>ad.gated_rmsnorm(x, w, gate, …) →<br/>ad.silu(gate) + ad.mul(x, silu_gate)<br/>then rmsnorm primitives (or reverse order<br/>depending on norm_before_gate)
    end

    Decompose-->>Transform: num_decomposed = 53

    deactivate Decompose

    Note over Transform: Step 3: Discover Subgraphs
    Transform->>Discovery: discover_fusible_subgraphs(mlir_module)
    activate Discovery

    Note over Discovery: Pass 1: Walk ops in topo order.<br/>Merge connected fusible ops<br/>into groups via data flow.<br/>Multi-branch merge if op<br/>consumes from 2+ groups.

    Note over Discovery: Pass 2: Pull zero-operand ops<br/>(ad.splat constants) into groups<br/>if all users are in one group.

    Note over Discovery: Build FusibleSubgraph objects<br/>for groups with ≥2 ops.<br/>Compute inputs (external SSA values)<br/>and outputs (values with external users).

    Discovery-->>Transform: subgraphs (75 found)

    deactivate Discovery

    Note over Transform: Step 4: Codegen + Replace
    loop For each subgraph
        Transform->>Transform: Check rank filter<br/>(max_input_rank ≥ 2, min_output_rank ≥ 2)

        alt Passes rank filter
            Transform->>Codegen: generate_kernel_from_subgraph(sg)
            activate Codegen

            Codegen->>Codegen: hash_subgraph(sg)
            Codegen->>Codegen: Check kernel cache

            Note over Codegen: Walk subgraph ops in topo order.<br/>Emit Triton expressions via _EMIT table.<br/>All loads upcast to f32.<br/>All stores downcast to original dtype.

            Codegen->>Codegen: Write kernel source to tempfile
            Codegen->>Codegen: importlib.load_module(tempfile)

            Codegen->>TorchLib: @torch.library.custom_op(<br/>"auto_deploy::mlir_fused_{hash}")
            TorchLib-->>Codegen: registered op

            Codegen->>TorchLib: @op.register_fake (shape propagation)
            TorchLib-->>Codegen: fake impl registered

            Codegen-->>Transform: kernel_fn (callable wrapper)
            deactivate Codegen

            Transform->>Replace: replace_subgraph_with_fused_op(<br/>sg, kernel_fn, hash, metadata)
            activate Replace

            Note over Replace: 1. Build AdOpaque op with<br/>   subgraph inputs as operands<br/>2. Set synthetic metadata:<br/>   _original_target = torch.ops fn<br/>   _args_template = operand indices<br/>   val = FakeTensor tuple<br/>3. Insert fused op before first sg op<br/>4. Replace sg output uses → fused op outputs<br/>5. Erase original sg ops (reverse order)

            Replace-->>Transform: (mlir_module modified in-place)
            deactivate Replace

        else Fails rank filter (1D weight ops)
            Transform->>Transform: skip (num_skipped++)
        end
    end

    Note over Transform: Step 5: MLIR → FX
    Transform->>MLIR2FX: MLIRToFXConverter(gm).convert(<br/>mlir_module, metadata)
    activate MLIR2FX

    loop For each MLIR op
        MLIR2FX->>MLIR2FX: _convert_op(mlir_op, graph, metadata)
        Note over MLIR2FX: ad.graph_input → placeholder / get_attr<br/>ad.add/mul/sub → aten.add/mul/sub.Tensor<br/>ad.neg/silu/gelu/relu/tanh/sigmoid → aten.*<br/>ad.exp/softplus/rsqrt/sqrt → aten.*<br/>ad.pow → aten.pow.Tensor_Scalar<br/>ad.reduce_mean/sum → aten.mean/sum.dim<br/>ad.splat → aten.scalar_tensor<br/>ad.cast → aten.to.dtype<br/>ad.rmsnorm → flashinfer_rms_norm<br/>ad.gated_rmsnorm → triton_rmsnorm_gated<br/>ad.opaque (fused) → torch.ops.auto_deploy.mlir_fused_{hash}<br/>  + getitem(result, i) for each output<br/>ad.opaque (other) → original FX target<br/>ad.graph_output → output
    end

    MLIR2FX-->>Transform: new_gm (FX GraphModule with fused kernel calls)

    deactivate MLIR2FX

    Transform-->>Pipeline: (new_gm, TransformInfo(matches=75))
```

## Data Flow Through the Pipeline

```
Original FX Graph (1045 call_function nodes)
    │
    ▼
┌─────────────────────────────────────┐
│  FX → MLIR  (FXToMLIRConverter)     │
│                                     │
│  aten.add/mul/sub   → ad.add/mul/sub│
│  aten.neg           → ad.neg        │
│  aten.pow.T_Scalar  → ad.pow        │
│  aten.mean.dim      → ad.reduce_mean│
│  aten.rsqrt/sqrt    → ad.rsqrt/sqrt │
│  aten.silu/gelu/relu/tanh/sigmoid   │
│    → ad.silu/gelu/relu/tanh/sigmoid │
│  aten.exp/softplus  → ad.exp/…      │
│  rmsnorm variants   → ad.rmsnorm    │
│  gated rmsnorm      → ad.gated_rms… │
│  aten.to.dtype      → ad.to_dtype   │
│  aten.add(t, 1e-5)  → ad.splat+add  │
│  everything else    → ad.opaque     │
└──────────────┬──────────────────────┘
               │  MLIR ModuleOp
               ▼
┌─────────────────────────────────────┐
│  Decompose  (PatternRewriter)       │
│                                     │
│  ad.rmsnorm(x, w, eps)              │
│       ↓                             │
│  ad.mul(x, x)         # x²         │
│  ad.reduce_mean(sq)   # variance    │
│  ad.splat(eps)         # constant   │
│  ad.add(var, eps_t)    # var+eps    │
│  ad.rsqrt(var_eps)     # 1/√(v+e)  │
│  ad.mul(x, inv)        # normalize  │
│  ad.mul(normed, w)     # scale      │
│                                     │
│  ad.gated_rmsnorm(x, w, gate, …)   │
│       ↓                             │
│  ad.silu(gate) + ad.mul(x, silu_g) │
│  then rmsnorm primitives above      │
│  (order depends on norm_before_gate)│
│                                     │
│  53 rmsnorm ops → 371 primitives    │
└──────────────┬──────────────────────┘
               │  MLIR with primitives
               ▼
┌─────────────────────────────────────┐
│  Discover Subgraphs (greedy merge)  │
│                                     │
│  Walk ops in topo order.            │
│  Merge connected fusible ops.       │
│  Multi-branch: if op reads from     │
│    2 groups, merge them.            │
│  Pull in zero-operand ops (splat).  │
│                                     │
│  75 subgraphs (417 total ops)       │
│  Filter: skip if rank < 2          │
└──────────────┬──────────────────────┘
               │  List[FusibleSubgraph]
               ▼
┌─────────────────────────────────────┐
│  For each eligible subgraph:        │
│                                     │
│  ┌───────────────────────────────┐  │
│  │  Codegen (triton_emitter)     │  │
│  │                               │  │
│  │  1. Hash subgraph structure   │  │
│  │  2. Emit Triton kernel source │  │
│  │     - loads: upcast to f32    │  │
│  │     - ops: _EMIT table        │  │
│  │     - stores: downcast        │  │
│  │  3. Write to tempfile         │  │
│  │  4. importlib.load_module     │  │
│  │  5. Register custom_op       │  │
│  │  6. Register fake impl       │  │
│  └───────────────┬───────────────┘  │
│                  │ kernel_fn         │
│                  ▼                   │
│  ┌───────────────────────────────┐  │
│  │  Replace (subgraph_replace)   │  │
│  │                               │  │
│  │  1. Create AdOpaque op with   │  │
│  │     subgraph inputs           │  │
│  │  2. Add metadata for MLIR→FX  │  │
│  │  3. Wire outputs              │  │
│  │  4. Erase old ops             │  │
│  └───────────────────────────────┘  │
└──────────────┬──────────────────────┘
               │  MLIR with fused AdOpaque ops
               ▼
┌─────────────────────────────────────┐
│  MLIR → FX  (MLIRToFXConverter)     │
│                                     │
│  ad.add/mul/sub → aten.add/mul/sub  │
│  ad.neg/silu/… → aten.neg/silu/…    │
│  ad.rmsnorm → flashinfer_rms_norm   │
│  ad.gated_rmsnorm → triton_rmsnorm… │
│                                     │
│  ad.opaque(mlir_fused_{hash})       │
│       ↓                             │
│  call_function(torch.ops.auto_      │
│    deploy.mlir_fused_{hash}, ...)   │
│  getitem(result, 0)  # output 0    │
│  getitem(result, 1)  # output 1    │
│                                     │
│  Other opaques → original targets   │
└──────────────┬──────────────────────┘
               │
               ▼
          New FX GraphModule
     (with generated kernel calls)
```

## Example: What Happens to One RMSNorm

```
FX Graph:
  %torch_rmsnorm = call_function[torch_rmsnorm](added, weight, 1e-5)

  ↓ FX → MLIR

MLIR:
  %0 = ad.rmsnorm(%added, %weight) {eps = 1e-5}

  ↓ Decompose

MLIR (7 ops):
  %sq     = ad.mul(%added, %added)
  %var    = ad.reduce_mean(%sq, dim=-1, keepdim=true)
  %eps    = ad.splat(1e-5)
  %vareps = ad.add(%var, %eps)
  %inv    = ad.rsqrt(%vareps)
  %normed = ad.mul(%added, %inv)
  %result = ad.mul(%normed, %weight)

  ↓ Discover (all 7+1 ops form one subgraph with the preceding ad.add)

FusibleSubgraph:
  ops: [ad.add, ad.mul, ad.reduce_mean, ad.splat, ad.add, ad.rsqrt, ad.mul, ad.mul]
  inputs: [x, residual, weight]
  outputs: [result, added]

  ↓ Codegen

@triton.jit
def fused_kernel_abc123(in0_ptr, in1_ptr, in2_ptr, out0_ptr, out1_ptr, ...):
    v0 = tl.load(in0_ptr + row_off + offs, mask=mask).to(tl.float32)  # x
    v1 = tl.load(in1_ptr + row_off + offs, mask=mask).to(tl.float32)  # residual
    v2 = tl.load(in2_ptr + offs, mask=mask).to(tl.float32)            # weight (broadcast)
    t0 = (v0 + v1)           # add
    t1 = (t0 * t0)           # mul (x²)
    t2 = (tl.sum(t1, 0) * (1.0 / N_COLS))  # reduce_mean
    t3 = 1e-05               # splat
    t4 = (t2 + t3)           # add (var + eps)
    t5 = (1.0 / tl.sqrt(t4)) # rsqrt
    t6 = (t0 * t5)           # mul (normalize)
    t7 = (t6 * v2)           # mul (scale by weight)
    tl.store(out0_ptr + row_off + offs, t0.to(tl.bfloat16), mask=mask)  # added
    tl.store(out1_ptr + row_off + offs, t7.to(tl.bfloat16), mask=mask)  # result

  ↓ Register as torch.ops.auto_deploy.mlir_fused_abc123

  ↓ Replace in MLIR

MLIR:
  %fused:2 = ad.opaque(%x, %residual, %weight) {node_key = "mlir_fused_abc123"}

  ↓ MLIR → FX

FX Graph:
  %mlir_fused_abc123 = call_function[mlir_fused_abc123](x, residual, weight)
  %getitem   = call_function[getitem](mlir_fused_abc123, 0)  # added
  %getitem_1 = call_function[getitem](mlir_fused_abc123, 1)  # result
```
