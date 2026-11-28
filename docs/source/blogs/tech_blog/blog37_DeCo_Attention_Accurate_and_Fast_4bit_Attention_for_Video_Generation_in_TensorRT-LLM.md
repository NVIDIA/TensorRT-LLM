<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeCo-Attention: Accurate and Fast 4-Bit Attention for Video Generation in TensorRT-LLM

By NVIDIA TensorRT-LLM team

*An algorithm-system co-design that makes NVFP4 attention fast and accurate for video generation without model retraining or calibration.*

**Paper**: [DeCo-Attention: Decompose to Compensate for Low-Precision Attention](https://arxiv.org/abs/2611.32035)

**Table of Contents**
- [Introduction](#introduction)
- [Overview](#overview)
- [Background](#background)
  - [Attention Cost in Video Generation](#attention-cost-in-video-generation)
  - [Low-Precision Tensor Cores](#low-precision-tensor-cores)
  - [Attention Pipeline and Softmax](#attention-pipeline-and-softmax)
- [Error Analysis of 4-Bit Attention](#error-analysis-of-4-bit-attention)
  - [Rounding Error of NVFP4](#rounding-error-of-nvfp4)
  - [Softmax Amplification](#softmax-amplification)
- [Algorithm Design](#algorithm-design)
  - [Principal Residual Compensation](#principal-residual-compensation)
  - [Preparation Cost](#preparation-cost)
  - [Hadamard Rotation and V Smoothing](#hadamard-rotation-and-v-smoothing)
- [Kernel Design](#kernel-design)
  - [I-ExpCast](#i-expcast)
  - [Row Sum in the PV MMA](#row-sum-in-the-pv-mma)
  - [Lazy Rescale](#lazy-rescale)
- [Why It Needs Co-Design](#why-it-needs-co-design)
- [Results](#results)
  - [Speedup](#speedup)
  - [Accuracy](#accuracy)
- [Usage in TensorRT-LLM VisualGen](#usage-in-tensorrt-llm-visualgen)
- [Conclusion and Future Work](#conclusion-and-future-work)
- [Acknowledgements](#acknowledgements)
- [References](#references)

## Introduction

DeCo-Attention (Decompose to Compensate) is a low-precision attention for video diffusion transformers in TensorRT-LLM VisualGen. The query-key product runs in NVFP4, the 4-bit floating-point format of NVIDIA Blackwell and Rubin GPUs, while the generated videos stay close to the output of BF16 attention. Existing checkpoints run unchanged, without retraining or calibration data. Figure 1 places the kernel among existing attention kernels on Wan2.2 and gives the key results. DeCo-Attention, the two stars, is the fastest kernel and holds the error of the kernels that keep QK in FP8. A 4-bit query-key product therefore moves the Pareto frontier of accuracy and speed.

<p align="center">
  <img src="../media/tech_blog37_key_results.png" alt="Mean relative L2 error over all layers of Wan2.2 against the attention time on GB300, one point per attention kernel, beside the key results of DeCo-Attention" width="1000">
</p>

<p align="center"><sub><em>Figure 1. Mean relative L2 error over all layers of Wan2.2 against the attention time of one call on GB300, with one point per kernel and the key results of DeCo-Attention on Wan2.2 beside the plot. Kernels toward the lower left are both faster and more accurate.</em></sub></p>

Figure 2 shows what the error means for the generated videos. Each clip shows one prompt and seed under three attention kernels. On GB300, DeCo-Attention runs NVFP4 QK with FP8 PV. The middle clip, the uncompensated NVFP4 QK of TensorRT-LLM, drifts from the BF16 video, while DeCo-Attention follows the BF16 reference and preserves its output fidelity.

<p align="center">
  <img src="../media/tech_blog37_demo_wan22_fp8pv.gif" alt="Wan2.2: BF16 attention, the NVFP4 QK and FP8 PV attention of TensorRT-LLM, DeCo-Attention" width="860">
</p>

<p align="center"><sub><em>Wan2.2</em></sub></p>

<p align="center">
  <img src="../media/tech_blog37_demo_minimax_h3_fp8pv.gif" alt="MiniMax-H3: BF16 attention, the NVFP4 QK and FP8 PV attention of TensorRT-LLM, DeCo-Attention" width="860">
</p>

<p align="center"><sub><em>MiniMax-H3</em></sub></p>

On RTX PRO 6000, DeCo-Attention runs both products in NVFP4. The middle clip, the fully 4-bit SageAttention3, drifts from the BF16 video, while DeCo-Attention follows the BF16 reference and preserves its output fidelity.

<p align="center">
  <img src="../media/tech_blog37_demo_hunyuan15_nvfp4pv.gif" alt="HunyuanVideo-1.5: BF16 attention, SageAttention3, DeCo-Attention with NVFP4 PV" width="860">
</p>

<p align="center"><sub><em>HunyuanVideo-1.5</em></sub></p>

<p align="center">
  <img src="../media/tech_blog37_demo_longcat_nvfp4pv.gif" alt="LongCat-Video: BF16 attention, SageAttention3, DeCo-Attention with NVFP4 PV" width="860">
</p>

<p align="center"><sub><em>LongCat-Video</em></sub></p>

<p align="center"><sub><em>Figure 2. Demo clips as reduced GIF previews of the generated videos, with BF16 attention on the left, the baseline in the middle, and DeCo-Attention on the right.</em></sub></p>

This blog explains how DeCo-Attention works and why. The sections below cover where the error of a 4-bit attention comes from, how the compensation removes most of that error, and how the kernels turn the FP4 tensor-core rate into real kernel speedup. The derivations and the full experiments are in the paper.

## Overview

DeCo-Attention is a co-design of an algorithm and a kernel, and each half solves one problem of 4-bit attention. The algorithm, principal residual compensation, makes the 4-bit QK product accurate. It decomposes the rounding error of Q and K and adds the main part of that error back inside the same 4-bit matrix product, without a second product, retraining, or calibration data. The kernel, written in CuTe DSL, turns the speed of the 4-bit product into faster attention. Once both matrix products are fast, the softmax between the two products limits the kernel, and three optimizations of the softmax remove this limit. Figure 3 shows the two halves of the design.

<p align="center">
  <img src="../media/tech_blog37_overview.png" alt="Overview of DeCo-Attention: preparation per head and the attention kernel per tile" width="1000">
</p>

<p align="center"><sub><em>Figure 3. The preparation in (a) runs once per head. The attention kernel in (b) runs per tile of queries and keys.</em></sub></p>

- The preparation rotates Q and K by a Hadamard matrix H and rounds both to NVFP4. It then projects the rounding residuals on a basis B of a few principal directions, estimated from 1/256 of the tokens, and appends the projections as extra channels next to the rounded ones. V is smoothed by subtracting the block means.
- The kernel computes the scores S together with the compensation in one NVFP4 MMA over the extended operands. I-ExpCast then forms the probabilities P without the special-function unit, and the PV MMA runs over the smoothed V and a column of ones, which returns the row sum l with the output. The lazy rescale skips most rescales, and one high-precision MMA adds the block means of V back.

With both halves, DeCo-Attention is the fastest attention kernel of our comparison on GB300, VR200, and RTX PRO 6000, while its accuracy and its generation quality reach those of methods that keep QK in FP8.

## Background

### Attention Cost in Video Generation

Attention takes most of the generation time. A video diffusion transformer denoises the latent of a whole clip over tens of steps, and every self-attention layer attends over all tokens of the clip at a cost quadratic in the token count. An 81-frame 720p clip of Wan2.2 has 75,600 tokens, and generating the clip takes 302.7 s on GB300, with attention at 67.6% of every denoising step. Therefore, any large end-to-end gain has to come from attention.

### Low-Precision Tensor Cores

The tensor cores execute matrix multiply-accumulates (MMAs) in several number formats, at a rate that rises as the precision falls. BF16 is the usual format of attention in diffusion models. Hopper introduced FP8 MMAs, whose E4M3 elements have four exponent bits and three mantissa bits. Blackwell and Rubin added MMAs in block-scaled FP4 formats such as NVFP4. NVFP4 elements are E2M1 with two exponent bits and one mantissa bit, an FP8 scale per block of 16 elements, and an FP32 scale per tensor, and a block-scaled MMA multiplies NVFP4 operands directly on the tensor cores. On GB300 and VR200, FP4 MMAs run at 2 to 3x the FP8 rate and 6 to 9x the BF16 rate. Running the two matrix products of attention in FP4 is therefore the most direct way to speed up attention.

### Attention Pipeline and Softmax

For one head, attention computes two matrix products with a softmax between the two products (Figure 4). The QK product gives the scores, the softmax turns the scores into attention weights, and the PV product gives the output.

<p align="center">
  <img src="../media/tech_blog37_attention_pipeline.png" alt="The three stages of attention: QK MMA, softmax, PV MMA" width="800">
</p>

<p align="center"><sub><em>Figure 4. The three stages of attention for one head and the units that run each stage.</em></sub></p>

The two products run on the tensor cores and can use the low-precision formats, whereas the softmax runs on the special-function unit and the ALU. In low precision, the softmax limits attention in two ways. First, the softmax becomes the latency bottleneck. Once both products run in low precision, the kernel waits on the exponential, which runs on the special-function unit at far fewer scores per cycle than the MMAs and hides the FP4 speed. Second, the softmax amplifies the quantization error. With one mantissa bit, rounding changes an E2M1 element by up to 1/4 against 1/16 in FP8, and the exponential amplifies this error of Q and K in the scores.

TensorRT-LLM VisualGen already ships an NVFP4 QK, FP8 PV attention, and the next section shows what the 4-bit QK costs without compensation. In this blog, a pair such as NVFP4 / FP8 names the formats of the two products, QK first and PV second.

## Error Analysis of 4-Bit Attention

Which part of attention produces the error? We quantize one product at a time and measure the relative L2 error of the attention output against FP32 attention, on the hard layers and over all layers of Wan2.2 (Figure 5(a)). The hard layers are the three layers of Wan2.2 where per-tensor FP8 attention is least accurate. In the same format, rounding Q and K costs several times the accuracy of rounding P and V. On the hard layers, an NVFP4 QK alone gives an error of 30.1%, and an NVFP4 PV alone gives 7.1%. Therefore, the problem to solve is reducing the error of the QK product.

<p align="center">
  <img src="../media/tech_blog37_motivation.png" alt="Quantization error per product and format, and share of the energy of K held by r directions" width="560">
</p>

<p align="center"><sub><em>Figure 5. Panel (a) gives the relative L2 error of the attention output on Wan2.2 when only Q and K or only P and V are quantized, where the formats marked with a star use the per-block scales of SageAttention. Panel (b) gives the share of the energy of K held by the first r directions, over the heads of Wan2.2, for three choices of directions, the principal directions of K, the shared basis of Q and K, and the largest channels.</em></sub></p>

Two properties of attention explain why the QK product is this sensitive.

- An NVFP4 QK starts with a large rounding error. The error follows from the floating-point format itself, and neither finer scales nor a rotation removes the error.
- The softmax then amplifies the error of the scores.

### Rounding Error of NVFP4

A floating-point format with *t* mantissa bits rounds every element with a relative error up to 2<sup>−(*t*+1)</sup>, whatever the scale of the element (Figure 6). FP8 (E4M3) has three mantissa bits and NVFP4 (E2M1) has one, which allows a change of up to 1/16 of an element in FP8 and up to 1/4 in NVFP4.

<p align="center">
  <img src="../media/tech_blog37_mantissa_bits.png" alt="Representable values of FP8 and FP4 elements between 1 and 4" width="900">
</p>

<p align="center"><sub><em>Figure 6. The representable values of an FP8 element and an FP4 element between 1 and 4, and where the value 1.3 rounds in each format.</em></sub></p>

Finer scales only decide where the representable range sits, and a rotation only changes which values are rounded. Neither adds mantissa bits, and the relative error of every element stays. Figure 5(a) shows the effect directly for the two formats with the same per-block scales, where a floating-point FP8 QK has an error of 12.6% on the hard layers against 1.0% for an integer INT8 QK, and NVFP4 has the finest scales and the largest error. Q and K also hold outliers, a few channels with very large values whose rounding errors are large in absolute terms.

### Softmax Amplification

A quantized kernel computes the scores from the rounded Q and K. The score of key *j* therefore carries an error *e*<sub>j</sub>, and the softmax sees *s*<sub>j</sub> + *e*<sub>j</sub> in place of the exact score *s*<sub>j</sub>, which multiplies every attention weight by the exponential of the score error of that key before the weights are normalized again.

$$\tilde p_j = \frac{p_j \exp(e_j)}{\sum_l p_l \exp(e_l)} \approx p_j \,(1 + e_j - \bar e), \qquad \bar e = \sum_l p_l e_l$$

The approximation holds while the errors are small, where exp(*x*) ≈ 1 + *x*, and leads to two conclusions.

- An error that all keys of a query share cancels, because such an error moves *e*<sub>j</sub> and *ē* alike.
- A weight changes in proportion to the centered error *e*<sub>j</sub> − *ē*, about 10% for an error of 0.1.

The approximation breaks as a centered error approaches 1.0, where exp(*e*<sub>j</sub>) grows much faster than 1 + *e*<sub>j</sub>. With an NVFP4 QK the centered errors do exceed 1.0, and the exponential amplifies the errors so the attention output tracks the wrong keys. Figure 7 shows one query on a hard layer of Wan2.2, where without compensation the top key loses about 60% of the weight, the second key gains about 70%, and the two swap places. With the compensation, the errors stay within about a third and the weights stay near the first-order line.

<p align="center">
  <img src="../media/tech_blog37_score_error.png" alt="Weight change against the centered score error for one query" width="520">
</p>

<p align="center"><sub><em>Figure 7. Relative change of the attention weight of each key against its centered score error for one query of Wan2.2, without and with compensation. The marker size indicates the weight of the key.</em></sub></p>

The PV product is not followed by an exponential. The rounding errors of P and V therefore add up in the output without amplification, which is why attention tolerates a low-precision PV much better.

## Algorithm Design

Earlier low-precision attention, such as SageAttention and VC-Attention, reduces the quantization error by smoothing Q and K or by finer scales. A floating-point element keeps its relative rounding error at any scale, however, so the rounding error of Q and K stays in the scores. An accurate 4-bit QK therefore cannot come from better scales alone, and the rounding error itself has to be treated. DeCo-Attention treats this error numerically. It splits the error of the 4-bit score into its terms and adds the main part back inside the QK product.

### Principal Residual Compensation

Rounding a tensor *x* gives a low-precision tensor *x̂* and leaves a residual δ*x* = *x* − *x̂*, which is known exactly after the rounding. For a query *q* and a key *k*, the exact score therefore has four terms.

$$q \cdot k = \hat q \cdot \hat k + \delta q \cdot \hat k + \hat q \cdot \delta k + \delta q \cdot \delta k$$

A 4-bit kernel computes only the first of the four terms. The two first-order terms, δ*q* · *k̂* and *q̂* · δ*k*, hold most of the error of the 4-bit score, and the second-order term δ*q* · δ*k* is small enough to leave out. Adding the two first-order terms back exactly would take two more full-width matrix products, which costs more than running QK in FP8. A cheaper compensation has to know where the first-order error lies.

Take the Q-side term of one query and add up its square over all keys. The total splits along the eigenvectors *u*<sub>m</sub> of the Gram matrix of the quantized keys.

$$\sum_j (\delta q \cdot \hat k_j)^2 = \sum_{m=1}^{d} \lambda_m \, (\delta q \, u_m)^2$$

Each direction contributes its eigenvalue λ<sub>m</sub>, the energy of the keys along the direction, times the squared component of the residual along the direction. The eigenvalues are strongly concentrated (Figure 5(b)). On Wan2.2, the 32 leading eigenvectors hold a median of 74.3% of the energy of K over the heads, while the 32 largest channels hold 46.9%. The error is therefore weighted toward a few leading directions, and these are the directions to compensate.

DeCo-Attention keeps each first-order term along the *r* leading eigenvectors and computes the kept part inside the QK product. Figure 8 states the method.

<p align="center">
  <img src="../media/tech_blog37_compensation_equations.png" alt="The exact score as four terms, the first-order term kept along r leading directions, and the extended operands" width="650">
</p>

<p align="center"><sub><em>Figure 8. Principal residual compensation.</em></sub></p>

The first line repeats the decomposition of the exact score. The second line approximates a first-order term in an orthonormal basis of the head, where most of the sum over the *d* directions lies along the leading eigenvectors of the keys. With *B*<sub>K</sub> holding the *r* leading eigenvectors, the term becomes a dot product of *r* numbers, and the K-side term is treated the same way along the *r* leading eigenvectors *B*<sub>Q</sub> of the queries. The third line appends the two dot products as extra channels of the same product (Figure 9), and *q*′ · *k*′ then gives the quantized score plus both corrections in one block-scaled NVFP4 MMA. With *r* = 32, a head of 128 channels grows to 192 without a second matrix product.

<p align="center">
  <img src="../media/tech_blog37_extended_operands.png" alt="The extended operands and the three terms of the product q prime times k prime" width="500">
</p>

<p align="center"><sub><em>Figure 9. The extended operands and the three terms of the product q′ · k′.</em></sub></p>

More directions remove more error but widen the product. Figure 10(a) shows the trade-off on Wan2.2 and MiniMax-H3. The first 16 directions remove most of the error, and the step to 32 still helps on the hard layers. DeCo-Attention therefore uses *r* = 32 on every model and GPU. Figure 10(b) splits the QK error that remains after the compensation into parts. The largest part is the first-order error in the dropped directions, which a larger *r* would reduce, and the second-order term is the smallest, which justifies leaving the term out.

<p align="center">
  <img src="../media/tech_blog37_compensated_error.png" alt="Kernel error against the number of directions, the parts of the remaining QK error, and the error per compensation term" width="1000">
</p>

<p align="center"><sub><em>Figure 10. Panel (a) gives the kernel error with FP8 PV against the number of compensated directions r. Panel (b) gives the three parts of the QK error that remain after the compensation with r = 32 on Wan2.2, each measured as the relative L2 error that part alone causes. Panel (c) gives the kernel error on Wan2.2 with FP8 PV when the compensation adds neither term, one term, or both terms. All four cases of (c) use the Hadamard rotation and V smoothing.</em></sub></p>

The two first-order terms are also not equally important, because the softmax passes on only the differences of the score errors between the keys of a query. In the K-side term, the residuals of different keys are multiplied by the whole query and do not cancel. In the Q-side term, the same residual δ*q* meets every key, and only the differences between keys remain. Figure 10(c) confirms the ordering of the two terms, since the K-side term alone removes most of the error and the Q-side term adds the rest.

### Preparation Cost

Because the principal directions depend on the data, DeCo-Attention estimates the directions at run time for every head of every call. Three choices keep the cost of the estimate low.

- One basis *B* replaces *B*<sub>K</sub> and *B*<sub>Q</sub>, since Q and K project the same activations and the leading subspaces of Q and K overlap by 0.75 on Wan2.2, against 0.25 for unrelated subspaces. Each Gram matrix is divided by the trace so that Q and K weigh equally, and the *r* leading eigenvectors of the summed Gram matrices form the basis. One eigendecomposition per head then serves both operands at almost no loss, 2.80% error on the hard layers of Wan2.2 against 2.91% with separate bases.
- The Gram matrices come from a random sample of 1/256 of the tokens, at least 64, which costs almost no kernel accuracy on Wan2.2.
- The preparation is a short chain of fused kernels, linear in the sequence length where attention is quadratic. On Wan2.2 the chain takes 1.55 ms on GB300 and 1.05 ms on VR200 and adds 2.7% to 3.6% to the attention time.

### Hadamard Rotation and V Smoothing

Two supporting techniques complete the algorithm, and the paper gives the details of both. Before rounding, DeCo-Attention rotates Q and K by a normalized Hadamard matrix with random row signs, which leaves the scores unchanged. The rotation makes the values of a block similar in size, which lowers the rounding error. It also spreads the residuals evenly over all directions, so that the error along a direction is proportional to its eigenvalue and no longer depends on the input. For the PV product, DeCo-Attention applies V smoothing, introduced by VC-Attention. The mean of each block of keys is subtracted from V before rounding and added back by one high-precision MMA, which keeps the attention output unbiased.

## Kernel Design

An accurate 4-bit QK is half of the work. With both matrix products in low precision, the kernel no longer waits on the tensor cores but on the softmax between the products, and mostly on the exponential. The exponential runs on the special-function unit (SFU) through MUFU instructions, at far fewer scores per cycle than the tensor cores. Earlier kernels lower this cost at a price. FlashAttention-4 emulates part of the exponentials on the fused multiply-add (FMA) units, which takes many more instructions. ExpCast, from VC-Attention, and Direct-P build the code of each probability from a linear function of the score, which is fast, but neither rounds to the nearest code nor covers the subnormal codes.

We implement the kernels of DeCo-Attention in CuTe DSL for GB300, VR200, and RTX PRO 6000. The format of the PV product depends on the GPU, FP8 on GB300 and VR200 where a 4-bit PV would gain little speed, and NVFP4 on RTX PRO 6000. The kernels shorten the softmax with I-ExpCast, the row sum in the PV MMA, and the lazy rescale.

### I-ExpCast

I-ExpCast, our improved ExpCast, does not use the MUFU instructions and still rounds correctly. One multiply-add builds the bit pattern of each probability from the score, which a hardware conversion then rounds to the nearest FP8 or FP4 code. The whole step takes three instructions per pair of scores and nearly matches the exact exponential.

The idea is to form each probability in FP16 and leave the final rounding to a hardware conversion. FP16 has ten mantissa bits and represents every value of E4M3 and of E2M1 exactly. A conversion from FP16 therefore rounds to the nearest code of either format, subnormal codes included. What remains is to produce the FP16 value cheaply. Let *x* be the base-2 logarithm of a scaled probability, which is a linear function of the score. Read as an integer, the FP16 bit pattern of 2<sup>*x*</sup> is close to a linear function of *x*.

$$\mathrm{bits}_{\mathrm{FP16}}\big(2^x\big) \approx 1024\,(x + 15)$$

I-ExpCast evaluates this linear function in the subnormal range of FP32, where the values are evenly spaced and the bit pattern of a value is the multiple of the spacing. One FMA therefore maps a score to the bit pattern and rounds the pattern to an integer in the same step. A saturating pack then narrows the pattern to the 16 bits of FP16, and the last step converts the FP16 value to E4M3 or E2M1.

$$\hat p_j = \mathcal{Q}\Big(\mathrm{sat}_{16}\big(\mathrm{bits}_{\mathrm{FP32}}(a\,s_j + c)\big)\Big)$$

The same routine serves FP8 and FP4 probabilities, and Figure 11 compares the decoded probabilities with the exact ones. I-ExpCast matches the exact exponential with rounding over the whole range including the subnormal codes, where ExpCast drops to zero. Direct-P departs from the exact value over much of the range. With exact scores and only P rounded to FP4, the Direct-P codes give an error of 5.06% over all layers of Wan2.2 against 1.90% for I-ExpCast.

<p align="center">
  <img src="../media/tech_blog37_p_codes.png" alt="Decoded value of P over the exact value, for each way of forming the codes" width="480">
</p>

<p align="center"><sub><em>Figure 11. Ratio of the decoded probability to the exact probability for each way of forming the codes, for (a) FP8 and (b) FP4 probabilities.</em></sub></p>

Inside the kernel, the way of forming P shows directly in the time (Figure 12). Computing the exponential on the SFU takes 21.5% longer with FP8 PV and 34.3% longer with NVFP4 PV. The polynomial of FlashAttention-4 does not close this gap. A score pair takes 13 instructions on the polynomial against 4 on the SFU, and with the polynomial on one eighth of the score pairs the FP8-PV kernel takes 25.7% longer than with I-ExpCast, more than with the SFU alone. ExpCast is close in time, 3.6% longer than I-ExpCast. Direct-P needs the fewest instructions for this step. The Direct-P bar (†) times the released Direct-P kernel as a whole, which is a different implementation from the other bars and takes 40.2% longer than the NVFP4-PV kernel of DeCo-Attention.

<p align="center">
  <img src="../media/tech_blog37_forming_p.png" alt="Kernel time per way of forming the probabilities" width="760">
</p>

<p align="center"><sub><em>Figure 12. Kernel time of one Wan2.2 call on GB300 for each way of forming the probabilities.</em></sub></p>

### Row Sum in the PV MMA

The softmax needs the sum of each row of probabilities, and computing the sum next to the exponentials adds work where the kernel is already limited. Since the tensor cores have spare capacity, a column of ones appended to V lets the PV MMA return the row sum of P with the output, which moves the work to the tensor cores.

Figure 13 compares this row sum with a sum on the ALU. An ALU sum of the same E4M3 values gives the same error and takes 7.2% longer, so the row sum in the PV MMA costs no accuracy. A sum of the probabilities before the rounding to E4M3 is also slower and slightly less accurate. The output is the PV product divided by the row sum, so when the row sum uses the unrounded P while the product uses the rounded P, the attention weights no longer sum to one and the output is biased.

<p align="center">
  <img src="../media/tech_blog37_row_sum.png" alt="Row sum of P in the PV MMA and on the ALU" width="520">
</p>

<p align="center"><sub><em>Figure 13. Attention time of a Wan2.2 call and relative L2 error (%) of the kernel with FP8 PV on GB300 without V smoothing, with the row sum of P in the PV MMA and on the ALU.</em></sub></p>

### Lazy Rescale

The online softmax rescales the accumulated output whenever a new tile raises the running row maximum. The lazy rescale moves the base only when the maximum grows by more than a factor of 2<sup>*T*</sup>, which skips most rescales. A larger *T* skips more of them but costs accuracy, and how much depends on the format of P.

An FP8 P is the more sensitive of the two. FP8 has a fixed range, and a larger *T* keeps more of it free at the top for probabilities that grow above the base. That leaves less room at the bottom for the small probabilities. With *T* = 8, the setting of FlashAttention-4, probabilities below 2<sup>−9</sup> of the base underflow to zero, and the error over all layers of Wan2.2 is 4.50% (Figure 14). With *T* = 4, probabilities survive down to 2<sup>−13</sup> and the error is 3.73%, against 3.70% with a rescale at every tile, while the kernel runs within 0.5% of the time of *T* = 8.

An NVFP4 P hardly reacts to *T*. NVFP4 has two levels of scale, a global scale and one scale per block of 16 keys, and the room for *T* is taken from these scales. Inside a block, the largest probability still maps to the largest 4-bit value, so the 4-bit values use their full range at every *T*. FP8 has no block scale, and the room for *T* is taken from the values themselves. The error with NVFP4 PV therefore changes only in the second decimal from *T* = 0 to *T* = 8.

DeCo-Attention therefore uses *T* = 4 for both formats, as the balance of accuracy and speed.

<p align="center">
  <img src="../media/tech_blog37_lazy_rescale_threshold.png" alt="Kernel time and error at three thresholds of the lazy rescale" width="520">
</p>

<p align="center"><sub><em>Figure 14. Attention time of a Wan2.2 call and relative L2 error (%) of the kernels with FP8 PV and NVFP4 PV on GB300 at three thresholds T of the lazy rescale.</em></sub></p>

## Why It Needs Co-Design

Figure 15 shows why the compensation and the kernel optimizations were designed together.

<p align="center">
  <img src="../media/tech_blog37_codesign_ablation.png" alt="Time, speedup, SM clock and error as the pieces are added" width="1000">
</p>

<p align="center"><sub><em>Figure 15. The pieces of DeCo-Attention added one at a time to the kernel with FP8 PV on GB300, with the time of one Wan2.2 call, the speedup over BF16 attention, the SM clock under the power limit, and the error on the hard layers. The FP8-QK reference is the same kernel with an uncompensated FP8 QK and one scale per tensor.</em></sub></p>

The ablation gives three conclusions:

- **A 4-bit QK needs compensation.** The NVFP4 QK, FP8 PV attention of TensorRT-LLM has an error of 30.56% on the hard layers, and the compensation lowers the error to 2.92%.
- **The softmax bottleneck has to be removed to turn the 4-bit QK into real speedup.** With compensation only, the softmax bounds the kernel and hides the gain of the 4-bit QK, 79.0 ms against 82.1 ms for the same kernel with an FP8 QK. I-ExpCast, the row sum in the PV MMA, and the lazy rescale bring the kernel to 55.8 ms. The optimized kernel also does more work per cycle and draws more power, which lowers the SM clock under the power limit from 2017 to 1582 MHz and gives back part of the gain.
- **The kernel optimizations must not cost accuracy.** I-ExpCast and the lazy rescale change the numerics of the softmax, yet the error stays between 2.92% and 2.95% through all three additions, because I-ExpCast rounds to the nearest code, subnormal codes included, and therefore combines with the lazy rescale. V smoothing, the last piece, costs 2.2 ms and lowers the error to 2.80%.

This is why DeCo-Attention is a co-design, since neither half is enough alone. The compensation alone gives an accurate 4-bit QK that is hardly faster than an FP8 QK. The kernel optimizations alone gain less, because with an FP8 QK the slower FP8 MMAs bound the kernel earlier. The same four changes speed up the FP8-QK reference by 1.20x against 1.36x with the NVFP4 QK, or 1.67x at a fixed clock. **Together, the two halves bring out both the accuracy and the speed of 4-bit attention.** The complete kernel is 1.17x faster than the optimized FP8-QK reference and more accurate on the hard layers, and the MMAs nearly bound its time.

Both halves are also general. The compensation needs no training, calibration, or model-specific code, and the softmax optimizations work for FP8 and FP4 probabilities alike, so both carry over to other models and attention kernels.

## Results

We evaluate DeCo-Attention at the kernel level and end to end. At the kernel level, we measure the error on attention calls captured from the models and the attention time, and end to end we measure the quality of the generated videos and the generation time in TensorRT-LLM VisualGen. The models are Wan2.2, LongCat-Video, HunyuanVideo-1.5, and MiniMax-H3, on GB300, VR200, and RTX PRO 6000.

### Speedup

Figure 16 gives the speed of the kernels and of the whole generation, with DeCo-Attention running FP8 PV on GB300 and VR200 and NVFP4 PV on RTX PRO 6000. On Wan2.2, the DeCo kernel runs 2.13x, 2.49x, and 2.82x faster than the fastest BF16 kernel on the three GPUs, and the whole generation runs 1.55x, 1.68x, and 1.80x faster, while on MiniMax-H3 the speedups are similar.

<p align="center">
  <img src="../media/tech_blog37_speedup.png" alt="Kernel-level and end-to-end speedup on GB300, VR200 and RTX PRO 6000" width="1000">
</p>

<p align="center"><sub><em>Figure 16. Speedup of the attention kernel over the fastest BF16 kernel in (a) and of the whole generation in TensorRT-LLM VisualGen over BF16 attention in (b). The SageAttention and VC-Attention bars show the FP8-QK kernels of both methods on GB300 and VR200 and the 4-bit kernels on RTX PRO 6000.</em></sub></p>

DeCo-Attention is the fastest of the compared attention kernels on every GPU. On GB300 and VR200 the DeCo kernel is 1.44x and 1.65x faster than the fastest prior method, VC-Attention, with the generation 1.22x and 1.32x faster, and on RTX PRO 6000, where SageAttention3 is the fastest prior method, the DeCo kernel is 1.08x faster at the kernel level and 1.09x end to end.

### Accuracy

Figure 17 gives the kernel error on the hard layers and over all layers of Wan2.2 and MiniMax-H3. With FP8 PV, DeCo-Attention has the lowest error of all kernels with an FP8 or a 4-bit QK on the hard layers of both models, and over all layers the DeCo error lies between the errors of VC-Attention and SageAttention2 with an FP8 QK. With NVFP4 PV, DeCo-Attention has the lowest error of the fully 4-bit kernels. On the hard layers of Wan2.2, the DeCo error is about a quarter of the error of SageAttention3 and of the 4-bit VC-Attention, and below that of SageAttention2 with an FP8 QK.

<p align="center">
  <img src="../media/tech_blog37_kernel_accuracy.png" alt="Kernel-level error on Wan2.2 and MiniMax-H3 with FP8 PV and with NVFP4 PV" width="1000">
</p>

<p align="center"><sub><em>Figure 17. Kernel-level accuracy as the relative L2 error of the attention output against FP32 attention, where lower is better, for the kernels with FP8 PV in (a) and with NVFP4 PV in (b). The SageAttention and VC-Attention bars show the FP8-QK kernels of both methods in (a) and the 4-bit kernels in (b). BF16 attention has an error of 0.15% to 0.17%.</em></sub></p>

For the end-to-end quality we generate videos for the 100 prompts of MovieGen Bench and compare each video with the BF16-attention video for the same prompt and seed (Figure 18). PSNR and SSIM are higher when two videos agree more, while LPIPS tracks what a viewer notices and is lower. A diffusion model amplifies any numeric difference along the sampling trajectory, which is why even two BF16 kernels do not produce identical videos. The BF16 row scores the BF16 attention of cuDNN against the reference and shows the scale.

<p align="center">
  <img src="../media/tech_blog37_e2e_quality.png" alt="End-to-end quality on four video models against BF16 attention" width="1000">
</p>

<p align="center"><sub><em>Figure 18. End-to-end quality on four video models against BF16 attention, as the mean over 100 prompts. QK/PV gives the bit widths of the two products, and DeCo-Attention is in bold. Blue cells mark the best PSNR, SSIM, and LPIPS within each group with a 4-bit QK. SageAttention2 and VC-Attention use the default INT8 QK on B200, and the rows with an FP8 QK run on GB300. The means marked † are reported by VC-Attention.</em></sub></p>

With FP8 PV, DeCo-Attention reaches the quality of the FP8-QK methods. DeCo-Attention is more accurate than MXFP8 and than SageAttention2 with an FP8 QK on every model. Against VC-Attention with an FP8 QK, DeCo-Attention is more accurate on Wan2.2 and LongCat-Video and less on the other two models. Compared with the existing NVFP4 QK, FP8 PV attention, the compensation gains about 3 to 5 dB of PSNR on every model. DeCo-Attention also narrows the gap of a 4-bit QK to INT8-QK attention, from 5 to 6 dB of PSNR behind VC-Attention with an INT8 QK to 1 to 3 dB. The INT8-QK methods need the INT8 tensor cores of B200, whereas DeCo-Attention runs on the FP4 tensor cores of GB300, VR200, and RTX PRO 6000. With NVFP4 PV, DeCo-Attention has better PSNR and LPIPS than SageAttention3 and the 4-bit VC-Attention on every model, and the fully NVFP4 DeCo kernel also matches SageAttention2 with an FP8 QK on every model, with both products in 4 bits.

## Usage in TensorRT-LLM VisualGen

DeCo-Attention is integrated into TensorRT-LLM VisualGen as an attention backend. The kernels are written in CuTe DSL with FP8 PV for GB300 and VR200 and NVFP4 PV for RTX PRO 6000, and run on existing checkpoints of Wan2.2, HunyuanVideo-1.5, and MiniMax-H3. DeCo-Attention is a switch on the quantized attention configuration of VisualGen and does not need calibration or model-specific code. The following YAML enables DeCo-Attention with FP8 PV.

```yaml
attention_config:
  backend: CUTEDSL
  quant_attention_config:
    qk_dtype: nvfp4
    v_dtype: fp8
    deco_attention: true
```

For NVFP4 PV, set `v_dtype` to `nvfp4`. Pass the file to the server with `--visual_gen_args`.

```bash
trtllm-serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --visual_gen_args config.yml
```

The same configuration in Python:

```python
args = VisualGenArgs(
    model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    attention_config=AttentionConfig(
        backend="CUTEDSL",
        quant_attention_config=QuantAttentionConfig(
            qk_dtype="nvfp4",
            v_dtype="fp8",
            deco_attention=True,
        ),
    ),
)
```

NVFP4 QK with FP8 PV runs on SM100, SM103, SM107, and SM120, which covers B200, GB300, VR200, and RTX PRO 6000, while NVFP4 PV runs on SM120 only. All options of the quantized attention are described in the [VisualGen quantized attention guide](../../features/visualgen-quantized-attention.md).

## Conclusion and Future Work

DeCo-Attention shows that the 4-bit tensor cores can serve the QK product of attention without giving up the quality of the generated video. The rounding error of a 4-bit QK lies mostly along a few principal directions, and compensating that error inside the same matrix product makes the 4-bit QK accurate, while softmax optimizations turn the speed of the FP4 MMAs into kernel speedup. The result is the fastest attention kernel of our comparison on GB300, VR200, and RTX PRO 6000, at the generation quality of FP8-QK methods. Three directions of future work remain.

- The principal directions are estimated at run time for every head of every call, which adds 2.7% to 3.6% to the attention time, and reducing this cost matters most in low-latency scenarios.
- The kernels target the self-attention of video diffusion models today, and bringing DeCo-Attention to LLM inference with a KV cache is the next step.
- We will keep improving the integration in TensorRT-LLM VisualGen and bring DeCo-Attention to more models and use cases.

## Acknowledgements

DeCo-Attention is a joint effort across several NVIDIA teams, from the algorithm and the numerical analysis to the CuTe DSL kernels, the integration in TensorRT-LLM VisualGen, and the evaluation. We thank everyone who took part in this work, and the colleagues who reviewed the work and shared feedback along the way.

## References

- Y. He et al. DeCo-Attention: Decompose to Compensate for Low-Precision Attention. [arXiv:2611.32035](https://arxiv.org/abs/2611.32035), 2026.
- T. Zadouri et al. FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling. MLSys, 2026.
- J. Zhang et al. SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization. ICML, 2025.
- J. Zhang et al. SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training. NeurIPS, 2025.
- X. Li et al. VC-Attention: Value Smoothing and Softmax Casting for Low-bit Attention. [arXiv:2609.15810](https://arxiv.org/abs/2609.15810), 2026.
- P. Zhang et al. Attn-QAT: 4-Bit Attention With Quantization-Aware Training. ICML, 2026.
- R. Hu. Hardware-Aware FP4 FlashAttention-4 (Direct-P). [arXiv:2609.04105](https://arxiv.org/abs/2609.04105), 2026.
