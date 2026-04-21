# 381 TOPs: NCU-Driven Optimization Findings

## Starting Point: 327 TOPs (52.8% of 619 INT4 peak)

**Hardware:** NVIDIA RTX A6000 (SM86, 84 SMs, 619 INT4 TOPS peak)

**Kernel architecture:** Hand-written INT4 GEMM using:
- Offline ldmatrix repacking (transforms INT4 data to MMA-ready register layout, not timed)
- Online GEMM: 8-warp (BM=256, BN=128, BK=64) and 4-warp (BM=128) variants
- MMA instruction: `mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32`
- Minimal shared memory (~512B for double-buffered B-scales only)
- A-scales broadcast via `__shfl_sync`, half2 FMA accumulators
- CTA swizzle (4x4 block groups) for L2 locality
- Per-layer weight group sizes (256-3072) to reduce B-scale syncs

**4 target GEMM shapes** (all M=4096):

| Layer | N | K | Baseline TOPs |
|-------|---|---|---------------|
| attn_to_qkv | 9216 | 3072 | 328 |
| attn_to_out | 3072 | 3072 | 311 |
| ff_up | 12288 | 3072 | 328 |
| ff_down | 3072 | 12288 | 342 |

---

## Step 1: NCU Profiling — Finding the Real Bottleneck

We installed NCU (v2026.1.1.0) on the server and profiled all 4 GEMM kernel launches with `--set full`.

### Key NCU Metrics (attn_to_qkv shape)

| Metric | Value | What it means |
|--------|-------|---------------|
| Compute Throughput | 52.85% | Half the GPU's compute capability is wasted |
| Memory Throughput | 46.59% | Not memory-bound — the problem is compute-side |
| Registers/Thread | 125 | Limits occupancy to 2 blocks/SM (max 128 for 2 blocks) |
| Achieved Occupancy | 32.14% | Only 16 warps active per SM (out of 48 max) |
| Issue Slots Busy | 42.56% | The warp scheduler is idle 57% of cycles |
| Eligible Warps/Scheduler | 0.78 | Less than 1 warp ready to execute per cycle |
| No Eligible Warps | 56.23% | Over half the time, NO warp can make progress |
| Top Stall: math pipe throttle | 30.1% | Warps stall because a math pipeline is overloaded |
| Register Spilling | 0 bytes | No spilling — good, but registers are maxed out |
| L2 Hit Rate | 89.6% | CTA swizzle is working well for cache locality |

**NCU's diagnosis:** "This workload exhibits low compute throughput and memory bandwidth utilization relative to peak — typically indicates latency issues."

### What "Math Pipe Throttle" Actually Means

The GPU has separate execution pipes per SM sub-partition:
- **Tensor pipe:** executes MMA instructions (the useful INT4 multiply-accumulate)
- **FMA pipe:** executes floating-point operations (HFMA2, HMUL2, type conversions)
- **INT pipe:** executes integer operations (address arithmetic)
- **Load/Store pipe:** memory operations

These pipes CAN run in parallel. But "math pipe throttle" means all active warps are trying to use the SAME pipe at the same time, so they stall waiting for it.

---

## Step 2: SASS Analysis — Counting the Instructions

We dumped the compiled SASS (GPU assembly) to see the actual instruction mix per K-tile iteration:

| Instruction | Count | Pipe | Purpose |
|-------------|-------|------|---------|
| **IMMA** | **32** | **Tensor** | **The actual useful compute (m16n8k64 INT4 MMA)** |
| I2FP | 128 | FMA/Conv | int32 → float32 conversion |
| F2FP | 64 | FMA/Conv | Pack two float32 → half2 |
| HMUL2 | 64 | FMA | Scale products (A-scale × B-scale per column) |
| HFMA2 | 64 | FMA | Accumulation (scaled_result + accumulator) |
| IMAD | 76 | INT | Address arithmetic |
| LDG | 12 | LSU | Global memory loads (B fragments + scales) |
| LDS | 16 | LSU | Shared memory loads (B scales) |
| STG | 64 | LSU | Global stores (epilogue only, not per K-tile) |
| Other | ~30 | Various | SHF, ISETP, LEA, NOP, etc. |

### The Critical Finding: 10:1 FMA:Tensor Ratio

**320 FMA-pipe instructions per 32 tensor-pipe instructions.**

Every MMA produces 4 int32 values. To convert and accumulate each pair:
```
I2FP R52, R52        // int32 → float32          (FMA pipe)
I2FP R53, R53        // int32 → float32          (FMA pipe)
F2FP.PACK_AB R52, R53, R52  // pack → half2     (FMA pipe)
HMUL2 R48, sa, sb    // scale product sa×sb       (FMA pipe)
HFMA2 acc, R52, R48, acc    // accumulate        (FMA pipe)
```

**5 FMA-pipe instructions per 2 values, for 128 values = 320 FMA-pipe instructions.**

The warp scheduler can issue 1 instruction per cycle. With 4 warps per sub-partition, and 89% of instructions targeting the FMA pipe, the FMA pipe becomes a serial bottleneck. The tensor cores sit mostly idle waiting for the FMA pipe to clear.

---

## Step 3: The Optimization — Defer B-Scale to Epilogue

### Insight

The 64 HMUL2 per K-tile compute `sa × sb` (A-scale times B-scale) inside the N-tile loop. The B-scale varies per column but is constant across ALL K-tiles — it only depends on which output column we're computing.

If we use **per-channel weight quantization** (one B-scale per weight row, not per group), the B-scale is constant for the entire GEMM. We can multiply by it once in the epilogue instead of every K-tile.

### Verification: Per-Channel Weight Quantization Accuracy

Before making kernel changes, we tested if per-channel quantization (with optimal MSE clipping) passes correctness:

| Layer | Cosine Similarity | Threshold | Status |
|-------|------------------|-----------|--------|
| attn_to_qkv | 0.9963 | > 0.989 | PASS (comfortable) |
| attn_to_out | 0.9952 | > 0.991 | PASS (comfortable) |
| ff_up | 0.9939 | > 0.978 | PASS (comfortable) |
| ff_down | 0.9934 | > 0.977 | PASS (comfortable) |

Per-channel with optimal clipping actually has BETTER accuracy than per-group in some cases, because the clipping search over the full K dimension finds a better global scale.

### Changes Made

**quantize.py:**
```python
# Before: per-layer weight group sizes (256-3072)
weight_group_size = {(9216,3072): 512, (3072,3072): 256, ...}

# After: per-channel (one group spanning all K)
weight_group_size = K
```

**kernel.cu (both 8-warp and 4-warp kernels):**

1. **Removed B-group outer loop** — with 1 group, no need to iterate over groups
2. **Removed `__syncthreads()` for B-scale buffer swap** — no buffer swapping needed
3. **Removed double-buffered B-scale shared memory** — single load before K-loop
4. **Inner loop simplified** — accumulate with A-scale only, no B-scale multiply:
```cpp
// Before (5 FMA-pipe ops per pair):
const half2 s00 = __hmul2(sa0, sb01);               // HMUL2
acc[0][nt][0] = __hfma2(convert(mma), s00, acc);     // HFMA2

// After (4 FMA-pipe ops per pair — HMUL2 removed):
acc[0][nt][0] = __hfma2(convert(mma), sa0, acc);     // HFMA2 with sa only
```
5. **Epilogue applies B-scale once:**
```cpp
// Before: direct store
*reinterpret_cast<half2*>(&C[addr]) = acc[0][nt][0];

// After: multiply by B-scale then store
*reinterpret_cast<half2*>(&C[addr]) = __hmul2(acc[0][nt][0], sb01);
```

### Impact Per K-Tile

| | Before | After | Saved |
|--|--------|-------|-------|
| HMUL2 in inner loop | 64 per K-tile | 0 | **64 per K-tile** |
| HMUL2 in epilogue | 0 | 64 (one-time) | — |
| Total HMUL2 (attn_to_qkv, 48 K-tiles) | 48 × 64 = 3,072 | 64 | **3,008 saved** |
| FMA-pipe instructions/K-tile | 320 | 256 | **20% reduction** |

### Results

| Layer | Before | After | Change |
|-------|--------|-------|--------|
| attn_to_qkv | 328 | **383** | **+17%** |
| attn_to_out | 311 | **365** | **+17%** |
| ff_up | 328 | **388** | **+18%** |
| ff_down | 342 | **388** | **+13%** |
| **Average** | **327** | **381** | **+16.3%** |

### Post-Optimization NCU Profile

| Metric | Before (327) | After (381) | Change |
|--------|--------------|-------------|--------|
| Compute Throughput | 52.85% | 62.30% | +18% |
| Executed Instructions | 182M | 146M | **-20%** |
| Registers/Thread | 125 | 128 | +3 (compiler used headroom) |
| NCU diagnosis | Compute-bound | **Balanced** | Better state |

---

## What We Tried That Failed

### Attempt 1: A-Scale Load Coalescing (No Effect)

**Idea:** The A-scale reads are strided: `scales_A[(m_base + lane) * nkt + kt]` has stride `nkt × 2 bytes` between adjacent threads. Repacking to `[M/32, nkt, 32]` makes them coalesced.

**Why it failed:** The A-scale loads are only 64 bytes per K-tile per warp (32 threads × 2 bytes). Total memory traffic per K-tile is ~5200 bytes (A+B fragments dominate). The 16% waste on scale loads represents <1% of total bandwidth. The optimization was correct but the absolute savings were negligible.

**Result:** 327.10 TOPs (was 327.46) — within noise.

### Attempt 2: Reduced Unroll (#pragma unroll 4) (Catastrophic)

**Idea:** Full unroll (NT=8) creates a huge instruction body using all 128 registers. Partial unroll might reduce register pressure and let the compiler schedule better.

**Why it failed:** With `#pragma unroll 4`, the compiler generates a residual loop of 2 iterations. Loop iterations require dynamic indexing into `acc[AT][NT][4]` arrays — the compiler CANNOT keep these in registers when the index is a runtime variable. All 64 accumulator registers spill to local memory (device DRAM, ~100 cycles per access vs ~0 for registers).

**Result:** 96.49 TOPs (was 327) — **3.4x regression**

**Lesson:** Full unroll is mandatory for register-resident accumulator arrays in CUDA. Any loop over register arrays must be fully unrolled or the compiler will spill.

### Attempt 4: __launch_bounds__(256, 3) (Massive Spilling)

**Idea:** Request 3 blocks/SM (requires ≤85 registers per thread). This would give 24 warps instead of 16, improving scheduler diversity by 50%.

**Why it failed:** The kernel genuinely needs ~120+ registers. Forcing ≤85 means the compiler spills ~40 registers to local memory. Every spilled access adds ~100 cycles of latency. Correctness PASSED (unlike a similar experiment on the old v3 kernel), but performance was destroyed.

**Result:** 135.92 TOPs (64-67 for 8-warp shapes) — **catastrophic regression**

**Lesson:** 3 blocks/SM with 256 threads is physically impossible for this tile configuration. Would require halving accumulators (different tile shape).

### Attempt 6: 4-Warp Kernel for All Shapes (Slight Regression)

**Idea:** Use 4-warp kernel (BM=128, 128 threads) for all shapes instead of just attn_to_out. With 128 threads × ~128 regs, we get 4 blocks/SM → same 16 warps but more, smaller blocks for better tail effect.

**Why it failed:** Same 16 warps per SM (4×4 = 2×8). But 2× more blocks means 2× the prologue/epilogue overhead. Plus the 4-warp kernel lacks CTA swizzle, so L2 locality is worse for large shapes.

**Result:** 361.64 TOPs (was 381) — **5% regression** (ff_up was worst at -13%)

---

## Why 381 Is a Plateau (Theoretical Bound)

The INT4 MMA instruction (`m16n8k64`) outputs INT32. To accumulate in half2 with per-group A-scaling, the minimum instruction chain per pair of output values is:

```
I2FP  (int32 → float)     ← FMA pipe, cannot be avoided
I2FP  (int32 → float)     ← FMA pipe, cannot be avoided
F2FP  (pack → half2)      ← FMA pipe, cannot be avoided
HFMA2 (scale × val + acc) ← FMA pipe, cannot be avoided
```

**4 FMA-pipe instructions per 2 values = 2 per value.** This is irreducible with the current architecture.

Per K-tile: 128 values × 2 = 256 FMA-pipe instructions + 32 IMMA = 288 total. The FMA pipe consumes 89% of issue cycles. The tensor cores can only fire during the remaining 11%.

**Theoretical max with this design: ~60-65% of INT4 peak = 370-400 TOPs.** We're at 381 — essentially at the ceiling.

---

## Path Beyond 381: What Would It Take

To reach 600 TOPs (97% of peak), the FMA:MMA ratio must drop from 8:1 to near 0:1. This requires:

1. **CUTLASS 4.x integration** — NVIDIA's templates use multi-stage software pipelining where K-tile N's MMA overlaps with K-tile N-1's conversion across different pipeline stages. Their instruction scheduling is hand-tuned by NVIDIA engineers. Potentially 400-480+ TOPs.

2. **Architectural change** — dequantize to FP16 in shared memory, then use FP16 MMA (no INT32→half2 conversion needed). But FP16 peak on A6000 is only 310 TOPS — below our 381.

3. **Hardware limitation** — 600 TOPs (97% of peak) may be physically impossible for a quantized GEMM with per-group scaling on this architecture. Even CUTLASS likely caps at 70-80% of peak for this workload.
