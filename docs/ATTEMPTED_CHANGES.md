# Attempted Changes Log

## Session: 2026-04-20 (NCU-Driven Profiling)

### Setup
- Server: `ssh -i ~/Downloads/ssh-gpu ubuntu@instance-2044640971865460736.yottadeos.com`
- Project on server: `/home/ubuntu/mikeb/`
- NCU installed: v2026.1.1.0 (via conda nsight-compute package)
- Claude Code installed: v2.1.116 (not yet authenticated)
- Baseline confirmed: **327.46 TOPs** (correctness passing all 4 layers)

---

### NCU Profile Results (Baseline, 327 TOPs)

**Kernel 0: gemm_direct_kernel (attn_to_qkv, M=4096 N=9216 K=3072, grid 72×16)**

| Category | Metric | Value |
|----------|--------|-------|
| **Speed of Light** | Compute (SM) Throughput | 52.85% |
| | Memory Throughput | 46.59% |
| **Occupancy** | Theoretical | 33.33% |
| | Achieved | 32.14% |
| | Block Limit (Registers) | 2 blocks/SM |
| | Registers/Thread | 125 |
| **Scheduler** | Issue Slots Busy | 42.56% |
| | No Eligible Warps | 56.23% of cycles |
| | Active Warps/Scheduler | 3.87 |
| | Eligible Warps/Scheduler | 0.78 |
| **Stalls** | Top stall: math pipe throttle | 30.1% of CPI |
| | Warp Cycles/Issued Instruction | 8.84 |
| **Memory** | L2 Hit Rate | 89.60% |
| | L1/TEX Hit Rate | 74.15% |
| | DRAM Throughput | 31.64% |
| | Register Spilling | 0 bytes |
| **Coalescing** | Global loads: bytes utilized/sector | 26.9/32 (16% waste) |
| | Global stores: bytes utilized/sector | 16.0/32 (50% waste) |
| | Excessive sectors | 18% of total |

**NCU Recommendations:**
1. "This workload exhibits low compute throughput and memory bandwidth utilization relative to peak — typically indicates **latency issues**"
2. "Each scheduler only issues an instruction every 2.3 cycles"
3. "Each warp spends 2.7 cycles stalled waiting for the **execution pipe to be available**"
4. Math pipe throttle = 30.1% of total 8.84 stall cycles

---

### SASS Instruction Mix (gemm_direct_kernel, 8-warp)

Per K-tile (fully unrolled, NT=8):

| Instruction | Count | Pipe | Purpose |
|-------------|-------|------|---------|
| IMMA | 32 | Tensor | m16n8k64 INT4 MMA |
| I2FP | 128 | FP/Conv | int32 → float32 conversion |
| F2FP | 64 | FP/Conv | float pack → half2 |
| HMUL2 | 64 | FMA | scale products (sa × sb) |
| HFMA2 | 64 | FMA | accumulation (val × scale + acc) |
| STG | 64 | LSU | epilogue stores (not per K-tile) |
| IMAD | 76 | INT | address arithmetic |
| LDG | 12 | LSU | global loads (B frags + scales) |
| LDS | 16 | LSU | shared memory reads (B scales) |
| Other | ~30 | Various | SHF, ISETP, LEA, NOP, etc. |
| **Total** | **~550** | | |

**Key ratio: 10 FMA-pipe instructions per 1 MMA (320 FMA vs 32 IMMA)**

This means the FMA/conversion pipe is the actual bottleneck, NOT the tensor core pipe. The tensor cores are starved because the scheduler is busy feeding the FMA pipe.

---

### Attempt 1: A-Scale Coalescing

**Hypothesis:** Strided A-scale reads (`scales_A[(m_base + lane) * nkt + kt]`, stride=nkt×2 bytes between threads) waste 16% of load bandwidth. Repacking scales to [M/32, nkt, 32] layout makes reads coalesced.

**Changes:**
- Added `s_sa_cache` for repacked scales
- Reshaped scales_A from [M, nkt] → [M/32, nkt, 32] via `reshape().transpose().contiguous()`
- Changed kernel read from strided to `scales_A[((bm * NW + warp) * nkt + kt) * WM + lane]`

**Result:** 327.10 TOPs (was 327.46) — **no improvement, within noise**

**Why it failed:** The A-scale loads are only 32 × 2 bytes = 64 bytes per K-tile per warp, vs ~5120 bytes for A+B fragment loads. Even with 16% waste on scale loads, the absolute wasted bandwidth is tiny (~960 bytes/K-tile) compared to total traffic. The optimization saved ~15.6% of memory traffic that was already only 5% of total.

**Status:** Reverted from kernel (scale read back to original pattern). The repacking cache code is still present but unused since the kernel reads the original layout.

---

### Attempt 2: Reduced Unroll Factor (#pragma unroll 4)

**Hypothesis:** Full unroll (NT=8) creates huge code that uses all 125 registers. Partial unroll might let the compiler allocate fewer registers → more blocks/SM → more scheduling diversity.

**Changes:** `#pragma unroll` → `#pragma unroll 4` for the N-tile loop

**Result:** 96.49 TOPs — **catastrophic regression (3.4x slower)**

**Why it failed:** Partial unroll forces a residual loop (2 iterations) with dynamic indexing into `acc[AT][NT][4]` arrays. The compiler can't keep these in registers with non-constant indices → spills accumulators to local memory. Each accumulator access becomes a local load/store (~100 cycles) instead of a register read (~0 cycles).

**Lesson:** Full unroll is MANDATORY for register-resident accumulator arrays. Any loop with register arrays must be fully unrolled or the compiler will spill.

**Status:** Reverted.

---

### Root Cause Analysis

The fundamental performance limiter at 52% of peak is **FMA pipe saturation**:

1. Each m16n8k64 MMA produces 4 int32 values
2. Converting to scaled half2 requires: 4× I2FP + 2× F2FP + 2× HMUL2 + 2× HFMA2 = 10 FMA-pipe instructions per 4 outputs
3. With 32 MMAs per K-tile, that's 320 FMA-pipe instructions vs 32 tensor-pipe instructions
4. All 4 warps per sub-partition compete for the single FMA pipe
5. Result: warps stall waiting for FMA pipe (reported as "math pipe throttle")

**To significantly improve, we must reduce the FMA:MMA ratio.** Options:
- Defer B-scale multiply to epilogue (removes 64 HMUL2/K-tile = 20% FMA reduction)
- Use per-channel weight quantization (enables the above)
- Use integer accumulation (eliminate all FP conversions in inner loop — risky for accuracy)
- Use CUTLASS (their instruction scheduling is likely much better)
- Structural: get more occupancy (requires reducing to ~85 regs — very hard)

---

### Attempt 3: Per-Channel Weight Quantization + Deferred B-Scale Epilogue ✅

**Hypothesis:** The 64 HMUL2 per K-tile (sa×sb products inside the N-loop) are a major FMA-pipe bottleneck. By using per-channel weight quantization (weight_group_size = K → 1 B-group), the B-scale becomes constant across all K-tiles and can be moved from the inner loop to the epilogue. This removes 64 HMUL2/K-tile from the inner loop, reducing FMA-pipe demand by ~20%.

**Changes:**
- `quantize.py`: Set `weight_group_size = K` for all layers (per-channel with optimal clipping)
- `kernel.cu` (both 8-warp and 4-warp kernels):
  - Removed B-group outer loop (only 1 group now)
  - Removed `__syncthreads()` for B-scale buffer swap
  - Inner loop: `acc += convert(mma) * sa` (no sb multiply)
  - Epilogue: `C[i,j] = acc[i,j] * sb[j]` (apply B-scale once at output)
  - Removed double-buffered B-scale (only need single load before K-loop)

**Result:** 380.82 TOPs (was 327.46) — **+16.3% improvement!**

| Layer | Before | After | Change |
|-------|--------|-------|--------|
| attn_to_qkv | 327.77 | 383.24 | +17% |
| attn_to_out | 310.69 | 364.72 | +17% |
| ff_up | 327.54 | 387.66 | +18% |
| ff_down | 342.39 | 387.66 | +13% |

**Correctness:** All pass (cosine: qkv=0.990, out=0.992, up=0.981, down=0.979)

**Why it worked:**
- Removed 64 HMUL2 per K-tile from the hot inner loop
- For attn_to_qkv (48 K-tiles): saved 48×64 = 3072 FMA-pipe instructions per warp per kernel call
- Per-channel weight quantization with optimal clipping still passes all cosine thresholds comfortably
- Simpler loop structure (no B-group outer loop, no __syncthreads) → better compiler scheduling
- Potential register savings from fewer live values in inner loop

**Status:** KEPT. Tagged as v4.0 on server.

---

### Attempt 4: __launch_bounds__(256, 3) on 8-warp kernel

**Hypothesis:** Requesting 3 blocks/SM (≤85 regs) would give 24 warps instead of 16, improving scheduler diversity by 50%.

**Changes:** Added `__launch_bounds__(256, 3)` to `gemm_direct_kernel`

**Result:** 135.92 TOPs — **catastrophic regression (64-67 TOPs for 8-warp shapes)**

**Why it failed:** The compiler had to reduce from 128 to ≤85 registers, spilling ~40 registers to local memory. Each spilled register access adds ~100 cycles of latency. Correctness PASSED (unlike old experiment 8 on v3 kernel), but performance is destroyed.

**Lesson:** 3 blocks/SM with 256 threads (8 warps) is impossible without fundamental tile restructuring. Need ≤85 regs which requires halving the accumulator (BN=64 or AT=1).

**Status:** Reverted.

---

### Attempt 5: Single-buffer ssb cleanup

**Hypothesis:** Removing the unused second ssb buffer (from old double-buffered B-group logic) frees 256 bytes of shared memory.

**Changes:** `__shared__ half ssb[2][BN]` → `__shared__ half ssb[BN]`, updated all references.

**Result:** 378.66 TOPs (was 380.82) — **neutral (within noise)**

**Status:** Kept (cleaner code, no performance impact).

---

### Attempt 6: 4-Warp Kernel for All Shapes

**Hypothesis:** Using the 4-warp kernel (BM=128, 128 threads) for all shapes instead of just attn_to_out. With 128 threads × ~128 regs, we get 4 blocks/SM instead of 2. Same 16 warps but 4 smaller blocks → better tail effect and more grid flexibility.

**Changes:** Changed dispatch: `use_4w = aligned && (M % BM4 == 0)` (all shapes use 4-warp)

**Result:** 361.64 TOPs (was 381) — **5% regression**

| Layer | 8-warp | 4-warp | Change |
|-------|--------|--------|--------|
| attn_to_qkv | 383 | 366 | -4.4% |
| ff_up | 388 | 337 | **-13%** |
| ff_down | 388 | 384 | -1% |
| attn_to_out | 365 | 360 | -1.4% |

**Why it failed:** Same 16 warps/SM (4 blocks × 4 warps = 2 blocks × 8 warps). But 2× more blocks means 2× the prologue/epilogue overhead. Plus the 4-warp kernel lacks CTA swizzle → worse L2 locality for large shapes (ff_up had worst regression at -13%).

**Status:** Reverted.

---

### Current Best: v4.0 — 381 TOPs (61.5% of peak)

### Theoretical Analysis: Why 381 Is a Plateau

The conversion chain per MMA output value is irreducible:
- `I2FP` (int32→float32) + `HFMA2` (scale × val + acc) = 2 FMA-pipe instructions minimum per value
- 32 MMAs per K-tile × 4 outputs each = 128 values → 256 FMA-pipe instructions per K-tile
- 32 IMMA + 256 FMA = 288 total → 89% of issue cycles go to FMA pipe, 11% to tensor pipe
- This limits us to ~60-65% of tensor core peak with the current kernel architecture

**To go significantly beyond 381, we need one of:**
1. **CUTLASS 4.2 integration** — NVIDIA's templates use deeply pipelined multi-stage mainloops that overlap conversion with MMA across warps, potentially achieving 70-80% of peak
2. **Entirely different algorithm** — e.g., dequantize to FP16 in shared memory, then use FP16 MMA (but FP16 peak is only 310 TOPS, below our 381)
3. **Integer accumulation** — only works if multiple K-tiles share the same A-scale (not possible with group_size=64=BK)

---

### Next Steps to Try

1. **CUTLASS 4.2 integration** (HIGH EFFORT, HIGH POTENTIAL)
   - Headers at: `.../cutlass_library/source/include/cutlass/`
   - Need custom mainloop for per-K-tile A-scale + epilogue B-scale
   - CUTLASS's Ampere templates have multi-stage software pipelining
   - Could potentially reach 400-480+ TOPs

2. **Per-channel weight quantization + deferred B-scale epilogue** ✅ DONE (Attempt 3)
   - Set weight_group_size = K in quantize.py
   - Remove 64 HMUL2 from inner loop, add 64 HMUL2 to epilogue (one-time)
   - Must verify correctness thresholds still pass
   - Expected: ~20% FMA reduction → maybe 5-10% throughput gain

2. **CUTLASS 4.2 integration**
   - Headers available at: `.../cutlass_library/source/include/cutlass/`
   - Need custom mainloop for per-group scaling
   - High effort, potentially large payoff

3. **Integer accumulation with deferred scaling**
   - Accumulate raw int32 MMA outputs across multiple K-tiles
   - Only works if multiple K-tiles share the same A-scale (they don't with group_size=64=BK)
   - Would need BK < group_size, but m16n8k64 mandates K=64 per MMA
   - Likely not feasible without changing quantization format

4. **Different tile shape for higher occupancy**
   - AT=1 (WM=16) saves ~32 acc registers
   - Combined with other savings, might reach ~90 regs → 3 blocks/SM
   - But halves per-warp compute (failed before as experiment 6)
