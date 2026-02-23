# `get_adc_values` Kernel Rework

## Overview

The original `get_adc_values` kernel in `fee.py` is a monolithic CUDA kernel that
implements the full self-trigger and ADC readout logic for the simulation. While functional, its structure makes it effectively
impossible to optimize or maintain. This document describes the analysis and
phased decomposition of the kernel into a set of smaller, well-scoped kernels that
each do one job and can be independently profiled, tuned, and reasoned about.

> NOTE: Currently I have the function [`save_kernel_arrays`](https://github.com/DUNE/larnd-sim/blob/ebrinckm/experiment/cli/simulate_pixels.py#L44) in place [before](https://github.com/DUNE/larnd-sim/blob/ebrinckm/experiment/cli/simulate_pixels.py#L1328) and [after](https://github.com/DUNE/larnd-sim/blob/ebrinckm/experiment/cli/simulate_pixels.py#L1371) the call to the monolith kernel `get_adc_values`. I recommend removing this before doing a full simulation.

### Rework Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Isolate kernel (freeze inputs/outputs for reproducibility) | Done |
| 1 | Identify hard kernel boundaries and concepts | Done |
| 2 | Extract RNG/noise generation into its own kernel | Done |
| 3a | Extract signal integration into its own kernel | Done |
| 3b/4 | Extract track accumulation into its own kernel | Done |
| 5 | Extract the while-loop (ADC window discovery + integration) | Done |

---

## Phase 1: Identifying Hard Kernel Boundaries

### The Problem

The original `get_adc_values` kernel (`fee.py`, line 538) is not a conventional
CUDA kernel. A well-structured CUDA kernel typically performs a single, uniform
computation across a grid of threads — the same operation applied to different data
(SIMT: Single Instruction, Multiple Threads). The original kernel violates this
principle in nearly every way possible. It is better described as an entire
**simulation pipeline jammed into a single kernel**, resulting in severe warp
divergence, excessive register pressure, duplicated logic, and a structure that
fundamentally resists optimization.

The subsections below break down the specific anti-patterns present in the original
kernel and why they matter from a GPU computing perspective.

### 1.1 Warp Divergence from a Data-Dependent While Loop

The core of the kernel is a `while` loop whose termination depends on runtime data:

```python
while ic < curre.shape[0] or adc_busy > 0:
    ...
```

In CUDA, threads are grouped into **warps** of 32 threads that execute in lockstep
(SIMT execution). When threads within a warp take different control-flow paths (i.e.
*diverge*), the warp must serialize execution: it runs one branch while masking off
the threads that took the other, then runs the other branch. Both paths are paid for
in wall-clock time.

Because the while loop's iteration count depends on the actual pixel signal (whether
and when thresholds are crossed, how many ADC triggers fire, etc.), different threads
within a warp will almost certainly iterate a different number of times. Threads that
finish early are forced to idle while the longest-running thread in their warp
completes. This is the textbook worst case for GPU utilization.

Inside this outer while loop, there is a **second** while loop for the ADC
integration window:

```python
while ic <= integrate_end:
    ...
```

This means the divergence is nested — even among the threads that are still active in
the outer loop, the inner loop introduces another layer of variable-length execution
within the same warp.

### 1.2 Inline RNG Generation Coupled to Control Flow

The kernel generates random numbers on-the-fly using Numba's `xoroshiro128p` RNG
functions. These calls are scattered throughout the kernel body, in both the outer
while loop and the inner integration loop:

- `xoroshiro128p_uniform_float32` — once, for the periodic reset phase
- `xoroshiro128p_normal_float32` — repeatedly, for reset noise, uncorrelated noise,
  and discriminator noise

Because these RNG calls are embedded inside data-dependent branches and loops, the
number of RNG draws per thread varies depending on the signal. This has two
consequences:

1. **Non-reproducibility under refactoring.** The RNG state is a single stream per
   thread. If the control flow changes (e.g. a thread takes a different branch), the
   sequence of random numbers consumed shifts, producing different noise values for
   all subsequent draws. This makes it very difficult to refactor the kernel while
   preserving exact numerical output.

2. **Inability to pre-compute or batch noise.** Because noise generation is woven
   into the control flow, it cannot be separated from the simulation logic, profiled
   independently, or replaced with a more efficient generation strategy (e.g.
   cuRAND device API, bulk generation on the host, etc.).

### 1.3 Duplicated Physics Logic

The signal integration logic — the convolution with the buffer risetime and the
per-track current fraction accumulation — appears **twice** in the kernel, nearly
identically:

- **First occurrence:** in the outer while loop (lines ~616–631 of `fee.py`), where
  charge `q` is accumulated tick-by-tick while scanning for a threshold crossing.
- **Second occurrence:** in the inner integration while loop (lines ~659–674),
  where charge is accumulated over the ADC hold window after a threshold crossing is
  detected.

This duplication means:
- Bug fixes or physics changes must be applied in two places.
- The two copies can (and do) subtly drift apart over time.
- The duplicated convolution loops are the most compute-intensive part of the kernel,
  and duplicating them roughly doubles the instruction footprint.

### 1.4 Excessive Register Pressure

The kernel is decorated with `@cuda.jit(max_registers=128)`, which is already an
explicit acknowledgment that register usage is a concern. High register usage per
thread directly limits **occupancy** — the number of warps that can be resident on a
streaming multiprocessor (SM) simultaneously. Lower occupancy means the GPU has fewer
warps to switch between when one is stalled on a memory access, reducing its ability
to hide latency.

The high register count is a direct consequence of the monolithic design: the kernel
must keep alive all state for RNG, signal integration, track bookkeeping, ADC
triggering, and reset logic simultaneously. Splitting into smaller kernels allows
each to use only the registers it needs, improving occupancy for the computationally
intensive phases.

### 1.5 Mixed Concerns — Four Jobs in One Kernel

Abstractly, the kernel performs four distinct tasks:

1. **Noise generation** — drawing random numbers for reset noise, uncorrelated noise,
   and discriminator noise.
2. **Signal integration** — convolving the raw induced currents with the buffer
   risetime response to produce integrated charge per tick.
3. **Track accumulation** — performing the same convolution on per-track signal
   contributions and accumulating current fractions for backtracking.
4. **ADC state machine** — scanning the integrated + noisy charge for threshold
   crossings, managing the ADC busy/hold/reset cycle, and recording ADC values and
   timestamps.

In the original kernel these four concerns are **interleaved instruction by
instruction**. The signal integration runs inside the ADC state machine loop. The
noise draws happen between integration steps. The track accumulation is nested inside
the integration. There are no clear boundaries between any of them.

This interleaving means:
- You cannot profile one concern in isolation to find bottlenecks.
- You cannot tune the thread/block configuration for one concern without affecting
  the others (e.g. signal integration is naturally 2D over pixels × ticks, but the
  ADC state machine is inherently 1D over pixels).
- You cannot test one concern against a known-good reference without running the
  entire pipeline.

### 1.6 Identified Kernel Boundaries

Based on the analysis above, four natural kernel boundaries emerge, corresponding to
the four concerns listed in Section 1.5:

| Concern | Natural parallelism | Phase |
|---------|---------------------|-------|
| Noise generation | 1D over pixels (with per-tick loop) | Phase 2 |
| Signal integration | 2D over pixels × ticks | Phase 3a |
| Track accumulation | 2D over pixels × ticks (with per-track inner loop) | Phase 3b/4 |
| ADC state machine (window discovery + integration) | 1D over pixels | Phase 5 |

Each of these can be implemented as a standalone CUDA kernel with its own grid
dimensions, register budget, and memory access pattern. The subsequent phases
(2 through 5) describe the extraction of each.

---

## Phase 2: Extracting RNG / Noise Generation

### Goal

Pull every `xoroshiro128p_*` call out of the monolithic kernel and into a dedicated
`generate_noise` kernel that pre-computes all noise values into device arrays before
the simulation logic runs.

### RNG Call Sites in the Original Kernel

The original kernel makes the following RNG draws, scattered across its control flow.
Each call advances the per-thread `xoroshiro128p` state by one step.

| # | Location | RNG function | Value produced |
|---|----------|-------------|----------------|
| 1 | Before while loop | `uniform_float32` | `periodic_reset_phase` — integer in `[0, PERIODIC_RESET_CYCLES]` |
| 2 | Before while loop | `normal_float32` | Initial `q_sum` reset noise (`RESET_NOISE_CHARGE * e`) |
| 3 | Outer while — periodic reset branch | `normal_float32` | Reset noise on periodic reset (`RESET_NOISE_CHARGE * e`) |
| 4 | Outer while — per tick | `normal_float32` | Uncorrelated noise (`UNCORRELATED_NOISE_CHARGE * e`) |
| 5 | Outer while — per tick | `normal_float32` | Discriminator noise (`DISCRIMINATOR_NOISE * e`) |
| 6 | Inner while — periodic reset branch | `normal_float32` | Reset noise on periodic reset (`RESET_NOISE_CHARGE * e`) |
| 7 | After inner while — ADC check | `normal_float32` | Uncorrelated noise for ADC sum |
| 8 | After inner while — ADC check | `normal_float32` | Discriminator noise for ADC comparison |
| 9 | After inner while — failed ADC | `normal_float32` | Reset noise (`RESET_NOISE_CHARGE * e`) |
| 10 | After inner while — passed ADC | `normal_float32` | Reset noise (`RESET_NOISE_CHARGE * e`) |

All of these draws produce one of three noise types scaled by a constant:

- **Reset noise:** `normal() * RESET_NOISE_CHARGE * e`
- **Uncorrelated noise:** `normal() * UNCORRELATED_NOISE_CHARGE * e`
- **Discriminator noise:** `normal() * DISCRIMINATOR_NOISE * e`

Plus one uniform draw for the periodic reset phase.

### Key Observation

None of these noise values depend on the simulation state (charge, threshold
crossings, etc.). They depend only on the pixel index and the tick index. The reason
they are *inside* the simulation loop in the original kernel is historical — they were
generated at the point of use. But because they are pure functions of the RNG state,
they can be pre-generated into lookup arrays without changing the physics.

### The `generate_noise` Kernel

The extracted kernel (`get_adc_values.py`, lines 57–107) allocates four device arrays
and fills them in a single 1D launch over pixels:

```python
@cuda.jit
def generate_noise(
    rng_states,
    noise_uncorr,       # (n_pixels, n_ticks)
    noise_disc,         # (n_pixels, n_ticks)
    noise_reset,        # (n_pixels, n_ticks)
    periodic_reset_phase  # (n_pixels,)
):
```

The draw order within a thread is:

1. One `uniform_float32` draw for `periodic_reset_phase[ip]`.
2. Three `normal_float32` draws for the tick-0 values of `noise_reset`, `noise_uncorr`,
   and `noise_disc`.
3. A loop over ticks 1 through `n_ticks - 1`, drawing three `normal_float32` values
   per tick (`noise_uncorr`, `noise_disc`, `noise_reset`).

### Output Arrays

| Array | Shape | Description |
|-------|-------|-------------|
| `noise_uncorr` | `(n_pixels, n_ticks)` | Pre-scaled uncorrelated noise per pixel per tick |
| `noise_disc` | `(n_pixels, n_ticks)` | Pre-scaled discriminator noise per pixel per tick |
| `noise_reset` | `(n_pixels, n_ticks)` | Pre-scaled reset noise per pixel per tick |
| `periodic_reset_phase` | `(n_pixels,)` | Integer phase offset for periodic resets |

These arrays are allocated on the device and passed as inputs to the downstream
kernels (Phases 3–5), replacing every inline RNG call.

### Trade-offs

**Memory.** The original kernel consumed zero extra memory for noise — values were
generated and used in registers. The extracted version materializes three full
`(n_pixels, n_ticks)` arrays plus one `(n_pixels,)` array on the device. For typical
problem sizes this is modest (a few MB), and the memory is freed after the downstream
kernels complete.

**Draw order change.** In the original kernel, the RNG draws for a given thread are
interleaved with simulation logic and conditional branches. The pre-generation kernel
draws them in a fixed, deterministic order (reset → uncorrelated → discriminator,
per tick). This means the *specific* random values assigned to each tick will differ
from the original kernel even with the same initial RNG state. The statistical
properties are identical (same distributions, same per-pixel independence), but
exact numerical reproducibility against the original kernel is broken at this phase.
This is an expected and accepted consequence of decoupling RNG from control flow.

**Benefit.** Downstream kernels no longer carry `rng_states` or any RNG logic. This
removes a source of register pressure, eliminates a coupling between control flow and
random number sequencing, and allows the noise generation to be profiled, replaced,
or unit-tested in complete isolation.

---

## Phase 3: Extracting Signal Integration (3a) and Track Accumulation (3b)

### Goal

Pull the signal integration convolution and per-track current fraction accumulation
out of the ADC state machine loop and into standalone kernels. Originally planned as
two separate phases (3 and 4), track accumulation turned out to be functionally
inseparable from signal integration — it is the same convolution applied to the
per-track signal contributions — so the two were combined into Phase 3a and Phase 3b
respectively.

### What the Original Kernel Does

In the original kernel, the signal integration is performed **inline, per tick,
inside the while loop**. At each iteration of the ADC state machine, the kernel
recomputes the integrated charge for the current tick by convolving the raw induced
current with an exponential buffer risetime response:

```python
if detector.BUFFER_RISETIME > 0:
    conv_start = max(last_reset, floor(ic - 10 * detector.BUFFER_RISETIME / detector.TIME_SAMPLING))
    for jc in range(conv_start, min(ic + 1, curre.shape[0])):
        w = exp((jc - ic) * detector.TIME_SAMPLING / detector.BUFFER_RISETIME) \
            * (1 - exp(-detector.TIME_SAMPLING / detector.BUFFER_RISETIME))
        q += curre[jc] * detector.TIME_SAMPLING * w

        for itrk in range(ntrks):
            idx = total_backtracks * jc + offset_backtrack[ip] + itrk
            current_fractions[ip][iadc][itrk] += pixels_signals_tracks[idx] \
                * detector.TIME_SAMPLING * w
```

This block appears **twice** — once in the outer while loop (scanning for threshold
crossings) and again in the inner while loop (integrating the ADC hold window). The
track accumulation inner loop is nested inside both copies.

There are two problems with computing integration this way:

1. **Redundant recomputation.** The integrated charge at tick `ic` depends only on the
   raw signal up to that tick and the buffer risetime constant. It does not depend on
   the ADC state machine's state (`q_sum`, `adc_busy`, thresholds, etc.). Yet the
   original kernel recomputes it from scratch on every tick, inside a loop that is
   already bottlenecked on warp divergence.

2. **Wrong parallelism axis.** The convolution at each tick is independent of the
   convolution at every other tick (for a given pixel). This is a textbook 2D
   parallel workload over `(pixels, ticks)`, but in the original kernel it is forced
   into the 1D-over-pixels ADC state machine loop, serializing all tick-level work
   per thread.

### Phase 3a: `integrate_signal`

The extracted kernel (`get_adc_values.py`, lines 128–163) computes the integrated
charge for every `(pixel, tick)` pair in a single 2D launch:

```python
@cuda.jit
def integrate_signal(
    pixels_signals,   # (n_pixels, n_ticks) — raw induced currents
    signal_charge     # (n_pixels, n_ticks) — output: integrated charge
):
    ip, ic = cuda.grid(2)
    ...
```

Each thread handles one `(pixel, tick)` pair. For a given tick `ic`, it convolves the
raw current backwards over a window of `10 * BUFFER_RISETIME / TIME_SAMPLING` ticks
using the same exponential weight as the original:

```python
w = exp((jc - ic) * TIME_SAMPLING / BUFFER_RISETIME) \
    * (1.0 - exp(-TIME_SAMPLING / BUFFER_RISETIME))
```

When `BUFFER_RISETIME` is zero (no shaping), it falls through to the simple case:
`q = curre[ic] * TIME_SAMPLING`.

The result is written to a materialized `signal_charge` array of shape
`(n_pixels, n_ticks)` that downstream phases can read by index, replacing the
on-the-fly recomputation in the original kernel.

**Grid configuration.** Launched with 2D thread blocks `(16, 16)` over a 2D grid
covering `(n_pixels, n_ticks)`, which matches the natural parallelism of the
workload.

### Phase 3b: `integrate_signal_tracks`

The track accumulation kernel (`get_adc_values.py`, lines 186–241) is structurally
identical to `integrate_signal` but operates on the per-track signal array
`pixels_signals_tracks`. This array is stored in a flat/jagged layout indexed by:

```python
idx = total_backtracks * jc + offset_backtrack[ip] + itrk
```

where `total_backtracks = offset_backtrack[-1] + num_backtrack[-1]` serves as the
stride across ticks, `offset_backtrack[ip]` is the pixel's offset into the jagged
track dimension, and `itrk` selects the track.

```python
@cuda.jit
def integrate_signal_tracks(
    pixels_signals_tracks,  # flat jagged array
    num_backtrack,          # (n_pixels,) — track count per pixel
    offset_backtrack,       # (n_pixels,) — offset per pixel
    signal_charge_track     # (n_pixels, n_ticks, MAX_TRACKS_PER_PIXEL) — output
):
    ip, ic = cuda.grid(2)
    ...
```

Each thread handles one `(pixel, tick)` pair and loops over up to
`min(num_backtrack[ip], MAX_TRACKS_PER_PIXEL)` tracks, performing the same
convolution as `integrate_signal` but reading from the per-track signal data. The
output is a 3D array `signal_charge_track` of shape
`(n_pixels, n_ticks, MAX_TRACKS_PER_PIXEL)`.

**Why a separate kernel instead of folding tracks into `integrate_signal`?** The
per-track version has a fundamentally different memory access pattern (jagged indexing
into a flat buffer) and an additional inner loop over tracks. Keeping it separate
means `integrate_signal` stays simple and fast for the common path, while the
track kernel can be profiled and tuned (or skipped entirely) independently.

### Output Arrays

| Array | Shape | Produced by | Description |
|-------|-------|-------------|-------------|
| `signal_charge` | `(n_pixels, n_ticks)` | `integrate_signal` | Integrated charge per pixel per tick |
| `signal_charge_track` | `(n_pixels, n_ticks, MAX_TRACKS_PER_PIXEL)` | `integrate_signal_tracks` | Integrated charge per pixel per tick per track |

### What Changed from the Original

**Convolution window lower bound.** In the original kernel, `conv_start` is bounded
by `last_reset` (the tick at which the most recent ADC reset occurred), which
prevents the convolution from looking back past the last reset. In the extracted
kernel, `conv_start` is bounded only by `0` (start of the array). This is because
the integration kernel has no knowledge of the ADC state machine — resets have not
happened yet at this stage. The downstream ADC kernel (Phase 5) is responsible for
using the pre-computed `signal_charge` values within the correct reset windows.

**Duplication eliminated.** The original kernel's two copies of the integration +
track accumulation logic are replaced by a single execution of each kernel. The
results are materialized once and reused by the ADC state machine.

**Parallelism unlocked.** The workload moves from 1D (one thread per pixel, serial
over ticks) to 2D (one thread per pixel-tick pair), exposing the full tick dimension
to the GPU scheduler.

---

## Phase 5: Extracting the ADC State Machine

### Goal

With noise pre-generated (Phase 2) and signal integration materialized (Phase 3a/3b),
the remaining work is the ADC self-trigger state machine itself — the while loop that
scans integrated charge for threshold crossings, manages the busy/hold/reset cycle,
and records ADC values. This phase extracts that logic and splits it into two kernels:

- **`discover_adc_windows`** (Phase 5a) — walks the tick axis per pixel, finds
  threshold crossings, accumulates charge through the ADC hold window, performs the
  second threshold check, and records the window boundaries, accumulated charge, and
  reset state for each accepted trigger.
- **`integrate_windows`** (Phase 5b) — given the discovered windows, accumulates
  per-track current fractions over each window, normalizes them, and writes the final
  ADC values, timestamps, and current fractions.

### Why Two Kernels Instead of One

In the original kernel, window discovery and window integration are fused: the inner
while loop both advances `ic` through the hold interval *and* accumulates charge and
track fractions tick-by-tick as it goes. Fusing them was necessary in the original
design because integrated charge was computed on the fly — there was no materialized
`signal_charge` array to sum over after the fact.

With Phase 3's materialized arrays, the two concerns partially decouple:

1. **Discovery** is inherently serial per pixel (each threshold crossing depends on
   the cumulative charge from all prior ticks), so it must remain a 1D kernel over
   pixels with a while loop. It also retains an inner while loop to walk through the
   ADC hold window and accumulate `q_sum` for the second threshold check — but this
   inner loop only performs cheap array reads, not convolution or track accumulation.
2. **Track fraction accumulation** over each window is parallel once the window
   bounds are known — each `(pixel, adc_index)` pair can be handled by an
   independent thread that sums `signal_charge_track[ip, lr:end+1, itrk]` and
   normalizes.

Splitting them means the serial part (discovery) does only charge accumulation and
threshold logic — array lookups and arithmetic, no convolution or track indexing —
while the parallel part (track fractions, timestamps) gets its own 2D launch with no
warp divergence.

### Phase 5a: `discover_adc_windows`

The kernel (`get_adc_values.py`, lines 254–374) retains the while loop structure of
the original because the ADC state machine is fundamentally sequential per pixel:
whether a threshold crossing occurs at tick `ic` depends on the accumulated charge
from all preceding ticks.

```python
@cuda.jit(max_registers=96)
def discover_adc_windows(
    signal_charge,          # (n_pixels, n_ticks) — from Phase 3a
    noise_uncorr,           # (n_pixels, n_ticks) — from Phase 2
    noise_disc,             # (n_pixels, n_ticks) — from Phase 2
    noise_reset,            # (n_pixels, n_ticks) — from Phase 2
    periodic_reset_phase,   # (n_pixels,) — from Phase 2
    pixel_thresholds,       # (n_pixels,)
    adc_start,              # (n_pixels, MAX_ADC_VALUES) — output
    adc_end,                # (n_pixels, MAX_ADC_VALUES) — output
    adc_ticks_idx,          # (n_pixels, MAX_ADC_VALUES) — output
    adc_counts,             # (n_pixels,) — output
    adc_q_sum,              # (n_pixels, MAX_ADC_VALUES) — output
    adc_last_reset          # (n_pixels, MAX_ADC_VALUES) — output
):
```

The per-pixel state machine logic is:

1. Initialize `q_sum` from `noise_reset[ip, 0]` (the pre-generated initial reset
   noise).
2. Walk ticks: read `signal_charge[ip, ic]` — a single array lookup, replacing the
   entire convolution loop from the original kernel.
3. On a periodic reset tick, reset `q_sum` from `noise_reset[ip, ic]` and continue.
4. Accumulate charge: `q_sum += q`.
5. Read `noise_uncorr[ip, ic]` and `noise_disc[ip, ic]` — array lookups replacing
   inline RNG draws.
6. If `q_sum + q_noise >= threshold + disc_noise` and the ADC is not busy, enter an
   inner while loop that walks `ic` through the ADC hold window
   (`integrate_end = ic + interval`), continuing to accumulate `q_sum` from
   `signal_charge` lookups (handling periodic resets within the window).
7. After the inner loop, perform a second threshold check on the accumulated charge.
   If the charge has fallen below threshold (a false trigger), reject the window:
   advance `ic` past the reset cycle, reset `q_sum`, and continue scanning. If the
   charge remains above threshold, accept the window: record `(start, end)`,
   `adc_ticks_idx`, `adc_q_sum`, and `adc_last_reset`.
8. After the outer loop exits, write the total number of accepted triggers to
   `adc_counts[ip]`.

The kernel accumulates total charge (`q_sum`) but does **not** perform signal
convolution or track fraction accumulation — those are handled by the upstream
Phase 3 kernels and the downstream `integrate_windows` kernel, respectively. The
per-tick work inside the while loop remains lightweight: one read from
`signal_charge`, two reads from the noise arrays, an addition, and a comparison.
The inner while loop adds charge reads over the hold window, but each read is a
single array lookup into the pre-computed `signal_charge`, not a recomputation of
the convolution.

**Register budget.** Decorated with `max_registers=96`, down from the original's 128.
This is possible because the kernel no longer carries signal integration state,
convolution loop variables, track indexing, or RNG state.

**Grid configuration.** 1D launch, 128 threads per block over `n_pixels`. This is the
natural (and only possible) parallelism for a serial state machine.

#### Output Arrays

| Array | Shape | Description |
|-------|-------|-------------|
| `adc_start` | `(n_pixels, MAX_ADC_VALUES)` | First tick of each ADC integration window |
| `adc_end` | `(n_pixels, MAX_ADC_VALUES)` | Last tick of each ADC integration window |
| `adc_ticks_idx` | `(n_pixels, MAX_ADC_VALUES)` | Post-integration tick index (used for timestamp computation) |
| `adc_counts` | `(n_pixels,)` | Number of ADC triggers per pixel |
| `adc_q_sum` | `(n_pixels, MAX_ADC_VALUES)` | Accumulated charge including uncorrelated noise at end of integration window (`q_sum + noise_uncorr`) |
| `adc_last_reset` | `(n_pixels, MAX_ADC_VALUES)` | Tick index of the most recent ADC reset at the time each window was accepted |

### Phase 5b: `integrate_windows`

The kernel (`get_adc_values.py`, lines 425–495) takes the window bounds from
Phase 5a and the materialized arrays from Phases 3a/3b, and produces the final
simulation outputs: ADC charge values, timestamps, and per-track current fractions.

```python
@cuda.jit
def integrate_windows(
    signal_charge,        # (n_pixels, n_ticks) — from Phase 3a
    signal_charge_track,  # (n_pixels, n_ticks, MAX_TRACKS_PER_PIXEL) — from Phase 3b
    num_backtrack,        # (n_pixels,) — track count per pixel
    adc_end,              # (n_pixels, MAX_ADC_VALUES) — from Phase 5a
    adc_counts,           # (n_pixels,) — from Phase 5a
    adc_ticks_idx,        # (n_pixels, MAX_ADC_VALUES) — from Phase 5a
    adc_q_sum,            # (n_pixels, MAX_ADC_VALUES) — from Phase 5a
    adc_last_reset,       # (n_pixels, MAX_ADC_VALUES) — from Phase 5a
    adc_list,             # (n_pixels, MAX_ADC_VALUES) — output
    adc_ticks_list,       # (n_pixels, MAX_ADC_VALUES) — output
    current_fractions,    # (n_pixels, MAX_ADC_VALUES, MAX_TRACKS_PER_PIXEL) — output
    time_ticks,           # (n_ticks,)
    time_padding,         # scalar
    periodic_reset_phase  # (n_pixels,) — from Phase 2
):
    ip, iadc = cuda.grid(2)
```

Each thread handles one `(pixel, adc_index)` pair. Early-exit if
`iadc >= adc_counts[ip]` (this pixel had fewer triggers). The per-thread work is:

1. Iterate from `adc_last_reset[ip, iadc]` through `adc_end[ip, iadc]`, accumulating
   `signal_charge[ip, ic]` into `true_q` and `signal_charge_track[ip, ic, itrk]` into
   per-track accumulators for up to `min(num_backtrack[ip], MAX_TRACKS_PER_PIXEL)`
   tracks. If a periodic reset tick is encountered, zero the accumulators and restart.
2. Normalize current fractions: if `true_q > 0`, divide each track's accumulated
   charge by the total to get fractional contributions.
3. Write `adc_list[ip, iadc] = adc_q_sum[ip, iadc]` (the ADC value was already
   computed by `discover_adc_windows`, including uncorrelated noise).
4. Compute the timestamp from `adc_ticks_idx[ip, iadc]`: look up
   `time_ticks[tick] + time_padding`, with overflow handling for ticks past the end
   of the time array.

**Grid configuration.** 2D launch with thread blocks `(16, 8)` over
`(n_pixels, MAX_ADC_VALUES)`. The second dimension is bounded by `MAX_ADC_VALUES`
(not `n_ticks`), so blocks are small and threads that exceed the pixel's actual
`adc_counts` exit immediately.

**No warp divergence.** Every active thread executes the same loop
(`for ic in range(lr, end + 1)`) with an interval length that is constant across
all triggers (determined by `CLOCK_CYCLE` and `ADC_HOLD_DELAY`). The only divergence
is the early exit for pixels with fewer triggers than the thread's `iadc`, which
affects at most a handful of threads per warp.

### What Changed from the Original

**The inner while loop is lighter, not gone.** The original kernel's inner
`while ic <= integrate_end` loop did heavy work on every tick: recomputing the full
signal convolution (the `for jc in range(conv_start, ...)` loop), accumulating
per-track current fractions (a nested loop over `ntrks`), and drawing RNG values for
noise. In the rework, `discover_adc_windows` retains an inner while loop over the
ADC hold window — this turned out to be necessary for correctness (Issues 1–3, 7) —
but the body is reduced to a single `signal_charge[ip, ic]` array read and an
addition to `q_sum`. There is no convolution recomputation, no track indexing, and no
RNG calls inside the inner loop.

In the original monolith, the convolution loop was duplicated: it appeared identically
in both the outer while loop (scanning for threshold crossings) and the inner while
loop (integrating the hold window). This duplication is eliminated in the rework.
The convolution is computed once by `integrate_signal` / `integrate_signal_tracks`
(Phase 3) and materialized into arrays. Both the outer loop and inner loop of
`discover_adc_windows` read from the same pre-computed `signal_charge` array, so the
convolution work is performed exactly once regardless of how many threshold crossings
occur or how many hold windows are walked.

**Separation of serial and parallel work.** The original kernel forced all work into a
single 1D serial-per-pixel launch. The rework isolates the irreducibly serial part
(threshold scanning and charge accumulation) into `discover_adc_windows` and moves
the parallelizable part (track fraction accumulation and normalization) into a
separate 2D kernel (`integrate_windows`) with clean, uniform control flow.

**Simplified state machine body.** The per-tick work inside the while loops of
`discover_adc_windows` is reduced to array lookups and arithmetic — no convolution
inner loop, no track indexing, no RNG calls. This makes the state machine easier to
reason about, debug, and profile.

**Full replacement of `get_adc_values`.** With Phase 5 complete, the original
monolithic `get_adc_values` kernel is entirely replaced. What was a single kernel
launch is now a sequence of five kernel launches:

1. `generate_noise` — pre-compute all noise arrays (Phase 2)
2. `integrate_signal` — pre-compute integrated charge per pixel per tick (Phase 3a)
3. `integrate_signal_tracks` — pre-compute integrated charge per track (Phase 3b)
4. `discover_adc_windows` — run the ADC state machine to find trigger windows (Phase 5a)
5. `integrate_windows` — sum charge and track fractions over each window (Phase 5b)

Each kernel reads the outputs of the previous stages and writes its own outputs,
forming a straightforward data-flow pipeline. The original `get_adc_values` kernel
can be removed and replaced by this sequence of calls with no change to the
surrounding simulation code — the final outputs (`adc_list`, `adc_ticks_list`,
`current_fractions`) are identical in shape and meaning.

---

## Known Issue: Output Divergence from Frozen Reference

The reworked pipeline initially produced outputs that differed significantly from the
frozen reference. A detailed comparison identified eight structural flow issues
during the kernel decomposition. Six have been fixed, and two have been accepted as
inherent trade-offs of the decomposed architecture:

| # | Issue | Status |
|---|-------|--------|
| 1 | Missing second threshold check after integration | **Fixed** — `discover_adc_windows` now runs an inner while loop and performs the post-integration threshold comparison, rejecting false triggers. |
| 2 | `adc_list` stored only window charge, not total since reset | **Fixed** — `discover_adc_windows` accumulates the full `q_sum` through the integration window and outputs it (with uncorrelated noise) as `adc_q_sum`. |
| 3 | Tick advancement off by `interval + 1` after trigger | **Fixed** — the inner while loop in `discover_adc_windows` advances `ic` through the integration window before the reset advancement, matching the original's `ic = integrate_end + 1 + RESET_CYCLES_advance`. |
| 4 | Convolution lower bound uses `0` instead of `last_reset` | **Accepted** — `last_reset` is computed by the ADC state machine (Phase 5), which runs after `integrate_signal` (Phase 3). This circular dependency cannot be resolved without reintroducing the convolution into the state machine kernel. The exponential weighting heavily attenuates the distant contributions that the original would have excluded. |
| 5 | Timestamp used crossing tick instead of post-integration tick | **Fixed** — `adc_ticks_idx` now stores the post-integration `ic`, and `integrate_windows` replicates the original's `post_adc_ticks` overflow handling. |
| 6 | Periodic reset not handled during window integration | **Fixed** — `integrate_windows` now iterates from `last_reset` to `end`, zeroing the track fraction accumulator when a periodic reset tick is encountered. |
| 7 | Missing uncorrelated noise in ADC value | **Fixed** — absorbed into Issue 2; `adc_q_sum` includes the noise term. |
| 8 | Vastly different `current_fractions` values | **Accepted** — The old kernel leaves un-normalized raw charge in `current_fractions[ip][iadc]` for the trailing (never-accepted) ADC slot of every pixel, polluting any aggregate comparison. The reworked flow only writes to accepted ADC slots (gated by `adc_counts`), producing properly normalized fractions that sum to ~1.0 per window. |

### Remaining Expected Differences

After these fixes, the remaining output divergence should be attributable to:

1. **RNG draw-order change (Phase 2).** Pre-generating noise in a fixed order
   produces different random values than the original's control-flow-dependent draws.
   This is an expected statistical difference.

2. **Convolution lower bound (Issue 4).** The pre-computed `signal_charge` values
   use `conv_start = max(0, ...)` instead of `max(last_reset, ...)`. For ticks
   shortly after a reset, the signal may be slightly higher than the original, which
   can shift threshold crossing times and cascade through subsequent triggers.

Both of these are acknowledged trade-offs of the decomposed architecture.
