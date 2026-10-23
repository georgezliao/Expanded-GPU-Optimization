# CLAUDE.md — Writing in this repository

This project involves CUDA kernels, PyTorch/autograd, and GPU performance experiments. When editing the README, benchmark notes, PR descriptions, or technical comments, write like someone documenting work they ran, not someone presenting a case study.

## Voice

* Keep the technical detail. Readers care about the exact workload, implementation, profiler counter, regression, and measurement conditions. Don't replace those with general claims about performance.
* Prefer plain, specific sentences: "The 2 KB jet spilled to local memory" rather than "The naive approach suffered from significant memory inefficiencies."
* Use "I" when it helps explain a decision, but don't invent reactions or personal anecdotes. Informal shorthand like `smem`, `bwd`, `ncu`, and `64K` is fine.
* Don't make every bullet follow result → explanation → takeaway. Some results only need a number. Some need a caveat. Some remain unexplained.
* Don't restate all the numbers from a table. Use the surrounding prose for implementation details, strange results, profiler evidence, and limitations.
* Don't turn the sequence of experiments into a tidy success narrative. Keep failed versions and dead ends. A kernel that was slower than eager may be more informative than the winning kernel.
* Avoid stock transitions and verdicts: "crucially," "notably," "ultimately," "clear winner," "unlock," "game-changing," "the key takeaway," "the next bottleneck," "not worth pursuing." These aren't universally forbidden, but repeated punchlines sound synthetic.
* Don't force a conclusion at the end of every section, use em dashes everywhere, or manufacture contrasts of the form "not X, but Y." Vary the rhythm naturally; don't artificially introduce mistakes or slang.

## Evidence

* Never alter benchmark numbers, hardware, versions, dtype, network dimensions, tolerances, or test counts to make the writing smoother.
* Separate observation from inference. "ncu reports 93% DRAM" is a measurement; "the jet spill explains the slowdown" is an interpretation. Signal uncertainty when appropriate.
* Be precise about what was timed: kernel time vs. Python-to-Python step time, host launches vs. GPU kernels, forward vs. training step. Don't conflate these.
* Name the baseline and batch size when giving a speedup. Don't compare absolute runtimes from A6000, A5000, and A40 runs as though they're same-device measurements.
* CUDA graph pool allocations may not appear in a peak-memory statistic. Explain the measurement rather than saying the graph uses zero memory.
* Leave unsupported hypotheses unresolved or label them as experiments to try. Don't claim an optimization will work before it has been measured.

## How to edit existing benchmark text

Preserve the author's meaning, measurement tables, negative results, and useful technical shorthand. First remove repetitive explanations and overconfident conclusions; only then rewrite sentences that are actually awkward. Don't simplify away the mechanisms. If you suspect a technical error, flag it separately instead of silently rewriting the claim. When asked to rewrite, output the edited text directly, without praising it or listing your editorial choices.

## Example

Avoid: "The fused CUDA backward emerged as the clear winner, genuinely achieving an impressive 8x speedup and revealing shared-memory occupancy as the next optimization frontier."

Prefer: "On the A40, the 64K step went from 23.6 ms eager to 2.94 ms with the fused backward. Backward alone takes 2.42 ms, compared with 0.42 ms for forward. The kernel uses 86 KB of shared memory per block, leaving room for one block/SM. Next time though, I may try reducing that footprint before tuning the tile shape again."

Aim for notes another engineer could use to reproduce or question the result. Not a polished story about why every decision was correct.
