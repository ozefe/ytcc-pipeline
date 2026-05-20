"""Knob-sweep benchmark suite for the ytcc-pipeline.

Each sweep varies one `PipelineConfig` knob across a range of values and records speed
(wall + per-stage), resource usage (CPU% / RSS / VRAM as min / median / p95 / max
sampled every 250 ms), and quality (block counts, MISS fallbacks, formula LaTeX
recovery, table cells recovered, references parsed).

The runner is in `run_all`, the sweep definitions are in `sweeps`, the per-row execution
is in `runner`, the CSV / Markdown emission is in `reporting`, and the plot generation
is in `plot`.

Per-sweep outputs land under `benchmarks/results/sweeps/<name>.csv` and `<name>.md`;
per-sweep plots under `benchmarks/results/plots/<name>.png`. The runner is
resume-friendly -- sweeps with an existing CSV are skipped unless `--force` is passed.
"""
