# ============================================================================
# scripts/run_paper_experiments.ps1
# Run all experiments for paper with 10 seeds for statistical significance
# ============================================================================

$ErrorActionPreference = "Stop"

# Configuration
$SEEDS = 10
$FRAMES = 20000
$TRACES = "traces/telemetry_000000.parquet"
$CHECKPOINT = "checkpoints/ppo_lyapunov"
$OUTPUT_BASE = "results/paper_final"
$DEVICE = "cuda:0"
$LATENCY_NOISE_STD = 2.0

Write-Host "=============================================="
Write-Host "PAPER EXPERIMENT SUITE"
Write-Host "=============================================="
Write-Host "Seeds: $SEEDS"
Write-Host "Frames per eval: $FRAMES"
Write-Host "Noise std: $LATENCY_NOISE_STD ms"
Write-Host "=============================================="

# Create output directories
foreach ($dir in @("baselines", "ablations", "stress", "agent")) {
    New-Item -ItemType Directory -Force -Path "$OUTPUT_BASE/$dir" | Out-Null
}

# ──────────────────────────────────────────────────────────────────────────────
# 1. BASELINE COMPARISON (10 seeds)
# ──────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[1/4] Running baseline comparison with $SEEDS seeds..."
python scripts/eval_baselines.py `
    --traces "$TRACES" `
    --checkpoint "$CHECKPOINT" `
    --seeds $SEEDS `
    --frames $FRAMES `
    --latency-noise-std $LATENCY_NOISE_STD `
    --device "$DEVICE" `
    --output-dir "$OUTPUT_BASE/baselines"

Write-Host "    ✓ Baselines complete → $OUTPUT_BASE/baselines/"

# ──────────────────────────────────────────────────────────────────────────────
# 2. MAIN AGENT EVALUATION (10 seeds, detailed metrics)
# ──────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[2/4] Running main agent evaluation with $SEEDS seeds..."
python scripts/eval_agent.py `
    --traces "$TRACES" `
    --checkpoint "$CHECKPOINT" `
    --seeds $SEEDS `
    --frames $FRAMES `
    --device "$DEVICE" `
    --output-dir "$OUTPUT_BASE/agent"

Write-Host "    ✓ Agent evaluation complete → $OUTPUT_BASE/agent/"

# ──────────────────────────────────────────────────────────────────────────────
# 3. GENERATE FIGURES
# ──────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[3/4] Generating publication figures..."
python scripts/plot_results.py `
    --results-dir "$OUTPUT_BASE" `
    --output-dir "paper/figures"

Write-Host "    ✓ Figures generated → paper/figures/"

# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=============================================="
Write-Host "EXPERIMENT SUITE COMPLETE"
Write-Host "=============================================="
Write-Host ""
Write-Host "Results saved to: $OUTPUT_BASE/"
Write-Host ""
Write-Host "Key outputs:"
Write-Host "  - Baseline comparison: $OUTPUT_BASE/baselines/baselines_comparison.csv"
Write-Host "  - Wilcoxon tests:      $OUTPUT_BASE/baselines/wilcoxon_tests.csv"
Write-Host "  - Agent metrics:       $OUTPUT_BASE/agent/eval_summary.csv"
Write-Host ""
