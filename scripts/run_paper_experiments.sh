#!/bin/bash
# ============================================================================
# scripts/run_paper_experiments.sh
# Run all experiments for paper with 10 seeds for statistical significance
# ============================================================================

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate lei

# Configuration
SEEDS=10
FRAMES=20000
TRACES="traces/telemetry_000000.parquet"
CHECKPOINT="checkpoints/ppo_lyapunov"
OUTPUT_BASE="results/paper_final"
DEVICE="cuda:0"
LATENCY_NOISE_STD=2.0  # Stochastic noise for variance across seeds

echo "=============================================="
echo "PAPER EXPERIMENT SUITE"
echo "=============================================="
echo "Seeds: $SEEDS"
echo "Frames per eval: $FRAMES"
echo "Noise std: $LATENCY_NOISE_STD ms"
echo "=============================================="

# Create output directories
mkdir -p "$OUTPUT_BASE"/{baselines,ablations,stress,agent}

# ──────────────────────────────────────────────────────────────────────────────
# 1. BASELINE COMPARISON (10 seeds)
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "[1/4] Running baseline comparison with $SEEDS seeds..."
python scripts/eval_baselines.py \
    --traces "$TRACES" \
    --checkpoint "$CHECKPOINT" \
    --seeds "$SEEDS" \
    --frames "$FRAMES" \
    --latency-noise-std "$LATENCY_NOISE_STD" \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_BASE/baselines"

echo "    ✓ Baselines complete → $OUTPUT_BASE/baselines/"

# ──────────────────────────────────────────────────────────────────────────────
# 2. ABLATION STUDY (10 seeds × 8 configs = 80 runs)
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "[2/4] Running ablation study with $SEEDS seeds..."
python scripts/run_ablation.py \
    --traces "$TRACES" \
    --checkpoint "$CHECKPOINT" \
    --seeds "$SEEDS" \
    --frames "$FRAMES" \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_BASE/ablations"

echo "    ✓ Ablations complete → $OUTPUT_BASE/ablations/"

# ──────────────────────────────────────────────────────────────────────────────
# 3. STRESS TEST (10 seeds × 6 scenarios = 60 runs)
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Running stress tests with $SEEDS seeds..."
python scripts/stress_test.py \
    --traces "$TRACES" \
    --checkpoint "$CHECKPOINT" \
    --seeds "$SEEDS" \
    --frames "$FRAMES" \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_BASE/stress"

echo "    ✓ Stress tests complete → $OUTPUT_BASE/stress/"

# ──────────────────────────────────────────────────────────────────────────────
# 4. MAIN AGENT EVALUATION (10 seeds, detailed metrics)
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Running main agent evaluation with $SEEDS seeds..."
python scripts/eval_agent.py \
    --traces "$TRACES" \
    --checkpoint "$CHECKPOINT" \
    --seeds "$SEEDS" \
    --frames "$FRAMES" \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_BASE/agent"

echo "    ✓ Agent evaluation complete → $OUTPUT_BASE/agent/"

# ──────────────────────────────────────────────────────────────────────────────
# 5. GENERATE FIGURES
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Generating publication figures..."
python scripts/plot_results.py \
    --results-dir "$OUTPUT_BASE" \
    --output-dir "paper/figures"

echo "    ✓ Figures generated → paper/figures/"

# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "EXPERIMENT SUITE COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to: $OUTPUT_BASE/"
echo ""
echo "Key outputs:"
echo "  - Baseline comparison: $OUTPUT_BASE/baselines/baselines_comparison.csv"
echo "  - Wilcoxon tests:      $OUTPUT_BASE/baselines/wilcoxon_tests.csv"
echo "  - Ablation results:    $OUTPUT_BASE/ablations/ablation_summary.csv"
echo "  - Stress results:      $OUTPUT_BASE/stress/stress_summary.csv"
echo "  - Agent metrics:       $OUTPUT_BASE/agent/eval_summary.csv"
echo ""
echo "Generated LaTeX tables:"
echo "  - $OUTPUT_BASE/baselines/baselines_comparison.tex"
echo "  - $OUTPUT_BASE/ablations/ablation_table.tex"
echo ""
echo "Statistical significance:"
echo "  - With 10 seeds, Wilcoxon can achieve p < 0.002"
echo "  - Check wilcoxon_tests.csv for p-values"
echo ""
