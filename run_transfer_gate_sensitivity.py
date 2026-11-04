"""Run only the pre-specified COURT-NMF transfer-gate sensitivity analysis."""
from court_nmf_experiment import preprocess, run_transfer_gate_sensitivity


if __name__ == "__main__":
    frame, matrix, _ = preprocess()
    print(f"Analytic player-seasons: {len(frame):,}", flush=True)
    print(run_transfer_gate_sensitivity(frame, matrix).to_string(index=False), flush=True)
