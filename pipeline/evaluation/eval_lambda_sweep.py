"""
eval_lambda_sweep.py

SLO definitions:
- SLO(TTFT) = a + b * p_tokens (prefill scaling model)
- SLO(TPOT) = P70 percentile
- SLO(E2E)  = TTFT_SLO + TPOT_SLO * d_tokens
"""


import argparse
import pandas as pd
import numpy as np
import os
import json
from sklearn.linear_model import LinearRegression

from hw_router.constants import LAT_P95_LOG, STATIC_COST_P95, STATIC_COST_P95_IRT, STATIC_COST_P95_UMR

# ---------------------------------------------------------
#  Cost Normalization Constants (from training)
# ---------------------------------------------------------
latency_p95_log = LAT_P95_LOG
static_cost_p95 = STATIC_COST_P95
static_cost_p95_irt = STATIC_COST_P95_IRT
static_cost_p95_umr = STATIC_COST_P95_UMR

# ---------------------------------------------------------
#  SLO Calibration
# ---------------------------------------------------------

def fit_ttft_slo(df):
    valid = df.dropna(subset=["ttft_s", "p_tokens"])
    X = valid["p_tokens"].values.reshape(-1, 1)
    y = valid["ttft_s"].values

    model = LinearRegression()
    model.fit(X, y)

    a = float(model.intercept_)
    b = float(model.coef_[0])

    slack = 1.20
    return a * slack, b * slack


def compute_tpot_slo(df):
    vals = df["tpot_s_per_token"].dropna()
    return float(np.percentile(vals, 70))


# ---------------------------------------------------------
#  λ-sweep Logic
# ---------------------------------------------------------

def run_lambda_sweep(csv_path, lambdas=None):
    print(f"[Sweep] Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    # Predict latency = TTFT + length * TPOT
    df["pred_total_latency"] = (
        df["predicted_ttft"] +
        df["carrot_predicted_length"] * df["predicted_tpot"]
    ).replace([np.inf, -np.inf], np.nan).fillna(1e-6).clip(lower=1e-6)

    # ---------------------------------------------------------
    # COST NORMALIZATION (FIXED & CORRECT)
    # ---------------------------------------------------------

    # CARROT (static cost)
    df["static_cost_norm"] = (
        df["carrot_predicted_cost"] / static_cost_p95
    ).clip(upper=1.0)

    # OUR latency-based cost
    log_lat = np.log1p(df["pred_total_latency"])
    df["latency_cost_norm"] = (log_lat / latency_p95_log).clip(upper=1.0)

    #IRT Cost
    df["static_cost_norm_irt"] = (
        df["irt_cost_score"] / static_cost_p95_irt
    ).clip(upper=1.0)

    #UMR Cost
    df["static_cost_norm_umr"] = (
        df["umr_cost_score"] / static_cost_p95_umr
    ).clip(upper=1.0)

    # ---------------------------------------------------------
    # SLO calibration
    # ---------------------------------------------------------
    slo_a, slo_b = fit_ttft_slo(df)
    slo_tpot = compute_tpot_slo(df)

    print("\nCalibrated SLO definitions:")
    print(f"  TTFT_SLO(p) = {slo_a:.3f} + {slo_b:.7f} * p_tokens")
    print(f"  TPOT_SLO    = {slo_tpot:.5f} s/token\n")

    groups = df.groupby("prompt_id")
    results = []

    # ---------------------------------------------------------
    # λ-sweep
    # ---------------------------------------------------------
    for lam in lambdas:

        # CARROT score
        df["carrot_score"] = (
            lam * df["carrot_predicted_quality"]
            - (1 - lam) * df["static_cost_norm"]
        )

        # HW-Router score: IRT quality + hardware latency cost
        # (IRT chosen as quality component in the paper; any quality predictor can be used here)
        df["ours_score"] = (
            lam * df["irt_quality_score"]
            - (1 - lam) * df["latency_cost_norm"]
        )

        #IRT score
        df["irt_score"] = (
            lam * df["irt_quality_score"]
            - (1 - lam) * df["static_cost_norm"]
        )

        #UMR score
        df["umr_score"] = (
            lam * df["umr_quality_score"]
            - (1 - lam) * df["static_cost_norm"]
        )


        # select top model per prompt
        idx_c = groups["carrot_score"].idxmax()
        idx_o = groups["ours_score"].idxmax()
        idx_i = groups["irt_score"].idxmax()
        idx_u = groups["umr_score"].idxmax()

        sel_c = df.loc[idx_c]
        sel_o = df.loc[idx_o]
        sel_i = df.loc[idx_i]
        sel_u = df.loc[idx_u]

        # Quality / Latency metrics
        carrot_q = sel_c["actual_quality_score"].mean()
        carrot_lat = sel_c["latency_s"].mean()

        ours_q = sel_o["actual_quality_score"].mean()
        ours_lat = sel_o["latency_s"].mean()

        irt_q = sel_i["actual_quality_score"].mean()
        irt_lat = sel_i["latency_s"].mean()

        umr_q = sel_u["actual_quality_score"].mean()
        umr_lat = sel_u["latency_s"].mean()

        # -------------------------
        # SLO metrics
        # -------------------------
        c_ttft_slo = slo_a + slo_b * sel_c["p_tokens"]
        o_ttft_slo = slo_a + slo_b * sel_o["p_tokens"]
        i_ttft_slo = slo_a + slo_b * sel_i["p_tokens"]
        u_ttft_slo = slo_a + slo_b * sel_u["p_tokens"]

        carrot_slo_ttft = (sel_c["ttft_s"] <= c_ttft_slo).mean()
        ours_slo_ttft   = (sel_o["ttft_s"] <= o_ttft_slo).mean()
        irt_slo_ttft   = (sel_i["ttft_s"] <= i_ttft_slo).mean()
        umr_slo_ttft   = (sel_u["ttft_s"] <= u_ttft_slo).mean()

        carrot_slo_tpot = (sel_c["tpot_s_per_token"] <= slo_tpot).mean()
        ours_slo_tpot   = (sel_o["tpot_s_per_token"] <= slo_tpot).mean()
        irt_slo_tpot   = (sel_i["tpot_s_per_token"] <= slo_tpot).mean()
        umr_slo_tpot   = (sel_u["tpot_s_per_token"] <= slo_tpot).mean()

        c_e2e_slo = c_ttft_slo + slo_tpot * sel_c["d_tokens"]
        o_e2e_slo = o_ttft_slo + slo_tpot * sel_o["d_tokens"]
        i_e2e_slo = i_ttft_slo + slo_tpot * sel_i["d_tokens"]
        u_e2e_slo = u_ttft_slo + slo_tpot * sel_u["d_tokens"]

        carrot_slo_e2e = (sel_c["latency_s"] <= c_e2e_slo).mean()
        ours_slo_e2e   = (sel_o["latency_s"] <= o_e2e_slo).mean()
        irt_slo_e2e   = (sel_i["latency_s"] <= i_e2e_slo).mean()
        umr_slo_e2e   = (sel_u["latency_s"] <= u_e2e_slo).mean()

        # final record
        results.append({
            "lambda": lam,
            "carrot_quality": carrot_q,
            "carrot_latency": carrot_lat,
            "carrot_slo_ttft": carrot_slo_ttft,
            "carrot_slo_tpot": carrot_slo_tpot,
            "carrot_slo_e2e": carrot_slo_e2e,
            "ours_quality": ours_q,
            "ours_latency": ours_lat,
            "ours_slo_ttft": ours_slo_ttft,
            "ours_slo_tpot": ours_slo_tpot,
            "ours_slo_e2e": ours_slo_e2e,
            "irt_quality": irt_q,
            "irt_latency": irt_lat,
            "irt_slo_ttft": irt_slo_ttft,
            "irt_slo_tpot": irt_slo_tpot,
            "irt_slo_e2e": irt_slo_e2e,
            "umr_quality": umr_q,
            "umr_latency": umr_lat,
            "umr_slo_ttft": umr_slo_ttft,
            "umr_slo_tpot": umr_slo_tpot,
            "umr_slo_e2e": umr_slo_e2e,
            "num_prompts": len(sel_c)
        })

        print(f"\nλ = {lam:.2f}")
        print(f"  CARROT: Q={carrot_q:.4f}  LAT={carrot_lat:.2f}s  | "
              f"SLO(TTFT)={carrot_slo_ttft:.3f}  SLO(TPOT)={carrot_slo_tpot:.3f}  SLO(E2E)={carrot_slo_e2e:.3f}")
        print(f"  OURS  : Q={ours_q:.4f}  LAT={ours_lat:.2f}s  | "
              f"SLO(TTFT)={ours_slo_ttft:.3f}  SLO(TPOT)={ours_slo_tpot:.3f}  SLO(E2E)={ours_slo_e2e:.3f}")
        print(f"  IRT  : Q={irt_q:.4f}  LAT={irt_lat:.2f}s  | "
              f"SLO(TTFT)={irt_slo_ttft:.3f}  SLO(TPOT)={irt_slo_tpot:.3f}  SLO(E2E)={irt_slo_e2e:.3f}")
        print(f"  UMR  : Q={umr_q:.4f}  LAT={umr_lat:.2f}s  | "
              f"SLO(TTFT)={umr_slo_ttft:.3f}  SLO(TPOT)={umr_slo_tpot:.3f}  SLO(E2E)={umr_slo_e2e:.3f}")

    # save results
    out_path = "data/lambda_sweep_results_final_with_irt_umr.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\n[Sweep] Saved λ-sweep results → {out_path}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/evaluation_dataset_processed_full_with_umr_irt.csv")
    parser.add_argument("--lambda_start", type=float, default=0.0)
    parser.add_argument("--lambda_end",   type=float, default=1.0)
    parser.add_argument("--lambda_step",  type=float, default=0.1)

    args = parser.parse_args()

    lambdas = np.arange(
        args.lambda_start,
        args.lambda_end + 1e-9,
        args.lambda_step
    ).tolist()

    run_lambda_sweep(args.input, lambdas)


if __name__ == "__main__":
    main()
