import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests


# ------------------------------------------------------------
# 1) Friedman + Kendall's W (Effektstärke)
# ------------------------------------------------------------
def friedman_with_kendallW(X: np.ndarray):
    """
    X: shape (n_seeds, k_varianten), höhere Werte = besser.
    Gibt (chi2, p, W) zurück.
    """
    n, k = X.shape
    stat, p = friedmanchisquare(*[X[:, j] for j in range(k)])
    W = stat / (n * k * (k - 1)) if k > 1 else np.nan
    return stat, p, W


# ------------------------------------------------------------
# 2) Familie testen: Baseline vs. alle Varianten (Wilcoxon, ein-/zweiseitig), Holm
# ------------------------------------------------------------
def test_family_wilcoxon(df: pd.DataFrame,
                         baseline_col: str = "Baseline",
                         seed_col: str = "seed",
                         alternative: str = "greater"):
    """
    Führt pro Variante einen gepaarten Wilcoxon-Test gegen die Baseline durch.
    alternative ∈ {'two-sided', 'greater', 'less'}
    """
    # Spalten prüfen
    if seed_col not in df.columns:
        raise ValueError(f"'{seed_col}' nicht in DataFrame.")
    if baseline_col not in df.columns:
        raise ValueError(f"baseline_col '{baseline_col}' nicht in DataFrame.")

    # Nur numerische Varianten-Spalten (ohne seed)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if baseline_col not in num_cols:
        raise ValueError(f"Baseline-Spalte '{baseline_col}' ist nicht numerisch.")
    variants_no_base = [c for c in num_cols if c not in (seed_col, baseline_col, "DA_WCT2_DR_Hin1","DA_FCUT_DR_Hin1")]
    if baseline_col == "DR_Hin1":  # Sonderfall Background_R als Baseline
        variants_no_base = [c for c in num_cols if c in ("DA_WCT2_DR_Hin1","DA_FCUT_DR_Hin1")]

    rows = []
    for v in variants_no_base:
        # Diffs (nur zur Beschreibung); Test nutzt rohe Paare
        d = (df[v] - df[baseline_col]).to_numpy()
        delta_mean   = float(np.mean(d)) if d.size else np.nan
        delta_median = float(np.median(d)) if d.size else np.nan

        # Wilcoxon: zero_method='pratt' behält echte Nullen, method='auto' wählt exact/approx
        res = wilcoxon(df[baseline_col].to_numpy(), df[v].to_numpy(),
                       zero_method="wilcox",  # echte Nullen behalten
                       alternative=alternative,
                       method="exact")
        p_raw = float(res.pvalue)

        rows.append({
            "variant": v,
            "delta_median": delta_median,
            "delta_mean": delta_mean,
            "p_raw": p_raw
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Holm-Korrektur innerhalb der Familie
    out["p_holm"] = multipletests(out["p_raw"].values, method="holm")[1]
    out["signif"] = out["p_holm"] < 0.05

    # Sortierung: erst p_holm, dann (bei Greater) absteigend delta_median, sonst aufsteigend
    asc_delta = (alternative == "less")
    out = out.sort_values(["p_holm", "delta_median"], ascending=[True, asc_delta]).reset_index(drop=True)
    return out


# ------------------------------------------------------------
# 3) Komplettlauf für eine Datei
# ------------------------------------------------------------
def analyze_csv(csv_path: str,
                baseline_col: str = "Baseline",
                seed_col: str = "seed",
                alternative: str = "greater"):
    """
    Liest CSV, berechnet Friedman+Kendall's W über alle Varianten (inkl. Baseline),
    und führt post-hoc Wilcoxon-Tests (mit Holm) für Baseline vs. jede Variante durch.
    """
    df = pd.read_csv(csv_path).sort_values(seed_col).reset_index(drop=True)

    # Numerische Matrix für Friedman (alle numerischen Spalten außer seed)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    ignore_cols = {"seed"}
    var_cols = [c for c in num_cols if c not in ignore_cols]


    if baseline_col not in var_cols:
        raise ValueError(f"Baseline-Spalte '{baseline_col}' fehlt oder ist nicht numerisch.")

    X = df[var_cols].to_numpy()
    chi2, p_fried, W = friedman_with_kendallW(X)

    # Post-hoc vs. Baseline (Wilcoxon, Holm)
    posthoc = test_family_wilcoxon(
        df, baseline_col=baseline_col, seed_col=seed_col, alternative=alternative
    )

    summary = {
        "friedman_chi2": chi2,
        "friedman_p": p_fried,
        "kendalls_W": W,
        "n_seeds": int(df.shape[0]),
        "k_variants": int(len(var_cols)),
        "baseline": baseline_col,
        "alternative": alternative
    }
    return summary, posthoc


# ------------------------------------------------------------
# 4) Beispiel-Aufruf (primäre/sekundäre Metrik als getrennte Familien)
# ------------------------------------------------------------
if __name__ == "__main__":
    # Pfade anpassen
    csv_primary = "0_map_50_seed.csv"        # mAP@0.5
    csv_secondary = "0_map_50_95_seed.csv"   # mAP@[0.5:0.95]

    # Richtung der H1:
    #   'greater'  -> Variante > Baseline
    #   'less'     -> Variante < Baseline
    #   'two-sided'-> Unterschied ungerichtet
    alt = "two-sided"

    summ_50, res_50 = analyze_csv(csv_primary, baseline_col="Baseline", alternative=alt)
    summ_5095, res_5095 = analyze_csv(csv_secondary, baseline_col="Baseline", alternative=alt)

    print("\n=== mAP@0.5 (primär) ===")
    print(summ_50)
    print(res_50)

    print("\n=== mAP@[0.5:0.95] (sekundär) ===")
    print(summ_5095)
    print(res_5095)

    summ_50, res_50 = analyze_csv(csv_primary, baseline_col="DR_Hin1", alternative=alt)
    summ_5095, res_5095 = analyze_csv(csv_secondary, baseline_col="DR_Hin1", alternative=alt)

    print("\n=== mAP@0.5 (primär) ===")
    print(summ_50)
    print(res_50)

    print("\n=== mAP@[0.5:0.95] (sekundär) ===")
    print(summ_5095)
    print(res_5095)

    # Pfade anpassen
    csv_primary = "1_map_50_seed.csv"        # mAP@0.5
    csv_secondary = "1_map_50_95_seed.csv"   # mAP@[0.5:0.95]

    # Richtung der H1:
    #   'greater'  -> Variante > Baseline
    #   'less'     -> Variante < Baseline
    #   'two-sided'-> Unterschied ungerichtet
    alt = "two-sided"

    summ_50, res_50 = analyze_csv(csv_primary, baseline_col="Baseline", alternative=alt)
    summ_5095, res_5095 = analyze_csv(csv_secondary, baseline_col="Baseline", alternative=alt)

    print("\n=== mAP@0.5 (primär) ===")
    print(summ_50)
    print(res_50)

    print("\n=== mAP@[0.5:0.95] (sekundär) ===")
    print(summ_5095)
    print(res_5095)

    summ_50, res_50 = analyze_csv(csv_primary, baseline_col="DR_Hin1", alternative=alt)
    summ_5095, res_5095 = analyze_csv(csv_secondary, baseline_col="DR_Hin1", alternative=alt)

    print("\n=== mAP@0.5 (primär) ===")
    print(summ_50)
    print(res_50)

    print("\n=== mAP@[0.5:0.95] (sekundär) ===")
    print(summ_5095)
    print(res_5095)
