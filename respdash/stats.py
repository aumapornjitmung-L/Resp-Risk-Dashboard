"""สถิติที่ใช้ในสไลด์: Pearson, ANOVA/Kruskal-Wallis, VIF, สรุปตามฤดูกาล/ภูมิภาค/เดือน"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .constants import REGION_ORDER

CORR_VARS = ["PM25_mean", "PM25_max", "Temp_mean", "Humidity_mean", "Pressure_mean",
             "Rainfall_sum", "AQI_mean"]


def pearson_table(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in CORR_VARS:
        r, p = stats.pearsonr(monthly[c], monthly["case_count_month"])
        rows.append({"ตัวแปร": c, "r": r, "p-value": p})
    return pd.DataFrame(rows).set_index("ตัวแปร")


def season_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    g = monthly.groupby("Season")[["PM25_mean", "case_count_month"]].mean()
    g["n"] = monthly["Season"].value_counts()
    return g


def region_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    g = monthly.groupby("Region")[["PM25_mean", "case_count_month"]].mean()
    return g.reindex([r for r in REGION_ORDER if r in g.index])


def month_snapshot(monthly: pd.DataFrame, ym: str) -> Optional[Dict[str, object]]:
    sub = monthly[monthly["YearMonth"].astype(str) == ym]
    if sub.empty:
        return None
    top = sub.loc[sub["case_count_month"].idxmax()]
    return {
        "n_province": int(len(sub)),
        "pm25_mean": float(sub["PM25_mean"].mean()),
        "cases_total": int(sub["case_count_month"].sum()),
        "cases_mean": float(sub["case_count_month"].mean()),
        "cases_median": float(sub["case_count_month"].median()),
        "top_province": str(top["Province"]),
        "top_cases": int(top["case_count_month"]),
        "top_share": float(top["case_count_month"] / sub["case_count_month"].sum()),
        "top_pm25": float(top["PM25_mean"]),
    }


def anova_region(merged: pd.DataFrame) -> Dict[str, object]:
    groups = [g["case_count"].values for _, g in merged.groupby("Region")]
    F, p = stats.f_oneway(*groups)
    allv = np.concatenate(groups)
    grand = allv.mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    sst = ((allv - grand) ** 2).sum()
    H, pk = stats.kruskal(*groups)
    means = merged.groupby("Region")["case_count"].mean().reindex(
        [r for r in REGION_ORDER if r in merged["Region"].unique()])
    return {"F": float(F), "p": float(p), "eta2": float(ssb / sst), "H": float(H),
            "p_kw": float(pk), "means": means}


def vif_table(merged: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = merged[["PM2.5", "Temp", "Humidity", "Pressure", "Rainfall", "WindSpeed"]].copy()
    X = (X - X.mean()) / X.std()
    X = X.assign(const=1)
    vals = [variance_inflation_factor(X.values, i) for i in range(X.shape[1] - 1)]
    return pd.DataFrame({"Variable": X.columns[:-1], "VIF": vals}).set_index("Variable")


def age_by_risk(monthly: pd.DataFrame) -> pd.DataFrame:
    d = monthly.copy()
    d["pct_age_15_64_month"] = 100 - d["pct_age_0_4_month"] - d["pct_age_65plus_month"]
    out = d.groupby("Risk_Level")[["pct_age_0_4_month", "pct_age_15_64_month",
                                   "pct_age_65plus_month"]].mean()
    out.columns = ["0–4 ปี", "15–64 ปี", "65 ปีขึ้นไป"]
    return out
