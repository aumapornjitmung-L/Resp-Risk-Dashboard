"""ค่าอ้างอิงจากสไลด์ (data/thesis_reference.json) และการเทียบกับผลที่คำนวณใหม่"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REF_PATH = Path(__file__).resolve().parent.parent / "data" / "thesis_reference.json"


def load_reference() -> Optional[dict]:
    try:
        return json.loads(REF_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def compare_with_slides(ref: dict, metrics: pd.DataFrame, monthly: pd.DataFrame,
                        quality: Optional[dict], tol_metric: float = 0.0006) -> pd.DataFrame:
    """ตารางเทียบค่าที่คำนวณกับค่าบนสไลด์ (สถานะ ✓ เมื่อต่างไม่เกินค่าคลาดเคลื่อนปัดเศษ)"""
    rows: List[Dict[str, object]] = []

    def add(item, slide, calc, tol, fmt="{:.4f}"):
        ok = abs(float(slide) - float(calc)) <= tol
        rows.append({"รายการ": item, "ค่าบนสไลด์": fmt.format(slide), "ค่าที่คำนวณ": fmt.format(calc),
                     "สถานะ": "✓ ตรง" if ok else "✗ ต่าง"})

    for model, vals in ref["metrics"].items():
        if model not in metrics.index:
            continue
        for k in ["Accuracy", "F1-Score (Macro)", "AUC (Macro OVR)"]:
            add(f"{model} · {k.split(' (')[0]}", vals[k], metrics.loc[model, k], tol_metric)

    add("จำนวนระเบียนรายเดือน", ref["dataset"]["records"], len(monthly), 0, "{:,.0f}")
    counts = monthly["Risk_Level"].value_counts()
    short = {"Low (L)": "Low Risk (L)", "Moderate (M)": "Moderate Risk (M)",
             "High (H)": "High Risk (H)", "Extremely (E)": "Extremely Risk (E)"}
    for k, v in ref["dataset"]["risk_counts"].items():
        add(f"ระเบียนระดับ {k}", v, int(counts.get(short[k], 0)), 0, "{:,.0f}")

    ss = monthly.groupby("Season")[["PM25_mean", "case_count_month"]].mean()
    for s, (pm, cs) in ref["season"].items():
        if s in ss.index:
            add(f"PM2.5 เฉลี่ย · {s}", pm, ss.loc[s, "PM25_mean"], 0.05, "{:.1f}")
            add(f"ผู้ป่วยเฉลี่ย · {s}", cs, ss.loc[s, "case_count_month"], 0.05, "{:.1f}")

    if quality and "rs_clean_rows" in quality:
        q = ref["quality"]
        for key, label in [("rs_duplicates", "ตัดข้อมูลซ้ำ"), ("rs_clean_rows", "ผู้ป่วยหลังทำความสะอาด"),
                           ("pm_missing_before", "PM2.5 ค่าว่างก่อนประมาณค่า"),
                           ("merged_rows", "แถวหลังรวม (77×วัน)"), ("pm_rows_filled", "แถว PM2.5 ที่ถูกเติมค่า")]:
            if key in quality:
                add(label, q[key], quality[key], 0, "{:,.0f}")
    return pd.DataFrame(rows)
