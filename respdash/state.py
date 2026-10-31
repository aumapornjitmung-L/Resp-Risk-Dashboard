"""สถานะของแอป: รวมผลจากข้อมูลที่นำเข้า (live) หรือค่าอ้างอิงจากสไลด์ (reference) ให้หน้าต่างๆ ใช้ร่วมกัน"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from .constants import CLASS_NAMES, MODEL_ORDER
from .modeling import ModelBundle
from .pipeline import PrepResult
from .reference import load_reference


@st.cache_data(show_spinner=False)
def _reference() -> Optional[dict]:
    return load_reference()


@dataclass
class View:
    ref: Optional[dict]
    prep: Optional[PrepResult] = None
    bundle: Optional[ModelBundle] = None
    source: str = ""        # ชื่อแหล่งข้อมูลที่แสดงให้ผู้ใช้

    @property
    def live(self) -> bool:
        return self.prep is not None and self.bundle is not None

    @property
    def monthly(self) -> Optional[pd.DataFrame]:
        return self.prep.monthly if self.prep is not None else None

    @property
    def merged(self) -> Optional[pd.DataFrame]:
        return self.prep.merged if self.prep is not None else None

    # ---- ตารางที่ทุกหน้าใช้ ----
    def metrics(self) -> Optional[pd.DataFrame]:
        if self.live:
            return self.bundle.metrics
        if self.ref:
            return pd.DataFrame(self.ref["metrics"]).T.reindex(MODEL_ORDER)
        return None

    def per_class(self) -> Optional[pd.DataFrame]:
        if self.live:
            return self.bundle.per_class
        if not self.ref:
            return None
        rows = []
        tc = self.ref["dataset"]["test_counts"]
        for cname in CLASS_NAMES:
            for m in MODEL_ORDER:
                p, r, f, a = self.ref["per_class"][m][cname]
                rows.append({"Class": cname, "Support": tc[cname], "Model": m,
                             "Precision": p, "Recall": r, "F1-Score": f, "AUC (OVR)": a})
        return pd.DataFrame(rows)

    def confusion(self) -> Optional[Dict[str, np.ndarray]]:
        if self.live:
            return self.bundle.confusion
        if self.ref:
            return {m: np.array(v) for m, v in self.ref["confusion"].items()}
        return None

    def risk_counts(self) -> Optional[Dict[str, int]]:
        if self.live:
            return self.bundle.split_counts["all_counts"]
        return self.ref["dataset"]["risk_counts"] if self.ref else None

    def test_counts(self) -> Optional[Dict[str, int]]:
        if self.live:
            return self.bundle.split_counts["test_counts"]
        return self.ref["dataset"]["test_counts"] if self.ref else None


def get_view() -> View:
    ss = st.session_state
    return View(ref=_reference(), prep=ss.get("prep"), bundle=ss.get("bundle"),
                source=ss.get("source_label", ""))


def set_data(prep: PrepResult, bundle: ModelBundle, label: str) -> None:
    st.session_state["prep"] = prep
    st.session_state["bundle"] = bundle
    st.session_state["source_label"] = label


def clear_data() -> None:
    for k in ("prep", "bundle", "source_label"):
        st.session_state.pop(k, None)


def per_class_wide(df: pd.DataFrame) -> pd.DataFrame:
    """จัดตารางผลรายคลาสให้เหมือนสไลด์: แถว = (คลาส, ตัวชี้วัด) คอลัมน์ = โมเดล"""
    rows = []
    for cname in CLASS_NAMES:
        sub = df[df["Class"] == cname]
        if sub.empty:
            continue
        n = int(sub["Support"].iloc[0])
        for metric in ["Precision", "Recall", "F1-Score", "AUC (OVR)"]:
            row = {"คลาส": f"{cname} (n={n})", "ตัวชี้วัด": metric}
            for m in MODEL_ORDER:
                v = sub.loc[sub["Model"] == m, metric]
                row[m] = float(v.iloc[0]) if len(v) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)
