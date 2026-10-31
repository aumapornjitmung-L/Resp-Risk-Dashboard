"""หน้าทำนาย: กรอกค่าเอง หรือนำเข้า CSV หลายแถว"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from respdash import state, ui
from respdash.constants import (FEATURE_COLS, FEATURE_TH, MODEL_ORDER, REGION_ORDER, REGION_TH,
                                SEASON_TH)
from respdash.modeling import predict_rows


def _row_from_inputs(b, vals: dict, region: str, season: str) -> pd.DataFrame:
    row = {c: vals[c] for c in FEATURE_COLS}
    for col in b.feature_names:
        if col.startswith("Region_"):
            row[col] = int(col == f"Region_{region}")
        elif col.startswith("Season_"):
            row[col] = int(col == f"Season_{season}")
    return pd.DataFrame([row])


def render() -> None:
    v = state.get_view()
    ui.hero("ทำนายระดับความเสี่ยง", "กรอกค่ารายเดือนของจังหวัด หรือนำเข้า CSV หลายแถวเพื่อทำนายพร้อมกัน", v.live)
    if not v.live:
        ui.note("ต้องนำเข้าข้อมูลและฝึกแบบจำลองที่หน้าแดชบอร์ดหลักก่อน", warn=True)
        return
    b = v.bundle
    model = st.selectbox("แบบจำลองที่ใช้ทำนาย", MODEL_ORDER, index=0)
    t1, t2 = st.tabs(["กรอกค่าเอง", "นำเข้า CSV"])
    with t1:
        med = v.monthly[FEATURE_COLS].median()
        lo, hi = v.monthly[FEATURE_COLS].min(), v.monthly[FEATURE_COLS].max()
        c = st.columns(3)
        region = c[0].selectbox("ภูมิภาค", [r for r in REGION_ORDER], format_func=lambda r: REGION_TH[r])
        season = c[1].selectbox("ฤดูกาล", list(SEASON_TH), format_func=lambda s: SEASON_TH[s])
        vals = {}
        cols = st.columns(3)
        for i, f in enumerate(FEATURE_COLS):
            vals[f] = cols[i % 3].number_input(FEATURE_TH[f], float(lo[f]), float(hi[f]) * 1.5 + 1e-6,
                                              float(med[f]), key=f"in_{f}")
        if st.button("ทำนาย", type="primary"):
            out = predict_rows(b, _row_from_inputs(b, vals, region, season), model)
            lvl = out["ระดับความเสี่ยงที่ทำนาย"].iloc[0]
            st.success(f"ระดับความเสี่ยงที่ทำนาย: **{lvl}**")
            st.dataframe(out.drop(columns=["ระดับความเสี่ยงที่ทำนาย"]).style.format("{:.1%}"),
                         width="stretch", hide_index=True)
            st.caption("ผลนี้เป็นการคัดกรองเบื้องต้น ใช้ประกอบการตัดสินใจ ไม่ใช่การวินิจฉัยหรือระบุสาเหตุ")
    with t2:
        st.caption("CSV ต้องมีคอลัมน์ตัวแปร 10 ตัว: " + ", ".join(FEATURE_COLS) +
                   " และ Region (North, Northeast, Central, East, South-East-Coast, South-West-Coast), "
                   "Season (Summer, Rainy, Winter)")
        f = st.file_uploader("CSV สำหรับทำนาย", type=["csv"], key="up_pred")
        if f is not None:
            df = pd.read_csv(f, encoding="utf-8-sig")
            need = FEATURE_COLS + ["Region", "Season"]
            miss = [c for c in need if c not in df.columns]
            if miss:
                st.error("ขาดคอลัมน์: " + ", ".join(miss))
            else:
                X = df[FEATURE_COLS].copy()
                for col in b.feature_names:
                    if col.startswith("Region_"):
                        X[col] = (df["Region"] == col.split("_", 1)[1]).astype(int)
                    elif col.startswith("Season_"):
                        X[col] = (df["Season"] == col.split("_", 1)[1]).astype(int)
                out = predict_rows(b, X, model)
                res = pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
                st.dataframe(res, width="stretch", hide_index=True)
                st.download_button("⬇ ดาวน์โหลดผลทำนาย (CSV)", res.to_csv(index=False).encode("utf-8-sig"),
                                   "predictions.csv", "text/csv")
