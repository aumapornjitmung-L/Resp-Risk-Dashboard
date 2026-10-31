"""หน้าผลแบบจำลอง: เปรียบเทียบ ผลรายคลาส Confusion Matrix ROC และเวลาประมวลผล"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from respdash import state, ui
from respdash.constants import MODEL_ORDER
from respdash.modeling import roc_curves, time_models


def _hl(df: pd.DataFrame, cols):
    return df.style.format({c: "{:.4f}" for c in cols}).highlight_max(subset=cols, color="#D9F2EC")


def render() -> None:
    v = state.get_view()
    ui.hero("ผลการเปรียบเทียบแบบจำลอง", "ประสิทธิภาพการพยากรณ์บนชุดทดสอบ 20% (stratified, random_state = 42)", v.live)
    m = v.metrics()
    if m is None:
        ui.note("ยังไม่มีข้อมูล กรุณานำเข้าไฟล์ที่หน้าแดชบอร์ดหลัก", warn=True)
        return

    st.subheader("1) ประสิทธิภาพการพยากรณ์ (Macro)")
    st.dataframe(_hl(m, list(m.columns)), width="stretch")
    best = m["F1-Score (Macro)"].idxmax()
    ui.note(f"<b>{best}</b> ให้ F1 (Macro) สูงสุด {m.loc[best, 'F1-Score (Macro)']:.4f} — Random Forest และ XGBoost "
            "ใกล้เคียงกันมาก ส่วน SVM และ Logistic Regression มี Recall ไม่ต่ำแต่ Precision ต่ำ เพราะถ่วงน้ำหนักคลาส "
            "(class_weight = balanced) ทำให้จัดระดับสูงกว่าจริงบ่อย")
    st.plotly_chart(ui.metrics_bar(m, list(m.columns)), width="stretch")

    st.subheader("2) ผลรายคลาส")
    pc = v.per_class()
    tc = v.test_counts()
    st.caption("ชุดทดสอบ n = " + " · ".join(f"{k} {n}" for k, n in tc.items()) +
               " — คลาสที่มีตัวอย่างน้อย (High 6, Moderate 15) ค่าประเมินมีความไม่แน่นอนสูง")
    wide = state.per_class_wide(pc)
    fmt = {c: "{:.2f}" for c in MODEL_ORDER}
    st.dataframe(wide.style.format(fmt, na_rep="–"), width="stretch", hide_index=True, height=640)

    st.subheader("3) Confusion Matrix")
    cms = v.confusion()
    sel = st.radio("เลือกแบบจำลอง", ["ทั้งหมด"] + MODEL_ORDER, horizontal=True)
    names = MODEL_ORDER if sel == "ทั้งหมด" else [sel]
    cols = st.columns(2)
    for i, n in enumerate(names):
        cols[i % 2].plotly_chart(ui.confusion_heatmap(cms[n], n), width="stretch")
    st.caption("สีแสดงสัดส่วนต่อแถว (Recall) ตัวเลขในช่องคือจำนวนระเบียนจริง")

    if v.live:
        st.subheader("4) ROC (One-vs-Rest)")
        mn = st.selectbox("แบบจำลอง", MODEL_ORDER, index=MODEL_ORDER.index(best))
        st.plotly_chart(ui.roc_figure(roc_curves(v.bundle, mn), f"ROC — {mn}"), width="stretch")

        st.subheader("5) เวลาในการประมวลผล")
        st.caption("วัดบนเครื่องที่รันแอปนี้ ข้อมูลชุดเดียวกับตอนฝึกจริง (LR และ SVM ใช้ข้อมูลที่ปรับมาตรฐานแล้ว) "
                   "ค่าแปรผันตามเครื่อง ใช้เปรียบเทียบลำดับเร็ว–ช้า")
        n = st.slider("จำนวนรอบ", 3, 10, 5)
        if st.button("วัดเวลา"):
            with st.spinner("กำลังวัดเวลา..."):
                st.session_state["timing"] = time_models(v.bundle, n)
        if "timing" in st.session_state:
            st.dataframe(st.session_state["timing"].style.format("{:.4f}"), width="stretch")
