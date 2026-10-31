"""หน้าตัวแปรสำคัญ: ความสำคัญของตัวแปร 19 → 12 ตัว"""
from __future__ import annotations

import streamlit as st

from respdash import state, ui
from respdash.constants import FEATURE_TH, SELECTED_FEATURES
from respdash.stats import age_by_risk


def render() -> None:
    v = state.get_view()
    ui.hero("ผลการวิจัย: ตัวแปรสำคัญ", "ความกดอากาศและโครงสร้างอายุผู้ป่วยมีความสำคัญสูงสุด", v.live)
    if not v.live:
        ui.note("หน้านี้คำนวณจากแบบจำลองที่ฝึกแล้ว กรุณานำเข้าข้อมูลที่หน้าแดชบอร์ดหลักก่อน "
                "(ค่าอ้างอิงจากสไลด์: ความกดอากาศ 0.174, สัดส่วนผู้ป่วยอายุ 65+ 0.173, อายุ 0–4 ปี 0.109)", warn=True)
        return
    b = v.bundle
    c1, c2 = st.columns([1.4, 1])
    c1.plotly_chart(ui.importance_bar(b.importance, 12, SELECTED_FEATURES), width="stretch")
    with c2:
        all_f1 = b.metrics.loc["Random Forest", "F1-Score (Macro)"]
        sel_f1 = b.selected_result["F1-Score (Macro)"]
        st.markdown("#### 19 → 12 ตัวแปร")
        st.metric("F1 (Macro) — 19 ตัวแปร", f"{all_f1:.4f}")
        st.metric("F1 (Macro) — 12 ตัวแปรที่คัดเลือก", f"{sel_f1:.4f}", f"{sel_f1 - all_f1:+.4f}")
        st.metric("AUC — 12 ตัวแปร", f"{b.selected_result['AUC (Macro OVR)']:.4f}")
        st.caption("แท่งสีเขียวคือ 12 ตัวแปรที่คัดเลือกไว้ในวิทยานิพนธ์")
    ui.note("<b>การตีความ:</b> ความกดอากาศเฉลี่ยน่าจะสะท้อนลักษณะที่ตั้งของพื้นที่ ไม่ใช่กลไกทางสุขภาพโดยตรง "
            "และตัวแปรอายุมาจากผู้ป่วยที่มารับบริการ ไม่ใช่ประชากร จึงตีความเป็นสาเหตุไม่ได้ "
            "ค่าความสำคัญนี้เป็นแบบ impurity-based ซึ่งอาจเอนเอียงไปทางตัวแปรต่อเนื่อง")
    with st.expander("ตารางความสำคัญของตัวแปรทั้งหมด"):
        t = b.importance.rename("ความสำคัญ").to_frame()
        t.insert(0, "ตัวแปร", [FEATURE_TH.get(i, i) for i in t.index])
        t["คัดเลือก"] = ["✓" if i in SELECTED_FEATURES else "" for i in t.index]
        st.dataframe(t.style.format({"ความสำคัญ": "{:.4f}"}), width="stretch")
    st.plotly_chart(ui.age_stack(age_by_risk(v.monthly)), width="stretch")
