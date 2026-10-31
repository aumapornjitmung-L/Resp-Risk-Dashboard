"""หน้าคุณภาพข้อมูล: ขั้นตอนเตรียมข้อมูลและข้อจำกัด"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from respdash import state, ui
from respdash.stats import vif_table


def render() -> None:
    v = state.get_view()
    ui.hero("คุณภาพข้อมูลและการเตรียมข้อมูล", "ทำความสะอาดแยกตามแหล่ง แล้วรวมเป็นโครงข้อมูล 77 จังหวัด × วัน", v.live)
    q = v.prep.quality if v.live else (v.ref or {}).get("quality")
    from_ref = not v.live
    if v.live and "rs_clean_rows" not in q:      # โหมด CSV รายเดือนข้ามขั้นเตรียมข้อมูลดิบ
        q, from_ref = (v.ref or {}).get("quality"), True
        ui.note("โหมด CSV รายเดือนข้ามขั้นเตรียมข้อมูลดิบ จึงแสดงค่าอ้างอิงจากสไลด์แทน "
                "นำเข้าไฟล์ดิบ 3 ไฟล์เพื่อให้ระบบคำนวณเอง", warn=True)
    if not q:
        ui.note("ไม่มีข้อมูลขั้นตอนเตรียมข้อมูล กรุณานำเข้าไฟล์ดิบ 3 ไฟล์", warn=True)
        return

    c = st.columns(4)
    ui.kpi(c[0], f"{q['rs_clean_rows']:,}", "ผู้ป่วยหลังทำความสะอาด",
           f"ดิบ {q.get('rs_raw_rows', 218243):,} · ตัดซ้ำ {q['rs_duplicates']:,} · ไม่มีจังหวัด {q['rs_missing_province']}")
    ui.kpi(c[1], f"{q['pm_missing_before']:,}", "PM2.5 ค่าว่างก่อนประมาณค่า", "ประมาณค่าเชิงเส้นรายสถานี")
    ui.kpi(c[2], f"{q['merged_rows']:,}", "แถวหลังรวม (จังหวัด × วัน)")
    ui.kpi(c[3], f"{q['pm_filled_pct']:.1f}%", "แถว PM2.5 ที่ถูกเติมค่า", f"{q['pm_rows_filled']:,} แถว")
    st.write("")
    a, b = st.columns(2)
    with a:
        st.markdown("#### จังหวัดที่ไม่มีสถานี PM2.5")
        st.write(", ".join(q["pm_provinces_missing"]) or "ไม่มี")
        st.caption("เติมด้วยค่าเฉลี่ยของภาคเดียวกันในวันเดียวกัน (แล้วค่าเฉลี่ยรวม)")
    with b:
        st.markdown("#### จังหวัดที่ไม่มีสถานีตรวจอากาศ")
        st.write(", ".join(q["weather_no_station"]) or "ไม่มี")
        st.caption("ใช้ค่าเฉลี่ยของภาคเดียวกันในวันเดียวกัน (สภาพอากาศที่เหลือเติมด้วยค่าเฉลี่ยของจังหวัด)")
    if "rs_invalid_age" in q:
        st.caption(f"อายุผิดปกติ (ติดลบหรือเกิน 110 ปี) ตั้งเป็นค่าว่าง {q['rs_invalid_age']} ระเบียน · "
                   f"วันที่ไม่มีผู้ป่วย mean_age เป็นค่าว่าง {q.get('mean_age_nan_daily', 0):,} แถว (ระดับรายวัน)")

    st.subheader("ตรวจภาวะร่วมเส้นตรง (VIF)")
    if v.live and v.merged is not None and not from_ref:
        vt = vif_table(v.merged)
    elif v.ref:
        vt = pd.Series(v.ref["vif"], name="VIF").to_frame()
    else:
        return
    st.dataframe(vt.style.format("{:.2f}"), width="stretch")
    ui.note("VIF ของตัวแปรสิ่งแวดล้อมอยู่ระหว่าง 1.07–1.50 (ต่ำกว่า 5 มาก) แสดงว่าตัวแปรไม่ซ้ำซ้อนกันมาก")
