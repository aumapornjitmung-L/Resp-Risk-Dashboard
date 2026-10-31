"""หน้าสำรวจข้อมูลและความสัมพันธ์: ฤดูกาล ภูมิภาค Pearson ANOVA"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from respdash import state, ui
from respdash.constants import REGION_TH
from respdash.stats import (anova_region, month_snapshot, pearson_table, region_summary, season_summary)


def _ref_frames(ref):
    s = pd.DataFrame(ref["season"], index=["PM25_mean", "case_count_month"]).T
    r = pd.DataFrame(ref["region"], index=["PM25_mean", "case_count_month"]).T
    return s, r


def render() -> None:
    v = state.get_view()
    ui.hero("สำรวจข้อมูลและความสัมพันธ์", "PM2.5 แปรผันตามฤดูกาลชัดเจน แต่ความสัมพันธ์เชิงเส้นกับจำนวนผู้ป่วยต่ำมาก", v.live)
    if v.live:
        season, region = season_summary(v.monthly), region_summary(v.monthly)
    elif v.ref:
        season, region = _ref_frames(v.ref)
    else:
        ui.note("ยังไม่มีข้อมูล", warn=True)
        return

    t1, t2, t3, t4 = st.tabs(["ฤดูกาลและภูมิภาค", "สหสัมพันธ์ Pearson", "ANOVA ภูมิภาค", "เจาะเดือน"])
    with t1:
        c1, c2 = st.columns(2)
        c1.plotly_chart(ui.season_chart(season), width="stretch")
        c2.plotly_chart(ui.region_chart(region), width="stretch")
        ui.note("ลำดับของฤดูร้อนและฤดูหนาวสลับกัน อาจสะท้อนการแพร่กระจายของเชื้อก่อโรคตามฤดูกาลควบคู่กับมลพิษ "
                "(ข้อสันนิษฐาน) ภาคกลางมีผู้ป่วยเฉลี่ยสูงเพราะรวมกรุงเทพฯ และปริมณฑล แม้ PM2.5 ต่ำกว่าภาคเหนือ — "
                "จำนวนผู้ป่วยที่บันทึกขึ้นกับขนาดประชากรและการเข้าถึงบริการ")
        if v.live:
            hm1, hm2 = st.columns(2)
            hm1.plotly_chart(ui.calendar_heatmap(v.monthly, "PM25_mean", "PM2.5 เฉลี่ยรายเดือน", "YlOrRd"),
                             width="stretch")
            hm2.plotly_chart(ui.calendar_heatmap(v.monthly, "case_count_month", "จำนวนผู้ป่วยเฉลี่ยต่อจังหวัด", "Blues"),
                             width="stretch")
    with t2:
        if v.live:
            pt = pearson_table(v.monthly)
            st.dataframe(pt.style.format({"r": "{:.3f}", "p-value": "{:.3g}"}), width="stretch")
            st.plotly_chart(ui.scatter_pm_cases(v.monthly), width="stretch")
            r = pt.loc["PM25_mean", "r"]
        else:
            pt = pd.Series(v.ref["pearson"], name="r").to_frame()
            st.dataframe(pt.style.format("{:.3f}"), width="stretch")
            r = v.ref["pearson"]["PM25_mean"]
        ui.note(f"<b>r = {r:.2f}</b> ระหว่าง PM2.5 เฉลี่ยรายเดือนกับจำนวนผู้ป่วยรายเดือน: ใกล้ 0 มาก แปลว่าไม่พบความสัมพันธ์"
                "เชิงเส้นในข้อมูลชุดนี้ ไม่ได้แปลว่า PM2.5 ไม่มีผลต่อสุขภาพ (ยังไม่ได้ทดสอบผลหน่วงเวลา)")
    with t3:
        if v.live and v.merged is not None:
            a = anova_region(v.merged)
            c = st.columns(4)
            ui.kpi(c[0], f"{a['F']:.2f}", "F (One-way ANOVA)")
            ui.kpi(c[1], f"{a['p']:.2e}", "p-value")
            ui.kpi(c[2], f"{a['eta2']:.3f}", "η² (ขนาดผล)", "ภูมิภาคอธิบายความแปรปรวนได้น้อย")
            ui.kpi(c[3], f"{a['H']:.1f}", "Kruskal-Wallis H", f"p = {a['p_kw']:.2g}")
            m = a["means"].rename("ผู้ป่วยเฉลี่ย/วัน").to_frame()
            m.index = [REGION_TH[i] for i in m.index]
            st.dataframe(m.style.format("{:.2f}"), width="stretch")
        elif v.live:
            ui.note("โหมด CSV รายเดือนไม่มีข้อมูลรายวัน จึงคำนวณ ANOVA รายวันไม่ได้", warn=True)
        else:
            c = st.columns(2)
            ui.kpi(c[0], f"{v.ref['anova']['F']:.2f}", "F (One-way ANOVA)", "ค่าอ้างอิงจากสไลด์")
            ui.kpi(c[1], f"{v.ref['anova']['eta2']:.3f}", "η² (ขนาดผล)")
        ui.note("ภูมิภาคต่างกันอย่างมีนัยสำคัญทางสถิติ แต่ขนาดผลเล็ก (η² ≈ 0.014) ควรพูดว่า “แตกต่างกัน” "
                "ไม่ใช่ “ภูมิภาคกำหนดจำนวนผู้ป่วย”")
    with t4:
        if not v.live:
            st.info("นำเข้าข้อมูลเพื่อเลือกเดือนที่ต้องการเจาะดู "
                    f"(ค่าอ้างอิง ก.ค. 2567: PM2.5 {v.ref['july_2024']['pm25_mean']}, ผู้ป่วยเฉลี่ย "
                    f"{v.ref['july_2024']['cases_mean']}/จังหวัด)")
        else:
            months = sorted(v.monthly["YearMonth"].astype(str).unique())
            default = months.index("2024-07") if "2024-07" in months else 0
            ym = st.selectbox("เลือกเดือน (ค.ศ.-เดือน)", months, index=default)
            s = month_snapshot(v.monthly, ym)
            c = st.columns(4)
            ui.kpi(c[0], f"{s['pm25_mean']:.1f}", "PM2.5 เฉลี่ยทุกจังหวัด", "µg/m³")
            ui.kpi(c[1], f"{s['cases_mean']:.1f}", "ผู้ป่วยเฉลี่ยต่อจังหวัด", f"มัธยฐาน {s['cases_median']:.0f}")
            ui.kpi(c[2], f"{s['cases_total']:,}", "ผู้ป่วยรวมทั้งประเทศ")
            ui.kpi(c[3], s["top_province"], f"{s['top_cases']:,} ราย", f"{s['top_share'] * 100:.0f}% ของทั้งประเทศ")
            st.caption("ค่าเฉลี่ยต่อจังหวัดถูกดึงขึ้นด้วยจังหวัดที่ใหญ่มาก (เช่น กรุงเทพมหานคร) "
                       "ควรรายงานมัธยฐานควบคู่")
