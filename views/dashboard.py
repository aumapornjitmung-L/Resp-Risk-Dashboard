"""หน้าแดชบอร์ดหลัก: เมนูนำเข้าไฟล์ + ภาพรวมผลการวิจัย"""
from __future__ import annotations


import pandas as pd
import streamlit as st

from respdash import state, ui
from respdash.constants import CLASS_NAMES, MONTHLY_REQUIRED
from respdash.io import (DataFormatError, load_monthly_csv, load_pm, load_rs, load_weather)
from respdash.modeling import train_all
from respdash.pipeline import prepare_monthly_csv, run_pipeline
from respdash.reference import compare_with_slides

ACCEPT = ["csv", "xlsx", "xls"]


def _bytes(f):
    return f.getvalue(), f.name


def _run_raw(rs_f, pm_f, w_f, st_f):
    bar = st.progress(0.0, text="เริ่มประมวลผล")
    try:
        bar.progress(0.02, text="กำลังอ่านไฟล์ (ไฟล์ Excel ขนาดใหญ่ใช้เวลาประมาณ 30–60 วินาที)")
        rs = load_rs(*_bytes(rs_f))
        pm = load_pm(*_bytes(pm_f))
        weather, station = load_weather(*_bytes(w_f), station_data=_bytes(st_f) if st_f else None)

        def cb(msg, frac):
            bar.progress(min(0.1 + frac * 0.6, 0.7), text=msg)

        prep = run_pipeline(rs, pm, weather, station, progress=cb)
        bundle = train_all(prep.monthly,
                           progress=lambda m, f: bar.progress(0.7 + 0.3 * f, text=m))
        state.set_data(prep, bundle, f"ไฟล์ดิบ: {rs_f.name}, {pm_f.name}, {w_f.name}")
        bar.progress(1.0, text="เสร็จสิ้น")
        return True
    except DataFormatError as e:
        bar.empty()
        st.error(str(e))
    except Exception as e:  # แสดงข้อความที่อ่านเข้าใจได้ ไม่ปล่อยให้หน้าพัง
        bar.empty()
        st.error(f"ประมวลผลไม่สำเร็จ: {e}")
    return False


def _run_monthly(f):
    bar = st.progress(0.0, text="กำลังอ่านไฟล์")
    try:
        df = load_monthly_csv(*_bytes(f))
        prep = prepare_monthly_csv(df)
        bundle = train_all(prep.monthly, progress=lambda m, fr: bar.progress(0.1 + 0.9 * fr, text=m))
        state.set_data(prep, bundle, f"CSV รายเดือน: {f.name}")
        bar.progress(1.0, text="เสร็จสิ้น")
        return True
    except (DataFormatError, ValueError) as e:
        bar.empty()
        st.error(str(e))
    except Exception as e:
        bar.empty()
        st.error(f"ประมวลผลไม่สำเร็จ: {e}")
    return False


def _import_panel(v: state.View) -> None:
    with st.expander("📥 นำเข้าข้อมูล", expanded=not v.live):
        t_raw, t_csv = st.tabs(["ไฟล์ดิบ 3 ไฟล์ (ผู้ป่วย · PM2.5 · สภาพอากาศ)", "CSV รายเดือนที่ประมวลผลแล้ว"])
        with t_raw:
            st.markdown('<span class="step">ขั้นที่ 1</span> เลือกไฟล์ (รองรับ .csv และ .xlsx)', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            rs_f = c1.file_uploader("ผู้ป่วยนอก ICD-10 J00–J99 (RS data)", type=ACCEPT, key="up_rs",
                                    help="คอลัมน์ที่ต้องมี: Date_serv, Birth day, pid, sex, Province name")
            pm_f = c2.file_uploader("PM2.5 รายวัน", type=ACCEPT, key="up_pm",
                                    help="คอลัมน์ที่ต้องมี: Date, Province ID, PM2.5, Province Name")
            w_f = c3.file_uploader("สภาพอากาศรายวัน", type=ACCEPT, key="up_w",
                                   help="Excel หลายชีต (มีชีตสถานี ID/Province) หรือ CSV ที่มีคอลัมน์ "
                                        "ID, Date, Wind Speed, Temp., Humi., Pres., Prec.")
            st_f = None
            if w_f is not None and w_f.name.lower().endswith(".csv"):
                st_f = st.file_uploader("ตารางสถานี (ID, Province) — จำเป็นเมื่อสภาพอากาศเป็น CSV",
                                        type=ACCEPT, key="up_st")
            ready = rs_f is not None and pm_f is not None and w_f is not None
            st.markdown('<span class="step">ขั้นที่ 2</span> ประมวลผลและฝึกแบบจำลอง 4 แบบ', unsafe_allow_html=True)
            if st.button("▶ ประมวลผลและฝึกแบบจำลอง", type="primary", disabled=not ready, key="run_raw"):
                if _run_raw(rs_f, pm_f, w_f, st_f):
                    st.rerun()
            if not ready:
                st.caption("เลือกครบทั้ง 3 ไฟล์เพื่อเปิดปุ่ม")
        with t_csv:
            st.markdown("ใช้เมื่อมีตารางรายเดือนอยู่แล้ว (เช่นดาวน์โหลดจากแดชบอร์ดนี้) ข้ามขั้นเตรียมข้อมูลดิบ "
                        "จึงเร็วกว่า แต่ไม่มีสถิติระดับรายวัน (VIF, ANOVA)")
            f = st.file_uploader("CSV รายเดือน (ระดับจังหวัด-เดือน)", type=["csv"], key="up_monthly")
            st.caption("คอลัมน์ที่ต้องมี: " + ", ".join(MONTHLY_REQUIRED))
            tmpl = pd.DataFrame(columns=MONTHLY_REQUIRED).to_csv(index=False).encode("utf-8-sig")
            st.download_button("ดาวน์โหลดไฟล์เทมเพลต (หัวคอลัมน์)", tmpl, "monthly_template.csv", "text/csv")
            if st.button("▶ โหลดและฝึกแบบจำลอง", type="primary", disabled=f is None, key="run_csv"):
                if _run_monthly(f):
                    st.rerun()
        if v.live and st.button("ล้างข้อมูลที่นำเข้า (กลับไปค่าอ้างอิงจากสไลด์)"):
            state.clear_data()
            st.rerun()


def render() -> None:
    v = state.get_view()
    ui.hero("แดชบอร์ดพยากรณ์ความเสี่ยงโรคระบบทางเดินหายใจ",
            "การพยากรณ์ระดับความเสี่ยงรายจังหวัดรายเดือนจาก PM2.5 สภาพอากาศ และโครงสร้างอายุผู้ป่วย "
            "ด้วย 4 แบบจำลองการเรียนรู้ของเครื่อง", v.live)
    _import_panel(v)

    if v.live:
        st.caption(f"แหล่งข้อมูล: {v.source}")
    metrics, ref = v.metrics(), v.ref
    if metrics is None:
        ui.note("ยังไม่มีข้อมูล — กรุณานำเข้าไฟล์ที่เมนู “นำเข้าข้อมูล” ด้านบน", warn=True)
        return

    # ---------- KPI ----------
    counts = v.risk_counts()
    total = sum(counts.values())
    best = metrics["F1-Score (Macro)"].idxmax()
    c = st.columns(5)
    n_prov = v.monthly["Province"].nunique() if v.live else ref["dataset"]["provinces"]
    n_month = v.monthly["YearMonth"].nunique() if v.live else ref["dataset"]["months"]
    ui.kpi(c[0], f"{total:,}", "ระเบียนจังหวัด-เดือน", f"{n_prov} จังหวัด × {n_month} เดือน")
    ui.kpi(c[1], f"{counts[CLASS_NAMES[3]] + counts[CLASS_NAMES[2]]:,}", "ระดับ High + Extremely",
           f"{(counts[CLASS_NAMES[3]] + counts[CLASS_NAMES[2]]) / total * 100:.1f}% ของทั้งหมด")
    ui.kpi(c[2], best, "แบบจำลองที่ F1 (Macro) สูงสุด")
    ui.kpi(c[3], f"{metrics.loc[best, 'F1-Score (Macro)']:.4f}", "F1-Score (Macro)", "ชุดทดสอบ 20%")
    ui.kpi(c[4], f"{metrics.loc[best, 'AUC (Macro OVR)']:.4f}", "AUC (One-vs-Rest, Macro)")
    st.write("")

    # ---------- กราฟหลัก ----------
    g1, g2 = st.columns([1, 1.25])
    g1.plotly_chart(ui.risk_distribution(counts), width="stretch")
    g2.plotly_chart(ui.metrics_bar(metrics, ["Accuracy", "Precision (Macro)", "Recall (Macro)",
                                             "F1-Score (Macro)", "AUC (Macro OVR)"]), width="stretch")
    if v.live:
        g3, g4 = st.columns([1.25, 1])
        g3.plotly_chart(ui.monthly_trend(v.monthly), width="stretch")
        g4.plotly_chart(ui.top_provinces(v.monthly), width="stretch")
    else:
        ui.note("แสดงค่าอ้างอิงจากสไลด์ นำเข้าข้อมูลเพื่อดูแนวโน้มรายเดือนและจังหวัดที่มีผู้ป่วยสูง "
                "รวมถึงให้ระบบคำนวณผลเองทั้งหมด")

    st.info("ข้อควรระวัง: ระดับความเสี่ยงสร้างจากจำนวนผู้ป่วยรายเดือน (ค่าตัด 50/100/150 ปรับใช้จากเกณฑ์อัตราป่วย"
            "ต่อแสนประชากรของกรมควบคุมโรค) จึงสะท้อนขนาดประชากรและการเข้าถึงบริการด้วย ผลนี้ใช้เป็นเครื่องมือคัดกรอง"
            "เบื้องต้น ไม่ใช่การระบุสาเหตุของโรค", icon="ℹ️")

    # ---------- เทียบกับสไลด์ + ดาวน์โหลด ----------
    if v.live:
        d1, d2 = st.columns(2)
        csv = v.monthly.assign(YearMonth=v.monthly["YearMonth"].astype(str)).to_csv(
            index=False).encode("utf-8-sig")
        d1.download_button("⬇ ดาวน์โหลดตารางรายเดือน (CSV)", csv, "monthly_dataset.csv", "text/csv")
        d2.download_button("⬇ ดาวน์โหลดตัวชี้วัดโมเดล (CSV)", v.bundle.metrics.round(4).to_csv().encode("utf-8-sig"),
                           "model_metrics.csv", "text/csv")
        if ref is not None:
            with st.expander("🔍 ตรวจเทียบกับค่าบนสไลด์ (ใช้เมื่อนำเข้าชุดข้อมูลเดิมของวิทยานิพนธ์)"):
                cmp = compare_with_slides(ref, v.bundle.metrics, v.monthly, v.prep.quality)
                ok = int((cmp["สถานะ"].str.startswith("✓")).sum())
                st.markdown(f"**ตรงกับสไลด์ {ok} จาก {len(cmp)} รายการ**")
                st.dataframe(cmp, width="stretch", hide_index=True)
