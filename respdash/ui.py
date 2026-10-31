"""องค์ประกอบหน้าจอ: ธีม CSS, การ์ด KPI และกราฟ Plotly"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .constants import CLASS_COLORS, CLASS_NAMES, MODEL_ORDER, REGION_ORDER, REGION_TH, SEASON_TH

TEAL = "#0F8B8D"
NAVY = "#12263F"
MODEL_COLORS = {"Random Forest": "#0F8B8D", "XGBoost": "#12263F",
                "SVM": "#E8833A", "Logistic Regression": "#9AA5B1"}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stDataFrame, button, input { font-family: 'Sarabun', sans-serif !important; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }
h1, h2, h3 { color: #12263F; letter-spacing: -0.01em; }
.hero { background: linear-gradient(120deg, #12263F 0%, #0F6F73 100%); color: #fff; border-radius: 18px;
        padding: 26px 30px; margin-bottom: 18px; }
.hero h1 { color:#fff; font-size: 1.7rem; margin: 0 0 6px 0; }
.hero p { color:#D8EDEE; margin:0; font-size: 1rem; }
.badge { display:inline-block; padding: 3px 12px; border-radius: 999px; font-size: .82rem; font-weight:600; margin-top: 10px; }
.badge-ref { background:#FFF3D6; color:#7A5600; }
.badge-live { background:#D9F2EC; color:#0B6B57; }
.kpi { background:#fff; border:1px solid #E3E8EE; border-radius:14px; padding:16px 18px; height:100%; }
.kpi .v { font-size: 1.9rem; font-weight:700; color:#0F8B8D; line-height:1.1; }
.kpi .l { color:#5B636B; font-size:.92rem; margin-top:4px; }
.kpi .s { color:#8A939C; font-size:.8rem; margin-top:2px; }
.note { background:#F1F7F8; border-left:4px solid #0F8B8D; border-radius:8px; padding:12px 16px; color:#2B3440; }
.warn { background:#FFF6E5; border-left:4px solid #E8A33A; border-radius:8px; padding:12px 16px; color:#5a4310; }
.step { font-weight:700; color:#0F8B8D; }
section[data-testid="stSidebar"] { background:#F7F9FB; }
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, live: bool) -> None:
    badge = ('<span class="badge badge-live">● แสดงผลจากข้อมูลที่นำเข้า</span>' if live else
             '<span class="badge badge-ref">● แสดงค่าอ้างอิงจากสไลด์ (ยังไม่ได้นำเข้าข้อมูล)</span>')
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{badge}</div>', unsafe_allow_html=True)


def kpi(col, value: str, label: str, sub: str = "") -> None:
    small = " style=\"font-size:1.25rem\"" if len(str(value)) > 9 else ""
    col.markdown(f'<div class="kpi"><div class="v"{small}>{value}</div><div class="l">{label}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def note(text: str, warn: bool = False) -> None:
    st.markdown(f'<div class="{"warn" if warn else "note"}">{text}</div>', unsafe_allow_html=True)


def _layout(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), template="plotly_white",
                      font=dict(family="Sarabun, sans-serif", size=13),
                      title=dict(x=0, xanchor="left", font=dict(size=16)),
                      legend=dict(orientation="h", yanchor="top", y=-0.12, x=0))
    return fig


# ------------------------------------------------------------------ กราฟ
def risk_distribution(counts: Dict[str, int]) -> go.Figure:
    names = list(counts.keys())
    fig = go.Figure(go.Bar(x=names, y=[counts[n] for n in names], marker_color=CLASS_COLORS,
                           text=[f"{counts[n]:,}" for n in names], textposition="outside"))
    fig.update_layout(title="การกระจายระดับความเสี่ยง (จังหวัด-เดือน)", yaxis_title="จำนวนระเบียน")
    fig.update_yaxes(range=[0, max(counts.values()) * 1.18])
    return _layout(fig)


def metrics_bar(metrics: pd.DataFrame, cols: Iterable[str]) -> go.Figure:
    fig = go.Figure()
    for m in MODEL_ORDER:
        if m in metrics.index:
            fig.add_bar(name=m, x=[c.split(" (")[0] for c in cols], y=[metrics.loc[m, c] for c in cols],
                        marker_color=MODEL_COLORS[m],
                        text=[f"{metrics.loc[m, c]:.3f}" for c in cols], textposition="outside")
    fig.update_layout(barmode="group", title="เปรียบเทียบประสิทธิภาพ 4 แบบจำลอง (ชุดทดสอบ)")
    fig.update_yaxes(range=[0, 1.1])
    return _layout(fig, 380)


def confusion_heatmap(cm: np.ndarray, title: str) -> go.Figure:
    cm = np.asarray(cm)
    row = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row, out=np.zeros_like(cm, dtype=float), where=row > 0)
    fig = go.Figure(go.Heatmap(
        z=pct, x=CLASS_NAMES, y=CLASS_NAMES, colorscale="Blues", zmin=0, zmax=1, showscale=False,
        text=[[f"{cm[i][j]}" for j in range(4)] for i in range(4)], texttemplate="%{text}",
        hovertemplate="จริง: %{y}<br>ทำนาย: %{x}<br>จำนวน: %{text}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title="ทำนาย", yaxis_title="ค่าจริง")
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, 340)


def roc_figure(curves: Dict[str, tuple], title: str) -> go.Figure:
    fig = go.Figure()
    for (cname, (fpr, tpr, a)), color in zip(curves.items(), CLASS_COLORS):
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{cname} (AUC {a:.3f})", line=dict(color=color, width=2.4))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#9AA5B1"), showlegend=False)
    fig.update_layout(title=title, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return _layout(fig, 380)


def importance_bar(imp: pd.Series, top: int = 12, selected: Optional[Iterable[str]] = None) -> go.Figure:
    from .constants import FEATURE_TH
    s = imp.head(top)[::-1]
    sel = set(selected or [])
    colors = [TEAL if i in sel else "#B8C4CE" for i in s.index]
    fig = go.Figure(go.Bar(x=s.values, y=[FEATURE_TH.get(i, i) for i in s.index], orientation="h",
                           marker_color=colors, text=[f"{v:.3f}" for v in s.values], textposition="outside"))
    fig.update_layout(title=f"ความสำคัญของตัวแปร (Random Forest) — {top} อันดับแรก")
    fig.update_xaxes(range=[0, float(s.max()) * 1.22])
    return _layout(fig, 420)


def monthly_trend(monthly: pd.DataFrame) -> go.Figure:
    d = monthly.copy()
    d["Date"] = d["YearMonth"].dt.to_timestamp()
    g = d.groupby("Date").agg(pm=("PM25_mean", "mean"), cases=("case_count_month", "sum")).reset_index()
    fig = go.Figure()
    fig.add_bar(x=g["Date"], y=g["cases"], name="ผู้ป่วยรวมทั้งประเทศ", marker_color="#C9DDE3", yaxis="y2")
    fig.add_scatter(x=g["Date"], y=g["pm"], name="PM2.5 เฉลี่ย (µg/m³)", mode="lines+markers",
                    line=dict(color="#C0392B", width=2.4))
    fig.update_layout(title="แนวโน้มรายเดือน: PM2.5 เฉลี่ย และจำนวนผู้ป่วยรวม",
                      yaxis=dict(title="PM2.5 (µg/m³)"),
                      yaxis2=dict(title="จำนวนผู้ป่วย", overlaying="y", side="right", showgrid=False))
    return _layout(fig, 360)


def top_provinces(monthly: pd.DataFrame, n: int = 15) -> go.Figure:
    g = monthly.groupby("Province").agg(cases=("case_count_month", "mean"),
                                        pm=("PM25_mean", "mean")).nlargest(n, "cases").reset_index()
    fig = go.Figure()
    fig.add_bar(x=g["Province"], y=g["cases"], name="ผู้ป่วยเฉลี่ย/เดือน", marker_color="#8DC3CB")
    fig.add_scatter(x=g["Province"], y=g["pm"], name="PM2.5 เฉลี่ย", mode="lines+markers",
                    line=dict(color="#C0392B"), yaxis="y2")
    fig.update_layout(title=f"{n} จังหวัดที่มีผู้ป่วยเฉลี่ยต่อเดือนสูงสุด",
                      yaxis=dict(title="ผู้ป่วย/เดือน"),
                      yaxis2=dict(title="PM2.5", overlaying="y", side="right", showgrid=False))
    return _layout(fig, 360)


def season_chart(df: pd.DataFrame) -> go.Figure:
    order = [s for s in ["Summer", "Winter", "Rainy"] if s in df.index]
    fig = go.Figure()
    fig.add_bar(x=[SEASON_TH[s] for s in order], y=[df.loc[s, "PM25_mean"] for s in order],
                name="PM2.5 เฉลี่ย", marker_color="#C0392B",
                text=[f"{df.loc[s, 'PM25_mean']:.1f}" for s in order], textposition="outside")
    fig.add_bar(x=[SEASON_TH[s] for s in order], y=[df.loc[s, "case_count_month"] for s in order],
                name="ผู้ป่วยเฉลี่ย/จังหวัด-เดือน", marker_color="#0F8B8D",
                text=[f"{df.loc[s, 'case_count_month']:.1f}" for s in order], textposition="outside")
    fig.update_layout(barmode="group", title="PM2.5 และผู้ป่วยเฉลี่ยตามฤดูกาล")
    return _layout(fig, 340)


def region_chart(df: pd.DataFrame) -> go.Figure:
    order = [r for r in REGION_ORDER if r in df.index]
    fig = go.Figure()
    fig.add_bar(x=[REGION_TH[r] for r in order], y=[df.loc[r, "PM25_mean"] for r in order],
                name="PM2.5 เฉลี่ย", marker_color="#C0392B",
                text=[f"{df.loc[r, 'PM25_mean']:.1f}" for r in order], textposition="outside")
    fig.add_bar(x=[REGION_TH[r] for r in order], y=[df.loc[r, "case_count_month"] for r in order],
                name="ผู้ป่วยเฉลี่ย/จังหวัด-เดือน", marker_color="#0F8B8D",
                text=[f"{df.loc[r, 'case_count_month']:.1f}" for r in order], textposition="outside")
    fig.update_layout(barmode="group", title="PM2.5 และผู้ป่วยเฉลี่ยตามภูมิภาค")
    fig.update_xaxes(tickangle=0, tickfont=dict(size=11))
    return _layout(fig, 380)


def scatter_pm_cases(monthly: pd.DataFrame) -> go.Figure:
    d = monthly.copy()
    d["เดือน"] = d["YearMonth"].astype(str)
    fig = px.scatter(d, x="PM25_mean", y="case_count_month", color="Risk_Level", hover_data=["Province", "เดือน"],
                     color_discrete_sequence=CLASS_COLORS, opacity=.65,
                     category_orders={"Risk_Level": ["Low Risk (L)", "Moderate Risk (M)", "High Risk (H)",
                                                     "Extremely Risk (E)"]},
                     labels={"PM25_mean": "PM2.5 เฉลี่ยรายเดือน", "case_count_month": "จำนวนผู้ป่วยรายเดือน"})
    fig.update_layout(title="PM2.5 เฉลี่ย กับจำนวนผู้ป่วยรายเดือน (จังหวัด-เดือน)")
    return _layout(fig, 400)


def calendar_heatmap(monthly: pd.DataFrame, col: str, title: str, scale: str) -> go.Figure:
    d = monthly.copy()
    d["Year"] = d["YearMonth"].dt.year
    d["Month"] = d["YearMonth"].dt.month
    p = d.pivot_table(index="Month", columns="Year", values=col).reindex(range(1, 13))
    fig = go.Figure(go.Heatmap(z=p.values, x=[str(c) for c in p.columns],
                               y=["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.",
                                  "พ.ย.", "ธ.ค."], colorscale=scale, texttemplate="%{z:.1f}"))
    fig.update_layout(title=title)
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, 420)


def age_stack(df: pd.DataFrame) -> go.Figure:
    order = [r for r in ["Low Risk (L)", "Moderate Risk (M)", "High Risk (H)", "Extremely Risk (E)"] if r in df.index]
    fig = go.Figure()
    for c, color in zip(df.columns, ["#E8833A", "#8DC3CB", "#12263F"]):
        fig.add_bar(x=order, y=[df.loc[r, c] for r in order], name=c, marker_color=color)
    fig.update_layout(barmode="stack", title="สัดส่วนกลุ่มอายุของผู้ป่วย แยกตามระดับความเสี่ยง (%)")
    return _layout(fig, 360)
