"""ไปป์ไลน์เตรียมข้อมูล เหมือนโน้ตบุ๊กวิทยานิพนธ์ (ตัดข้อมูลซ้ำ ประมาณค่า รวมชุด สรุปรายเดือน)

ตั้งใจเขียนให้ผลลัพธ์เท่ากับโน้ตบุ๊กทุกค่า (ตรวจด้วยข้อมูลจริงแล้ว) แต่ใช้การคำนวณแบบเวกเตอร์
แทน lambda ใน groupby เพื่อให้เร็วขึ้น
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from .constants import (EN_TO_TH, RISK_CODE, RISK_LABELS, RISK_THRESHOLDS, REGION_MAP)

ProgressFn = Optional[Callable[[str, float], None]]


@dataclass
class PrepResult:
    monthly: pd.DataFrame            # ระดับจังหวัด-เดือน (ใช้สร้างแบบจำลอง)
    merged: Optional[pd.DataFrame]   # ระดับจังหวัด-วัน (None เมื่อนำเข้า CSV รายเดือน)
    quality: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------- ฤดูกาล / AQI / Risk
def season_of(dates) -> pd.Series:
    """ฤดูกาลตามเกณฑ์กรมอุตุนิยมวิทยา: ร้อน 15 ก.พ.–15 พ.ค., ฝน 16 พ.ค.–15 ต.ค., นอกนั้นหนาว"""
    d = pd.to_datetime(pd.Series(dates))
    m, day = d.dt.month, d.dt.day
    summer = ((m == 2) & (day >= 15)) | m.isin([3, 4]) | ((m == 5) & (day <= 15))
    rainy = ((m == 5) & (day >= 16)) | m.isin([6, 7, 8, 9]) | ((m == 10) & (day <= 15))
    return pd.Series(np.select([summer, rainy], ["Summer", "Rainy"], default="Winter"),
                     index=d.index)


def pm_to_aqi_level(v: pd.Series) -> pd.Series:
    return pd.Series(np.select([v <= 15.0, v <= 25.0, v <= 37.5, v <= 75.0],
                               [1, 2, 3, 4], default=5), index=v.index)


def risk_level(n: float) -> str:
    t = RISK_THRESHOLDS
    if n > t["extremely"]:
        return RISK_LABELS[3]
    if n > t["high"]:
        return RISK_LABELS[2]
    if n >= t["moderate"]:
        return RISK_LABELS[1]
    return RISK_LABELS[0]


def _tick(cb: ProgressFn, msg: str, frac: float) -> None:
    if cb:
        cb(msg, frac)


# ---------------------------------------------------------------- ผู้ป่วยนอก
def clean_rs(rs_raw: pd.DataFrame, q: dict) -> pd.DataFrame:
    rs = rs_raw.copy()
    rs["Date_serv"] = pd.to_datetime(rs["Date_serv"])
    rs["Birth day"] = pd.to_datetime(rs["Birth day"], errors="coerce")
    q["rs_raw_rows"] = int(len(rs))
    q["rs_duplicates"] = int(rs.duplicated().sum())
    rs = rs.drop_duplicates()
    q["rs_missing_province"] = int(rs["Province name"].isna().sum())
    rs = rs.dropna(subset=["Province name"])
    rs["age_years"] = (rs["Date_serv"] - rs["Birth day"]).dt.days / 365.25
    invalid = (rs["age_years"] < 0) | (rs["age_years"] > 110)
    q["rs_invalid_age"] = int(invalid.sum())
    rs.loc[invalid, "age_years"] = np.nan
    a = rs["age_years"]
    ag = pd.Series(np.select([a < 5, a < 15, a < 65], ["0-4", "5-14", "15-64"], default="65+"),
                   index=rs.index, dtype=object)
    ag[a.isna()] = np.nan
    rs["age_group"] = ag
    q["rs_clean_rows"] = int(len(rs))
    return rs


def daily_rs(rs: pd.DataFrame) -> pd.DataFrame:
    g = rs.assign(
        _one=1,
        _male=(rs["sex"] == 1).astype(int),
        _female=(rs["sex"] == 2).astype(int),
        _known=rs["age_years"].notna().astype(int),
        _a04=(rs["age_group"] == "0-4").astype(int),
        _a65=(rs["age_group"] == "65+").astype(int),
    ).groupby(["Province name", "Date_serv"])
    out = pd.DataFrame({
        "case_count": g["_one"].sum(),
        "n_male": g["_male"].sum(),
        "n_female": g["_female"].sum(),
        "mean_age": g["age_years"].mean(),
        "n_age_known": g["_known"].sum(),
        "n_age_0_4": g["_a04"].sum(),
        "n_age_65plus": g["_a65"].sum(),
    }).reset_index().rename(columns={"Province name": "Province", "Date_serv": "Date"})
    denom = out["n_age_known"].replace(0, np.nan)
    out["pct_age_0_4"] = (out["n_age_0_4"] / denom * 100).fillna(0)
    out["pct_age_65plus"] = (out["n_age_65plus"] / denom * 100).fillna(0)
    return out


# ---------------------------------------------------------------- PM2.5
def clean_pm(pm_raw: pd.DataFrame, q: dict) -> pd.DataFrame:
    pm = pm_raw.copy()
    pm["Date"] = pd.to_datetime(pm["Date"])
    pm["Province Name"] = pm["Province Name"].replace({"สุราษฏร์ธานี": "สุราษฎร์ธานี"})
    pm = pm.sort_values(["Province ID", "Date"])
    q["pm_raw_rows"] = int(len(pm))
    q["pm_missing_before"] = int(pm["PM2.5"].isna().sum())
    pm["PM2.5"] = pm.groupby("Province ID")["PM2.5"].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both"))
    q["pm_missing_after"] = int(pm["PM2.5"].isna().sum())
    return pm


def daily_pm(pm: pd.DataFrame) -> pd.DataFrame:
    out = pm.groupby(["Province Name", "Date"])["PM2.5"].mean().reset_index()
    out.columns = ["Province", "Date", "PM2.5"]
    return out


# ---------------------------------------------------------------- สภาพอากาศ
_NUM = ["Wind Speed", "Temp.", "Humi.", "heatidx", "Pres.", "Prec."]
_BOUNDS = {"Temp.": (-5, 50), "Humi.": (0, 100), "Pres.": (700, 1100),
           "Prec.": (0, 2000), "Wind Speed": (0, 60)}


def _parse_weather_date(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, format="%y/%m/%d")
    except (ValueError, TypeError):
        return pd.to_datetime(s)


def clean_weather(weather_raw: pd.DataFrame, q: dict) -> pd.DataFrame:
    w = weather_raw.copy()
    w["Date"] = _parse_weather_date(w["Date"])
    for c in _NUM:
        if c in w.columns:
            w[c] = pd.to_numeric(w[c], errors="coerce")
    for c, (lo, hi) in _BOUNDS.items():
        bad = (w[c] < lo) | (w[c] > hi)
        w.loc[bad, c] = np.nan
    w = w.sort_values(["ID", "Date"])
    q["weather_raw_rows"] = int(len(w))
    for c in ["Temp.", "Humi.", "Pres.", "Prec."]:
        w[c] = w.groupby("ID")[c].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both"))
    return w


def daily_weather(w: pd.DataFrame, station_raw: pd.DataFrame, q: dict) -> pd.DataFrame:
    st = station_raw.copy()
    st["Province"] = st["Province"].astype(str).str.strip()
    st.loc[st["ID"] == 1033, "Province"] = "Phitsanulok"   # สถานีที่ไม่มีจังหวัดกำกับ
    smap = st.set_index("ID")["Province"].to_dict()
    w = w.copy()
    w["Province_EN"] = w["ID"].map(smap)
    # รองรับไฟล์ที่ใช้ชื่อจังหวัดภาษาไทยอยู่แล้ว
    w["Province"] = w["Province_EN"].map(EN_TO_TH)
    w.loc[w["Province"].isna() & w["Province_EN"].isin(REGION_MAP.keys()), "Province"] = \
        w["Province_EN"]
    w = w.dropna(subset=["Province"])
    wd = w.groupby(["Province", "Date"]).agg(
        Temp=("Temp.", "mean"), Humidity=("Humi.", "mean"),
        Pressure=("Pres.", "mean"), Rainfall=("Prec.", "mean"),
        WindSpeed=("Wind Speed", "mean")).reset_index()

    wd["Region_tmp"] = wd["Province"].map(REGION_MAP)
    cols = ["Temp", "Humidity", "Pressure", "Rainfall", "WindSpeed"]
    region_daily = wd.groupby(["Region_tmp", "Date"])[cols].mean().reset_index()
    missing = sorted(set(REGION_MAP) - set(wd["Province"].unique()))
    q["weather_no_station"] = missing
    rows = []
    for p in missing:
        tmp = region_daily[region_daily["Region_tmp"] == REGION_MAP[p]].copy()
        tmp["Province"] = p
        rows.append(tmp.drop(columns=["Region_tmp"]))
    return pd.concat([wd.drop(columns=["Region_tmp"])] + rows, ignore_index=True)


# ---------------------------------------------------------------- รวมชุดข้อมูล
def build_merged(rs_d: pd.DataFrame, pm_d: pd.DataFrame, wd: pd.DataFrame, q: dict) -> pd.DataFrame:
    provinces = sorted(REGION_MAP.keys())
    dmin = min(pm_d["Date"].min(), wd["Date"].min(), rs_d["Date"].min())
    dmax = max(pm_d["Date"].max(), wd["Date"].max(), rs_d["Date"].max())
    base = pd.MultiIndex.from_product(
        [provinces, pd.date_range(dmin, dmax, freq="D")], names=["Province", "Date"]
    ).to_frame(index=False)
    m = (base.merge(pm_d, on=["Province", "Date"], how="left")
             .merge(wd, on=["Province", "Date"], how="left")
             .merge(rs_d, on=["Province", "Date"], how="left"))
    for c in ["case_count", "n_male", "n_female", "n_age_known", "n_age_0_4", "n_age_65plus"]:
        m[c] = m[c].fillna(0).astype(int)
    for c in ["pct_age_0_4", "pct_age_65plus"]:
        m[c] = m[c].fillna(0)

    m["Region"] = m["Province"].map(REGION_MAP)
    q["pm_provinces_missing"] = sorted(set(REGION_MAP) - set(pm_d["Province"].unique()))
    q["pm_rows_filled"] = int(m["PM2.5"].isna().sum())
    q["merged_rows"] = int(len(m))
    region_pm = m.groupby(["Region", "Date"])["PM2.5"].transform("mean")
    m["PM2.5"] = m["PM2.5"].fillna(region_pm)
    m["PM2.5"] = m["PM2.5"].fillna(m["PM2.5"].mean())
    for c in ["Temp", "Humidity", "Pressure", "Rainfall", "WindSpeed"]:
        m[c] = m.groupby("Province")[c].transform(lambda s: s.fillna(s.mean()))

    m["Season"] = season_of(m["Date"]).values
    m["AQI_Level"] = pm_to_aqi_level(m["PM2.5"]).values
    return m.sort_values(["Province", "Date"]).reset_index(drop=True)


def build_monthly(merged: pd.DataFrame) -> pd.DataFrame:
    m = merged.copy()
    m["YearMonth"] = m["Date"].dt.to_period("M")
    agg = m.groupby(["Province", "Region", "YearMonth"]).agg(
        case_count_month=("case_count", "sum"),
        PM25_mean=("PM2.5", "mean"), PM25_max=("PM2.5", "max"),
        Temp_mean=("Temp", "mean"), Humidity_mean=("Humidity", "mean"),
        Pressure_mean=("Pressure", "mean"), Rainfall_sum=("Rainfall", "sum"),
        AQI_mean=("AQI_Level", "mean"),
        n_age_known_month=("n_age_known", "sum"),
        n_age_0_4_month=("n_age_0_4", "sum"),
        n_age_65plus_month=("n_age_65plus", "sum"),
    ).reset_index()

    m["_wsum"] = m["mean_age"] * m["n_age_known"]
    wsum = m.groupby(["Province", "YearMonth"])["_wsum"].sum().reset_index()
    agg = agg.merge(wsum, on=["Province", "YearMonth"], how="left")
    agg["mean_age_month"] = agg["_wsum"] / agg["n_age_known_month"].replace(0, np.nan)
    agg = agg.drop(columns=["_wsum"])

    denom = agg["n_age_known_month"].replace(0, np.nan)
    agg["pct_age_0_4_month"] = (agg["n_age_0_4_month"] / denom * 100).fillna(0)
    agg["pct_age_65plus_month"] = (agg["n_age_65plus_month"] / denom * 100).fillna(0)

    agg["Risk_Level"] = agg["case_count_month"].apply(risk_level)
    agg["Risk_Level_code"] = agg["Risk_Level"].map(RISK_CODE)
    mid = agg["YearMonth"].dt.to_timestamp() + pd.Timedelta(days=14)
    agg["Season"] = season_of(mid).values
    agg["mean_age_month"] = agg["mean_age_month"].fillna(agg["mean_age_month"].mean())
    return agg


# ---------------------------------------------------------------- จุดเข้าหลัก
def run_pipeline(rs_raw, pm_raw, weather_raw, station_raw, progress: ProgressFn = None) -> PrepResult:
    q: dict = {}
    _tick(progress, "ทำความสะอาดข้อมูลผู้ป่วยนอก", 0.05)
    rs = clean_rs(rs_raw, q)
    _tick(progress, "สรุปข้อมูลผู้ป่วยเป็นรายวัน", 0.20)
    rs_d = daily_rs(rs)
    _tick(progress, "ประมาณค่า PM2.5 ที่ขาดหาย", 0.35)
    pm = clean_pm(pm_raw, q)
    pm_d = daily_pm(pm)
    _tick(progress, "ทำความสะอาดข้อมูลสภาพอากาศ", 0.50)
    w = clean_weather(weather_raw, q)
    wd = daily_weather(w, station_raw, q)
    _tick(progress, "รวมชุดข้อมูลทั้ง 3 แหล่ง", 0.70)
    merged = build_merged(rs_d, pm_d, wd, q)
    _tick(progress, "สรุปเป็นรายเดือนและกำหนด Risk Level", 0.85)
    monthly = build_monthly(merged)
    q["monthly_rows"] = int(len(monthly))
    q["pm_filled_pct"] = q["pm_rows_filled"] / q["merged_rows"] * 100
    q["mean_age_nan_daily"] = int(merged["mean_age"].isna().sum())
    return PrepResult(monthly=monthly, merged=merged, quality=q)


def prepare_monthly_csv(df: pd.DataFrame) -> PrepResult:
    """โหมดนำเข้า CSV รายเดือนที่ประมวลผลแล้ว: ตรวจ/แปลงชนิดข้อมูล แล้วคำนวณ Risk Level ใหม่"""
    m = df.copy()
    m["YearMonth"] = pd.PeriodIndex(m["YearMonth"].astype(str), freq="M")
    m["case_count_month"] = pd.to_numeric(m["case_count_month"], errors="coerce").fillna(0)
    if "Risk_Level" not in m.columns:
        m["Risk_Level"] = m["case_count_month"].apply(risk_level)
    m["Risk_Level_code"] = m["Risk_Level"].map(RISK_CODE)
    if m["Risk_Level_code"].isna().any():
        m["Risk_Level"] = m["case_count_month"].apply(risk_level)
        m["Risk_Level_code"] = m["Risk_Level"].map(RISK_CODE)
    m["Risk_Level_code"] = m["Risk_Level_code"].astype(int)
    m["mean_age_month"] = m["mean_age_month"].fillna(m["mean_age_month"].mean())
    q = {"monthly_rows": int(len(m)), "mode": "monthly_csv"}
    return PrepResult(monthly=m.reset_index(drop=True), merged=None, quality=q)
