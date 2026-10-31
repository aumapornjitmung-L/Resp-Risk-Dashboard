"""อ่านไฟล์ที่ผู้ใช้นำเข้า (CSV หรือ Excel) เป็น DataFrame พร้อมตรวจคอลัมน์"""
from __future__ import annotations

import io
from typing import Dict, Optional, Tuple

import pandas as pd

RS_REQUIRED = ["Date_serv", "Birth day", "pid", "sex", "Province name"]
PM_REQUIRED = ["Date", "Province ID", "PM2.5", "Province Name"]
WEATHER_REQUIRED = ["ID", "Date", "Wind Speed", "Temp.", "Humi.", "Pres.", "Prec."]
STATION_REQUIRED = ["ID", "Province"]

_ENCODINGS = ("utf-8-sig", "utf-8", "cp874", "tis-620", "latin-1")


class DataFormatError(ValueError):
    """ไฟล์ที่นำเข้ามีรูปแบบไม่ตรงกับที่ระบบรองรับ (ข้อความเป็นภาษาไทย แสดงต่อผู้ใช้ได้)"""


def _is_excel(name: str) -> bool:
    return name.lower().endswith((".xlsx", ".xlsm", ".xls"))


def read_table(data: bytes, filename: str, sheet_name=0) -> pd.DataFrame:
    """อ่านไฟล์ CSV/Excel หนึ่งไฟล์ ลองหลายรหัสอักขระสำหรับ CSV ภาษาไทย"""
    if _is_excel(filename):
        return pd.read_excel(io.BytesIO(data), sheet_name=sheet_name)
    last = None
    for enc in _ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except UnicodeDecodeError as exc:  # ลองรหัสถัดไป
            last = exc
    raise DataFormatError(f"อ่านไฟล์ {filename} ไม่ได้ (รหัสอักขระไม่รองรับ): {last}")


def _check(df: pd.DataFrame, required, label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataFormatError(
            f"ไฟล์{label}ขาดคอลัมน์: {', '.join(missing)} "
            f"(คอลัมน์ที่พบ: {', '.join(map(str, df.columns[:12]))})")


def load_rs(data: bytes, filename: str) -> pd.DataFrame:
    df = read_table(data, filename)
    _check(df, RS_REQUIRED, "ผู้ป่วยนอก (RS)")
    return df


def load_pm(data: bytes, filename: str) -> pd.DataFrame:
    df = read_table(data, filename)
    _check(df, PM_REQUIRED, "PM2.5")
    return df


def load_weather(data: bytes, filename: str,
                 station_data: Optional[Tuple[bytes, str]] = None
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """คืน (weather_raw, station_raw)

    - Excel หลายชีต: ชีตที่มีคอลัมน์ Temp./Humi. ถือเป็นข้อมูลอากาศ รวมกันทุกชีต
      ชีตที่มีคอลัมน์ ID และ Province ถือเป็นตารางสถานี (เช่นชีต '2569')
    - CSV: ต้องเป็นข้อมูลอากาศ และต้องแนบตารางสถานี (CSV/Excel) แยก
      หรือมีคอลัมน์ Province อยู่ในไฟล์เดียวกัน
    """
    station = None
    if _is_excel(filename):
        sheets: Dict[str, pd.DataFrame] = pd.read_excel(io.BytesIO(data), sheet_name=None)
        frames = []
        for name, df in sheets.items():
            cols = set(map(str, df.columns))
            if {"Temp.", "Humi."} <= cols:
                frames.append(df)
            elif {"ID", "Province"} <= cols and station is None:
                station = df
        if not frames:
            raise DataFormatError("ไฟล์สภาพอากาศไม่พบชีตที่มีคอลัมน์ Temp. และ Humi.")
        weather = pd.concat(frames, ignore_index=True)
    else:
        weather = read_table(data, filename)
        if "Province" in weather.columns and "ID" not in weather.columns:
            weather = weather.copy()
            weather["ID"] = weather["Province"]
            station = (weather[["ID", "Province"]].drop_duplicates())

    if station is None and station_data is not None:
        station = read_table(*station_data)
    if station is None:
        raise DataFormatError(
            "ไม่พบตารางสถานี (คอลัมน์ ID และ Province) กรุณาแนบไฟล์สถานีเพิ่ม "
            "หรือใช้ไฟล์ Excel ที่มีชีตสถานีอยู่ด้วย")
    _check(weather, WEATHER_REQUIRED, "สภาพอากาศ")
    _check(station, STATION_REQUIRED, "ตารางสถานี")
    return weather, station


def load_monthly_csv(data: bytes, filename: str) -> pd.DataFrame:
    from .constants import MONTHLY_REQUIRED
    df = read_table(data, filename)
    _check(df, MONTHLY_REQUIRED, "รายเดือน")
    return df
