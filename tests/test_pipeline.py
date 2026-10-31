"""ทดสอบด้วยข้อมูลสังเคราะห์ (ไม่ใช้ข้อมูลวิจัยจริง) ให้ CI รันได้โดยไม่ต้องมีไฟล์ข้อมูล"""
import numpy as np
import pandas as pd
import pytest

from respdash.constants import REGION_MAP, TH_TO_EN, CLASS_NAMES
from respdash.io import DataFormatError, load_monthly_csv, load_rs, read_table
from respdash.modeling import build_model_frame, predict_rows, train_all
from respdash.pipeline import (prepare_monthly_csv, risk_level, run_pipeline, season_of)


def synthetic_raw(seed=0, days=420):
    rng = np.random.default_rng(seed)
    provs = sorted(REGION_MAP)
    start = pd.Timestamp("2023-01-01")
    dates = pd.date_range(start, periods=days)
    # ผู้ป่วย: จังหวัดใหญ่ (index เลขคี่) มีผู้ป่วยมาก เพื่อให้เกิดหลายระดับความเสี่ยง
    rows = []
    for i, p in enumerate(provs):
        lam = [0.2, 1.0, 3.0, 6.0][i % 4]
        for d in dates:
            for _ in range(rng.poisson(lam)):
                rows.append({"Diagcode": "J10", "Date_serv": d, "sex": int(rng.integers(1, 3)),
                             "Birth day": d - pd.Timedelta(days=int(rng.integers(0, 365 * 90))),
                             "pid": int(rng.integers(1, 10**9)), "Hospcode": 1.0, "Province name": p})
    rs = pd.DataFrame(rows)
    rs = pd.concat([rs, rs.iloc[:5]], ignore_index=True)           # แถวซ้ำ
    pm_provs = [p for p in provs if p not in ("พังงา", "ระนอง")]
    pm = pd.DataFrame([{"Date": d, "Province ID": f"{j:02d}T", "PM2.5": float(rng.uniform(5, 60)),
                        "Province Name": p} for j, p in enumerate(pm_provs) for d in dates])
    pm.loc[pm.sample(frac=0.03, random_state=1).index, "PM2.5"] = np.nan
    en_provs = [TH_TO_EN[p] for p in provs if p in TH_TO_EN and p not in ("ตราด", "นครปฐม")]
    st = pd.DataFrame({"ID": range(1, len(en_provs) + 1), "Station": "x", "Province": en_provs})
    w = pd.DataFrame([{"ID": i + 1, "Date": d.strftime("%y/%m/%d"), "Wind Speed": 1.0,
                       "Temp.": float(rng.uniform(20, 35)), "Humi.": float(rng.uniform(50, 95)),
                       "heatidx": 30.0, "Pres.": float(rng.uniform(990, 1010)), "Prec.": float(rng.uniform(0, 5))}
                      for i in range(len(en_provs)) for d in dates])
    return rs, pm, w, st


@pytest.fixture(scope="module")
def prep():
    rs, pm, w, st = synthetic_raw()
    return run_pipeline(rs, pm, w, st)


def test_season_boundaries():
    d = pd.to_datetime(["2024-02-14", "2024-02-15", "2024-05-15", "2024-05-16", "2024-10-15", "2024-10-16"])
    assert season_of(d).tolist() == ["Winter", "Summer", "Summer", "Rainy", "Rainy", "Winter"]


def test_risk_thresholds():
    assert [risk_level(n) for n in (0, 49, 50, 100, 101, 150, 151)] == [
        "Low Risk (L)", "Low Risk (L)", "Moderate Risk (M)", "Moderate Risk (M)",
        "High Risk (H)", "High Risk (H)", "Extremely Risk (E)"]


def test_pipeline_quality_flags(prep):
    q = prep.quality
    assert q["rs_duplicates"] >= 5
    assert set(q["pm_provinces_missing"]) == {"พังงา", "ระนอง"}
    assert {"ตราด", "นครปฐม"} <= set(q["weather_no_station"])
    assert q["pm_missing_after"] == 0
    assert q["merged_rows"] == 77 * 420
    assert prep.merged["PM2.5"].notna().all()


def test_monthly_shape_and_columns(prep):
    m = prep.monthly
    assert m["Province"].nunique() == 77
    assert m["Risk_Level_code"].between(0, 3).all()
    # ผลรวมผู้ป่วยรายเดือนต้องเท่ากับผลรวมรายวัน
    assert m["case_count_month"].sum() == prep.merged["case_count"].sum()


def test_model_frame_feature_order(prep):
    X, y, cols = build_model_frame(prep.monthly)
    assert cols[:10] == ["PM25_mean", "PM25_max", "Temp_mean", "Humidity_mean", "Pressure_mean",
                         "Rainfall_sum", "AQI_mean", "mean_age_month", "pct_age_0_4_month", "pct_age_65plus_month"]
    season = [c for c in cols if c.startswith("Season_")]
    region = [c for c in cols if c.startswith("Region_")]
    assert season == sorted(season) and region == sorted(region)
    assert X.shape[1] == len(cols) and not X.isna().any().any()


def test_train_and_predict(prep):
    counts = prep.monthly["Risk_Level_code"].value_counts()
    if counts.min() < 2 or len(counts) < 4:
        pytest.skip("ข้อมูลสังเคราะห์ไม่ครบ 4 ระดับ")
    b = train_all(prep.monthly)
    assert list(b.metrics.index) == ["Random Forest", "XGBoost", "SVM", "Logistic Regression"]
    assert b.metrics["Accuracy"].between(0, 1).all()
    assert all(cm.sum() == len(b.y_test) for cm in b.confusion.values())
    out = predict_rows(b, b.X_test.head(3), "Random Forest")
    assert out["ระดับความเสี่ยงที่ทำนาย"].isin(CLASS_NAMES).all()
    # โมเดลต้องฝึกด้วยข้อมูลเดียวกันซ้ำแล้วได้ผลเท่าเดิม (reproducible)
    b2 = train_all(prep.monthly)
    assert b.metrics.equals(b2.metrics)


def test_monthly_csv_roundtrip(prep, tmp_path):
    csv = prep.monthly.assign(YearMonth=prep.monthly["YearMonth"].astype(str))
    path = tmp_path / "m.csv"
    csv.to_csv(path, index=False, encoding="utf-8-sig")
    df = load_monthly_csv(path.read_bytes(), "m.csv")
    back = prepare_monthly_csv(df)
    assert len(back.monthly) == len(prep.monthly)
    assert (back.monthly["Risk_Level"].values == prep.monthly["Risk_Level"].values).all()


def test_bad_files_give_clear_errors():
    with pytest.raises(DataFormatError, match="ขาดคอลัมน์"):
        load_rs(b"a,b\n1,2\n", "x.csv")
    with pytest.raises(DataFormatError):
        load_monthly_csv("Province\nกรุงเทพมหานคร\n".encode("utf-8"), "m.csv")


def test_thai_csv_encoding_fallback():
    data = "Province,v\nกรุงเทพมหานคร,1\n".encode("cp874")
    assert read_table(data, "t.csv").iloc[0, 0] == "กรุงเทพมหานคร"
