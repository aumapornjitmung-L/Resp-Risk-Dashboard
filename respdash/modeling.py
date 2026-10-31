"""สร้างและประเมินแบบจำลอง 4 แบบ ตามโน้ตบุ๊กวิทยานิพนธ์ (แบ่ง 80/20 แบบ stratified, random_state=42)"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                             precision_score, recall_score, roc_auc_score, roc_curve, auc as auc_fn)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .constants import (CLASS_NAMES, FEATURE_COLS, MODEL_ORDER, RANDOM_STATE, SELECTED_FEATURES)

LABELS = [0, 1, 2, 3]


@dataclass
class ModelBundle:
    feature_names: List[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler
    models: Dict[str, object]
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]]
    metrics: pd.DataFrame
    per_class: pd.DataFrame
    confusion: Dict[str, np.ndarray]
    importance: pd.Series
    selected_result: Dict[str, float]
    split_counts: Dict[str, object]
    present_classes: List[int] = field(default_factory=list)


def build_model_frame(monthly: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """สร้าง X, y เหมือนโน้ตบุ๊ก: ตัวแปรตัวเลข 10 + dummy ฤดู + dummy ภูมิภาค (เรียงตามตัวอักษร)"""
    season_dum = pd.get_dummies(monthly["Season"], prefix="Season").astype(int)
    region_dum = pd.get_dummies(monthly["Region"], prefix="Region").astype(int)
    df = pd.concat([monthly[FEATURE_COLS + ["Risk_Level", "Risk_Level_code"]],
                    season_dum, region_dum], axis=1)
    df = df.fillna(df.mean(numeric_only=True))
    cols = FEATURE_COLS + list(season_dum.columns) + list(region_dum.columns)
    return df[cols], df["Risk_Level_code"], cols


def _evaluate(y_true, y_pred, y_proba, classes) -> Dict[str, float]:
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=classes)
    except ValueError:
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (Macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (Macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1-Score (Macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "AUC (Macro OVR)": auc,
    }


def _make_models() -> Dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced",
                                                  random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                   class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=None,
                                                class_weight="balanced",
                                                random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                 objective="multi:softprob", num_class=4, eval_metric="mlogloss",
                                 random_state=RANDOM_STATE, n_jobs=-1),
    }


def train_all(monthly: pd.DataFrame, progress=None) -> ModelBundle:
    X, y, cols = build_model_frame(monthly)
    counts = y.value_counts()
    if set(counts.index) != set(LABELS) or counts.min() < 2:
        raise ValueError(
            "ข้อมูลต้องมีครบทั้ง 4 ระดับความเสี่ยง (Low/Moderate/High/Extremely) "
            "และแต่ละระดับมีอย่างน้อย 2 ระเบียน จึงจะแบ่งชุดฝึก/ทดสอบแบบ stratified ได้ "
            f"(พบ: {counts.sort_index().to_dict()})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)

    classes = sorted(y.unique().tolist())
    models = _make_models()
    preds: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    n = len(models)
    for i, (name, m) in enumerate(models.items()):
        if progress:
            progress(f"ฝึก {name}", 0.1 + 0.7 * i / n)
        if name in ("Logistic Regression", "SVM"):
            m.fit(Xtr_s, y_train)
            preds[name] = (m.predict(Xte_s), m.predict_proba(Xte_s))
        elif name == "XGBoost":
            sw = compute_sample_weight(class_weight="balanced", y=y_train)
            m.fit(X_train, y_train, sample_weight=sw)
            preds[name] = (m.predict(X_test), m.predict_proba(X_test))
        else:
            m.fit(X_train, y_train)
            preds[name] = (m.predict(X_test), m.predict_proba(X_test))

    rows = []
    for name in MODEL_ORDER:
        pred, proba = preds[name]
        rows.append({"Model": name, **_evaluate(y_test, pred, proba, classes)})
    metrics = pd.DataFrame(rows).set_index("Model")

    # ผลรายคลาส (Precision/Recall/F1/AUC OVR) เหมือนตารางสไลด์ผลรายคลาส
    y_bin = label_binarize(y_test, classes=LABELS)
    per_rows = []
    for ci, cname in enumerate(CLASS_NAMES):
        support = int((np.asarray(y_test) == ci).sum())
        for name in MODEL_ORDER:
            pred, proba = preds[name]
            p, r, f, _ = precision_recall_fscore_support(y_test, pred, labels=LABELS, zero_division=0)
            try:
                a = roc_auc_score(y_bin[:, ci], np.asarray(proba)[:, ci])
            except ValueError:
                a = float("nan")
            per_rows.append({"Class": cname, "Support": support, "Model": name,
                             "Precision": p[ci], "Recall": r[ci], "F1-Score": f[ci], "AUC (OVR)": a})
    per_class = pd.DataFrame(per_rows)

    confusion = {name: confusion_matrix(y_test, preds[name][0], labels=LABELS) for name in MODEL_ORDER}
    importance = pd.Series(models["Random Forest"].feature_importances_, index=cols
                           ).sort_values(ascending=False)

    # Random Forest เมื่อใช้ 12 ตัวแปรที่คัดเลือก (สไลด์ 19 → 12)
    sel = [c for c in SELECTED_FEATURES if c in X.columns]
    Xs_train, Xs_test = X_train[sel], X_test[sel]
    rf_sel = RandomForestClassifier(n_estimators=300, max_depth=None, class_weight="balanced",
                                    random_state=RANDOM_STATE, n_jobs=-1).fit(Xs_train, y_train)
    selected = _evaluate(y_test, rf_sel.predict(Xs_test), rf_sel.predict_proba(Xs_test), classes)
    selected["n_features"] = len(sel)

    split = {
        "train": int(len(X_train)), "test": int(len(X_test)),
        "test_counts": {CLASS_NAMES[c]: int((y_test == c).sum()) for c in LABELS},
        "all_counts": {CLASS_NAMES[c]: int((y == c).sum()) for c in LABELS},
    }
    if progress:
        progress("เสร็จสิ้น", 1.0)
    return ModelBundle(cols, X_train, X_test, y_train, y_test, scaler, models, preds, metrics,
                       per_class, confusion, importance, selected, split, classes)


def roc_curves(bundle: ModelBundle, model_name: str):
    """คืน dict class -> (fpr, tpr, auc) แบบ One-vs-Rest"""
    _, proba = bundle.predictions[model_name]
    y_bin = label_binarize(bundle.y_test, classes=LABELS)
    out = {}
    for i, cname in enumerate(CLASS_NAMES):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        out[cname] = (fpr, tpr, auc_fn(fpr, tpr))
    return out


def time_models(bundle: ModelBundle, repeats: int = 5) -> pd.DataFrame:
    """วัดเวลาฝึกและทำนายด้วยข้อมูลชุดเดียวกับตอนฝึกจริง (LR/SVM ใช้ข้อมูลที่ปรับมาตรฐานแล้ว)"""
    from sklearn.base import clone
    Xtr_s = bundle.scaler.transform(bundle.X_train)
    Xte_s = bundle.scaler.transform(bundle.X_test)
    sw = compute_sample_weight(class_weight="balanced", y=bundle.y_train)
    rows = []
    for name in MODEL_ORDER:
        scaled = name in ("Logistic Regression", "SVM")
        Xa, Xb = (Xtr_s, Xte_s) if scaled else (bundle.X_train, bundle.X_test)
        tr, pr = [], []
        for _ in range(repeats):
            m = clone(bundle.models[name])
            t0 = time.perf_counter()
            m.fit(Xa, bundle.y_train, **({"sample_weight": sw} if name == "XGBoost" else {}))
            tr.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            m.predict(Xb)
            pr.append(time.perf_counter() - t0)
        rows.append({"Model": name, "ฝึกสอน (วินาที)": np.mean(tr), "ส่วนเบี่ยงเบนฝึก": np.std(tr),
                     f"ทำนาย {len(bundle.X_test)} ระเบียน (วินาที)": np.mean(pr)})
    return pd.DataFrame(rows).set_index("Model")


def predict_rows(bundle: ModelBundle, frame: pd.DataFrame, model_name: str = "Random Forest") -> pd.DataFrame:
    """ทำนายระดับความเสี่ยงจากตารางที่มีคอลัมน์ตรงกับ feature_names"""
    X = frame.reindex(columns=bundle.feature_names)
    X = X.fillna(bundle.X_train.mean(numeric_only=True))
    m = bundle.models[model_name]
    Xin = bundle.scaler.transform(X) if model_name in ("Logistic Regression", "SVM") else X
    pred = m.predict(Xin)
    proba = m.predict_proba(Xin)
    out = pd.DataFrame(proba, columns=[f"P({c})" for c in CLASS_NAMES], index=frame.index)
    out.insert(0, "ระดับความเสี่ยงที่ทำนาย", [CLASS_NAMES[int(p)] for p in pred])
    return out
