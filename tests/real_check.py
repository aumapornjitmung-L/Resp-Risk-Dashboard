"""สคริปต์ตรวจกับข้อมูลจริง (รันในเครื่องเท่านั้น ไม่ใช่ส่วนหนึ่งของ CI) ใช้: python tests/real_check.py rs.xlsx pm.xlsx tem.xlsx"""
import sys, time, pickle
sys.path.insert(0, ".")
from respdash.io import load_rs, load_pm, load_weather
from respdash.pipeline import run_pipeline
from respdash.modeling import train_all

def rb(p): return open(p, "rb").read(), p
rs_p, pm_p, tem_p = sys.argv[1:4]
t = time.time()
rs = load_rs(*rb(rs_p)); pm = load_pm(*rb(pm_p)); w, st = load_weather(*rb(tem_p))
print("load", round(time.time() - t, 1))
t = time.time(); prep = run_pipeline(rs, pm, w, st); print("pipeline", round(time.time() - t, 1))
print(prep.quality)
print(prep.monthly["Risk_Level"].value_counts().to_dict(), prep.monthly.shape)
t = time.time(); b = train_all(prep.monthly); print("train", round(time.time() - t, 1))
print(b.metrics.round(4)); print(b.selected_result); print(b.split_counts)
print(b.importance.head(5).round(3))
pickle.dump((prep, b), open("/tmp/real_prep.pkl", "wb"))
