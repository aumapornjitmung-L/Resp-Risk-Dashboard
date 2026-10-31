"""ค่าคงที่ที่ใช้ร่วมกัน ตรงกับโน้ตบุ๊กวิทยานิพนธ์ (NewRiskLevel_EN)"""

RANDOM_STATE = 42

NORTH = ["เชียงราย", "แม่ฮ่องสอน", "เชียงใหม่", "ลำพูน", "ลำปาง", "พะเยา", "น่าน", "แพร่",
         "อุตรดิตถ์", "สุโขทัย", "ตาก", "กำแพงเพชร", "พิษณุโลก", "พิจิตร", "เพชรบูรณ์"]
NORTHEAST = ["หนองคาย", "เลย", "หนองบัวลำภู", "อุดรธานี", "สกลนคร", "นครพนม", "มุกดาหาร",
             "กาฬสินธุ์", "ขอนแก่น", "มหาสารคาม", "ร้อยเอ็ด", "ยโสธร", "อำนาจเจริญ", "ชัยภูมิ",
             "นครราชสีมา", "บุรีรัมย์", "สุรินทร์", "ศรีสะเกษ", "อุบลราชธานี", "บึงกาฬ"]
CENTRAL = ["นครสวรรค์", "อุทัยธานี", "ชัยนาท", "สิงห์บุรี", "ลพบุรี", "อ่างทอง", "สระบุรี",
           "สุพรรณบุรี", "พระนครศรีอยุธยา", "กาญจนบุรี", "ราชบุรี", "นครปฐม", "นนทบุรี",
           "ปทุมธานี", "กรุงเทพมหานคร", "สมุทรปราการ", "สมุทรสงคราม", "สมุทรสาคร"]
EAST = ["นครนายก", "ฉะเชิงเทรา", "ปราจีนบุรี", "สระแก้ว", "ชลบุรี", "ระยอง", "จันทบุรี", "ตราด"]
SOUTH_EAST = ["เพชรบุรี", "ประจวบคีรีขันธ์", "ชุมพร", "สุราษฎร์ธานี", "นครศรีธรรมราช",
              "พัทลุง", "สงขลา", "ปัตตานี", "ยะลา", "นราธิวาส"]
SOUTH_WEST = ["ระนอง", "พังงา", "ภูเก็ต", "กระบี่", "ตรัง", "สตูล"]

REGION_MAP = {}
for _p in NORTH:
    REGION_MAP[_p] = "North"
for _p in NORTHEAST:
    REGION_MAP[_p] = "Northeast"
for _p in CENTRAL:
    REGION_MAP[_p] = "Central"
for _p in EAST:
    REGION_MAP[_p] = "East"
for _p in SOUTH_EAST:
    REGION_MAP[_p] = "South-East-Coast"
for _p in SOUTH_WEST:
    REGION_MAP[_p] = "South-West-Coast"

REGION_TH = {
    "North": "เหนือ", "Northeast": "อีสาน", "Central": "กลาง",
    "East": "ตะวันออก", "South-East-Coast": "ใต้ฝั่งตะวันออก",
    "South-West-Coast": "ใต้ฝั่งตะวันตก",
}
REGION_ORDER = ["North", "Northeast", "Central", "East", "South-East-Coast", "South-West-Coast"]

SEASON_TH = {"Summer": "ฤดูร้อน", "Rainy": "ฤดูฝน", "Winter": "ฤดูหนาว"}

EN_TO_TH = {
    "Amnat Charoen": "อำนาจเจริญ", "Ayutthaya": "พระนครศรีอยุธยา", "Bangkok": "กรุงเทพมหานคร",
    "Bueng Kan": "บึงกาฬ", "Buriram": "บุรีรัมย์", "Chachoengsao": "ฉะเชิงเทรา", "Chai Nat": "ชัยนาท",
    "Chaiyaphum": "ชัยภูมิ", "Chanthaburi": "จันทบุรี", "Chiang Mai": "เชียงใหม่", "Chiang Rai": "เชียงราย",
    "Chon Buri": "ชลบุรี", "Chumphon": "ชุมพร", "Kalasin": "กาฬสินธุ์", "Kampaeng Phet": "กำแพงเพชร",
    "Kanchanaburi": "กาญจนบุรี", "Khon Kaen": "ขอนแก่น", "Krabi": "กระบี่", "Lampang": "ลำปาง",
    "Lamphun": "ลำพูน", "Lob Buri": "ลพบุรี", "Loie": "เลย", "Mae Hong Son": "แม่ฮ่องสอน",
    "Maha Sarakham": "มหาสารคาม", "Mukdahan": "มุกดาหาร", "Nakhon Nayok": "นครนายก",
    "Nakhon Phanom": "นครพนม", "Nakhon Ratchasima": "นครราชสีมา", "Nakhon Sawan": "นครสวรรค์",
    "Nakhonsithammarat": "นครศรีธรรมราช", "Nan": "น่าน", "Narathiwat": "นราธิวาส",
    "Nong Bua Lamphu": "หนองบัวลำภู", "Nong Khai": "หนองคาย", "Pattani": "ปัตตานี", "Phang Nga": "พังงา",
    "Phatthalung": "พัทลุง", "Phayao": "พะเยา", "Phetchabun": "เพชรบูรณ์", "Phetchaburi": "เพชรบุรี",
    "Phichit": "พิจิตร", "Phitsanulok": "พิษณุโลก", "Phrae": "แพร่", "Phuket": "ภูเก็ต",
    "Prachin Buri": "ปราจีนบุรี", "Prachuap Khiri Khan": "ประจวบคีรีขันธ์", "Pratumtani": "ปทุมธานี",
    "Ranong": "ระนอง", "Ratcha Buri": "ราชบุรี", "Rayong": "ระยอง", "Roi Et": "ร้อยเอ็ด", "Sa Kaeo": "สระแก้ว",
    "Sakon Nakhon": "สกลนคร", "Samut Prakan": "สมุทรปราการ", "Samut Songkhram": "สมุทรสงคราม",
    "Satun": "สตูล", "Songkhla": "สงขลา", "Sri Sa Ket": "ศรีสะเกษ", "Sukhothai": "สุโขทัย",
    "Suphanburi": "สุพรรณบุรี", "Surat Thani": "สุราษฎร์ธานี", "Surin": "สุรินทร์", "Tak": "ตาก",
    "Trang": "ตรัง", "Ubon Ratchathani": "อุบลราชธานี", "Udon Thani": "อุดรธานี", "Uthaithani": "อุทัยธานี",
    "Uttaradit": "อุตรดิตถ์", "Yala": "ยะลา", "Yasothon": "ยโสธร",
}
TH_TO_EN = {v: k for k, v in EN_TO_TH.items()}

# ----- ตัวแปรและเป้าหมาย -----
FEATURE_COLS = [
    "PM25_mean", "PM25_max", "Temp_mean", "Humidity_mean", "Pressure_mean",
    "Rainfall_sum", "AQI_mean", "mean_age_month", "pct_age_0_4_month", "pct_age_65plus_month",
]
SELECTED_FEATURES = [
    "Pressure_mean", "pct_age_65plus_month", "pct_age_0_4_month", "mean_age_month",
    "Temp_mean", "PM25_mean", "Humidity_mean", "PM25_max", "Region_North",
    "Rainfall_sum", "AQI_mean", "Region_East",
]
FEATURE_TH = {
    "PM25_mean": "PM2.5 เฉลี่ย", "PM25_max": "PM2.5 สูงสุด", "Temp_mean": "อุณหภูมิเฉลี่ย",
    "Humidity_mean": "ความชื้นเฉลี่ย", "Pressure_mean": "ความกดอากาศเฉลี่ย",
    "Rainfall_sum": "ปริมาณฝนรวม", "AQI_mean": "AQI เฉลี่ย", "mean_age_month": "อายุเฉลี่ยของผู้ป่วย",
    "pct_age_0_4_month": "สัดส่วนผู้ป่วยอายุ 0–4 ปี", "pct_age_65plus_month": "สัดส่วนผู้ป่วยอายุ 65 ปีขึ้นไป",
    "Season_Rainy": "ฤดูฝน", "Season_Summer": "ฤดูร้อน", "Season_Winter": "ฤดูหนาว",
    "Region_North": "ภาคเหนือ", "Region_Northeast": "ภาคตะวันออกเฉียงเหนือ", "Region_Central": "ภาคกลาง",
    "Region_East": "ภาคตะวันออก", "Region_South-East-Coast": "ภาคใต้ฝั่งตะวันออก",
    "Region_South-West-Coast": "ภาคใต้ฝั่งตะวันตก",
}

RISK_LABELS = ["Low Risk (L)", "Moderate Risk (M)", "High Risk (H)", "Extremely Risk (E)"]
RISK_CODE = {name: i for i, name in enumerate(RISK_LABELS)}
CLASS_NAMES = ["Low (L)", "Moderate (M)", "High (H)", "Extremely (E)"]
CLASS_COLORS = ["#3FA796", "#F2C14E", "#E8833A", "#C0392B"]

# เกณฑ์ Risk Level (จำนวนผู้ป่วยรายเดือนต่อจังหวัด) ปรับใช้ค่าตัดจากเกณฑ์ผลกระทบของกรมควบคุมโรค
RISK_THRESHOLDS = {"moderate": 50, "high": 100, "extremely": 150}

MODEL_ORDER = ["Random Forest", "XGBoost", "SVM", "Logistic Regression"]

# ชื่อคอลัมน์ที่ต้องมีในไฟล์รายเดือน (โหมดนำเข้า CSV รายเดือน)
MONTHLY_REQUIRED = [
    "Province", "Region", "YearMonth", "case_count_month", "PM25_mean", "PM25_max",
    "Temp_mean", "Humidity_mean", "Pressure_mean", "Rainfall_sum", "AQI_mean",
    "mean_age_month", "pct_age_0_4_month", "pct_age_65plus_month", "Season",
]
