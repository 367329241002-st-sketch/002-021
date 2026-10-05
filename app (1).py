# -*- coding: utf-8 -*-
# =====================================================================
# app.py : เว็บแอป Streamlit สำหรับทำนาย "Risk Rating" ด้วยโมเดลจาก Orange
# หมายเหตุ: ไฟล์ *.pkcls เป็นโมเดลของ Orange Data Mining จึงต้องติดตั้ง Orange3
#          (ใน requirements.txt) เพื่อให้ joblib โหลดโมเดลได้
# =====================================================================

# ---------- 1) นำเข้าไลบรารีที่ต้องใช้ ----------
from pathlib import Path          # ใช้จัดการพาธไฟล์/โฟลเดอร์

import joblib                     # ใช้โหลดไฟล์ *.pkcls
import numpy as np                # ใช้สร้างอาร์เรย์ข้อมูลที่ผู้ใช้กรอก
import pandas as pd               # ใช้แสดงตารางสรุปผล
import streamlit as st            # ไลบรารีสร้างเว็บแอป
from Orange.classification import Model as OrangeModel   # ใช้ขอค่าความน่าจะเป็น
from Orange.data import Domain, Table                     # ใช้สร้างตารางข้อมูลแบบ Orange

# ---------- 2) ตั้งค่าหน้าเว็บ ----------
st.set_page_config(page_title="AI วิเคราะห์ความสามารถบริหารรายรับรายจ่าย", page_icon="💰")

# ---------- 3) กำหนดตำแหน่งโฟลเดอร์ที่เก็บโมเดล ----------
# แอปจะค้นหาไฟล์ *.pkcls ในโฟลเดอร์ "Model" ก่อน ถ้าไม่เจอจะหาในโฟลเดอร์เดียวกับ app.py
BASE_DIR = Path(__file__).parent
MODEL_DIRS = [BASE_DIR / "Model", BASE_DIR]

# ชื่อที่จะแสดงใน UI (ชื่อไฟล์ -> ชื่อที่อ่านง่าย) ถ้าไม่มีในนี้จะใช้ชื่อไฟล์แทน
MODEL_LABELS = {
    "random_FR.pkcls": "Random Forest",
    "tree.pkcls": "Decision Tree",
    "logisticRe.pkcls": "Logistic Regression",
}

# ชื่อคอลัมน์ -> ป้ายกำกับภาษาไทยที่แสดงใน UI
FEATURE_LABELS = {
    "Age": "อายุ (ปี)",
    "Gender": "เพศ",
    "Education Level": "ระดับการศึกษา",
    "Marital Status": "สถานภาพสมรส",
    "Income": "รายได้",
    "Credit Score": "คะแนนเครดิต",
    "Loan Amount": "จำนวนเงินกู้",
    "Loan Purpose": "วัตถุประสงค์การกู้",
    "Employment Status": "สถานะการทำงาน",
    "Years at Current Job": "อายุงานในที่ทำงานปัจจุบัน (ปี)",
    "Payment History": "ประวัติการชำระเงิน",
    "Debt-to-Income Ratio": "สัดส่วนหนี้ต่อรายได้ (0-1)",
    "Assets Value": "มูลค่าทรัพย์สิน",
    "Number of Dependents": "จำนวนผู้อยู่ในอุปการะ",
    "Previous Defaults": "จำนวนครั้งที่เคยผิดนัดชำระ",
    "Marital Status Change": "การเปลี่ยนสถานภาพสมรส (ครั้ง)",
}

# ค่าเริ่มต้น / ค่าต่ำสุด / ค่าสูงสุด ของคอลัมน์ตัวเลข (ปรับให้ตรงกับข้อมูลของคุณได้)
# รูปแบบ: "ชื่อคอลัมน์": (ค่าเริ่มต้น, ต่ำสุด, สูงสุด)
NUMERIC_CONFIG = {
    "Age": (35, 18, 100),
    "Income": (60000, 0, 10_000_000),
    "Credit Score": (650, 300, 850),
    "Loan Amount": (20000, 0, 10_000_000),
    "Years at Current Job": (5, 0, 60),
    "Debt-to-Income Ratio": (0.30, 0.0, 5.0),
    "Assets Value": (100000, 0, 100_000_000),
    "Number of Dependents": (1, 0, 20),
    "Previous Defaults": (0, 0, 20),
    "Marital Status Change": (0, 0, 10),
}

# คอลัมน์ที่ควรรับค่าเป็นจำนวนเต็ม (นอกนั้นจะรับเป็นทศนิยม)
INTEGER_COLUMNS = {
    "Age", "Income", "Credit Score", "Loan Amount", "Years at Current Job",
    "Assets Value", "Number of Dependents", "Previous Defaults", "Marital Status Change",
}


# ---------- 4) ฟังก์ชันค้นหาและโหลดโมเดล ----------
def find_model_files():
    """ค้นหาไฟล์ *.pkcls ทั้งหมด คืนค่าเป็น dict {ชื่อไฟล์: พาธ}"""
    found = {}
    for folder in MODEL_DIRS:
        if folder.exists():
            for p in sorted(folder.glob("*.pkcls")):
                found.setdefault(p.name, p)
    return found


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model(path_str):
    """โหลดโมเดลด้วย joblib (cache ไว้ จะโหลดแค่ครั้งเดียวต่อไฟล์)"""
    return joblib.load(path_str)


# ---------- 5) หัวข้อแอป ----------
st.title("AI วิเคราะห์ความสามารถบริหารรายรับรายจ่าย")
st.write("กรอกข้อมูลของผู้ขอสินเชื่อด้านล่าง แล้วกดปุ่ม **ทำนายผล** เพื่อประเมินระดับความเสี่ยง (Risk Rating)")

# ---------- 6) UI เลือกโมเดล ----------
model_files = find_model_files()
if not model_files:
    st.error("ไม่พบไฟล์ *.pkcls กรุณาวางไฟล์โมเดลไว้ในโฟลเดอร์ `Model` (หรือโฟลเดอร์เดียวกับ app.py)")
    st.stop()

chosen_file = st.selectbox(
    "เลือกโมเดล",
    options=list(model_files.keys()),
    format_func=lambda name: f"{MODEL_LABELS.get(name, name)}  ({name})",
)
model = load_model(str(model_files[chosen_file]))

# ---------- 7) ดึงรายชื่อ features จากโมเดล ----------
# โมเดล Orange เก็บโครงสร้างข้อมูลดั้งเดิม (ก่อน one-hot) ไว้ใน original_domain
# ดังนั้นเราป้อนค่าดิบ (เช่น Gender = "Male") แล้ว Orange จะ one-hot encode ให้เองแบบเดียวกับตอนฝึก
orig_domain = getattr(model, "original_domain", None)
if orig_domain is None:
    orig_domain = model.domain
features = list(orig_domain.attributes)       # ตัวแปรต้นทุกตัว
class_var = orig_domain.class_var             # ตัวแปรตาม (Risk Rating)

# ---------- 8) สร้างฟอร์มกรอกข้อมูล ----------
st.subheader("ข้อมูลผู้ขอสินเชื่อ")
user_inputs = {}        # เก็บค่าที่ผู้ใช้กรอก {ชื่อคอลัมน์: ค่า}
cols = st.columns(2)    # แบ่งเป็น 2 คอลัมน์ให้ดูกะทัดรัด

for i, var in enumerate(features):
    label = FEATURE_LABELS.get(var.name, var.name)
    with cols[i % 2]:
        if var.is_discrete:
            # คอลัมน์ข้อความ/หมวดหมู่ -> ใช้ selectbox โดยตัวเลือกมาจากโมเดลโดยตรง
            user_inputs[var.name] = st.selectbox(label, options=list(var.values), key=var.name)
        else:
            # คอลัมน์ตัวเลข -> ใช้ number_input
            default, vmin, vmax = NUMERIC_CONFIG.get(var.name, (0.0, None, None))
            if var.name in INTEGER_COLUMNS:
                user_inputs[var.name] = st.number_input(
                    label, min_value=int(vmin), max_value=int(vmax),
                    value=int(default), step=1, key=var.name)
            else:
                user_inputs[var.name] = st.number_input(
                    label, min_value=float(vmin) if vmin is not None else None,
                    max_value=float(vmax) if vmax is not None else None,
                    value=float(default), step=0.01, format="%.2f", key=var.name)

# ---------- 9) ปุ่มทำนายผล ----------
if st.button("ทำนายผล", type="primary"):
    # 9.1) แปลงค่าที่กรอกให้เป็นตัวเลขตามที่ Orange ต้องการ
    #      - คอลัมน์หมวดหมู่: แปลงข้อความเป็นลำดับ index ตาม var.values
    #      - คอลัมน์ตัวเลข: แปลงเป็น float
    row = []
    for var in features:
        value = user_inputs[var.name]
        row.append(var.to_val(value) if var.is_discrete else float(value))

    # 9.2) สร้างตารางข้อมูล 1 แถว (โดเมนไม่มีคอลัมน์คลาส เพราะเรายังไม่รู้คำตอบ)
    input_table = Table.from_numpy(Domain(features), np.array([row], dtype=float))

    # 9.3) ส่งเข้าโมเดล -> Orange จะ one-hot encode / จัดรูปแบบให้ตรงกับตอนฝึกให้อัตโนมัติ
    pred_index = int(model(input_table)[0])               # ลำดับของคลาสที่ทำนายได้
    probs = model(input_table, OrangeModel.Probs)[0]      # ความน่าจะเป็นของแต่ละคลาส
    pred_label = class_var.values[pred_index]             # ชื่อคลาสที่ทำนายได้

    # ---------- 10) แสดงผลการทำนาย ----------
    RISK_TH = {"Low": "ต่ำ", "Medium": "ปานกลาง", "High": "สูง"}   # แปลเป็นภาษาไทย
    th = RISK_TH.get(pred_label, pred_label)
    message = f"ผลการทำนาย: ความเสี่ยง **{th}** ({pred_label})"
    if pred_label == "Low":
        st.success(message)
    elif pred_label == "Medium":
        st.warning(message)
    else:
        st.error(message)

    # แสดงความน่าจะเป็นของแต่ละระดับความเสี่ยง
    prob_df = pd.DataFrame({
        "ระดับความเสี่ยง": [f"{RISK_TH.get(v, v)} ({v})" for v in class_var.values],
        "ความน่าจะเป็น (%)": [round(float(p) * 100, 2) for p in probs],
    })
    st.write("ความน่าจะเป็นของแต่ละระดับ:")
    st.dataframe(prob_df, hide_index=True)
    st.bar_chart(prob_df.set_index("ระดับความเสี่ยง"))
