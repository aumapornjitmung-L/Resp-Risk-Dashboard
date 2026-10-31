"""จุดเริ่มต้นของแอป Streamlit: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="แดชบอร์ดพยากรณ์ความเสี่ยงโรคระบบทางเดินหายใจ", page_icon="🫁",
                   layout="wide", initial_sidebar_state="expanded")

from respdash import ui  # noqa: E402
from views import dashboard, explore, features, models, predict, quality  # noqa: E402

ui.inject_css()

pages = [
    st.Page(dashboard.render, title="แดชบอร์ดหลัก", icon="📊", default=True),
    st.Page(models.render, title="ผลเปรียบเทียบแบบจำลอง", icon="🤖", url_path="models"),
    st.Page(features.render, title="ตัวแปรสำคัญ", icon="🔑", url_path="features"),
    st.Page(explore.render, title="สำรวจข้อมูลและความสัมพันธ์", icon="🔍", url_path="explore"),
    st.Page(quality.render, title="คุณภาพข้อมูล", icon="🧹", url_path="quality"),
    st.Page(predict.render, title="ทำนายระดับความเสี่ยง", icon="🎯", url_path="predict"),
]
nav = st.navigation(pages)
with st.sidebar:
    st.caption("วิทยานิพนธ์: การพยากรณ์ความเสี่ยงการเกิดโรคระบบทางเดินหายใจตามปัจจัยด้านสิ่งแวดล้อม "
               "โดยใช้การเรียนรู้ของเครื่อง")
nav.run()
