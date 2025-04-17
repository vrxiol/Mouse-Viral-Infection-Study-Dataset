import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Mouse Viral Infection Study", 
    page_icon="🦠",
    layout="wide"
)

# Sidebar Navigation
st.sidebar.title("เมนูหลัก")
page = st.sidebar.radio(
    "เลือกการทำงาน:",
    ["ดูข้อมูล (Data View)", "สรุปสถิติ (Summary)", "กราฟ (Visualization)"]
)

# โหลดข้อมูล พร้อม caching
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

df = load_data("./data/mouse.csv")

# 1. Data View
if page == "ดูข้อมูล (Data View)":
    st.header("ดูข้อมูลเต็ม (First & Last Rows)")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10")
        st.dataframe(df.head(10), use_container_width=True)
    with col2:
        st.subheader("Bottom 10")
        st.dataframe(df.tail(10), use_container_width=True)

# 2. Summary
elif page == "สรุปสถิติ (Summary)":
    st.header("สรุปสถิติพื้นฐาน")
    st.markdown("- **ขนาดข้อมูล:** {} แถว, {} คอลัมน์".format(df.shape[0], df.shape[1]))
    st.markdown("- **รายการคอลัมน์:** {}".format(list(df.columns)))
    
    st.subheader("สถิติเชิงตัวเลข")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("จำนวนค่าที่หายไป (Missing Values)")
    st.dataframe(df.isnull().sum(), use_container_width=True)

# 3. Visualization
else:
    st.header("การวิเคราะห์ด้วยกราฟ")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    chart_type = st.selectbox("ประเภทกราฟ:", ["Bar Chart", "Histogram", "Scatter Plot"])

    if chart_type == "Bar Chart":
        col = st.selectbox("เลือกคอลัมน์ (Categorical):", cat_cols)
        counts = df[col].value_counts()
        st.bar_chart(counts)

    elif chart_type == "Histogram":
        col = st.selectbox("เลือกคอลัมน์ (Numeric):", numeric_cols)
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=20)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    else:  # Scatter Plot
        x_col = st.selectbox("X-axis:", numeric_cols, index=0)
        y_col = st.selectbox("Y-axis:", numeric_cols, index=1)
        fig, ax = plt.subplots()
        ax.scatter(df[x_col], df[y_col])
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        st.pyplot(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Developed with 💙 by Your Name")