import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# --- Page Configuration ---
st.set_page_config(
    page_title="Mouse Viral Infection Dashboard",
    page_icon="🦠",
    layout="wide"
)

# --- Sidebar Navigation ---
st.sidebar.title("🦠 Mouse Study Dashboard")
page = st.sidebar.radio(
    "ไปที่:",
    ["Overview", "Data Explorer", "Analytics", "Prediction", "About"]
)

# --- Data Loading with Caching ---
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data("./data/mouse.csv")

# --- Overview Page ---
if page == "Overview":
    st.title("ภาพรวมการศึกษาการติดเชื้อไวรัสในหนูทดลอง")
    st.markdown("สรุปข้อมูลเบื้องต้นของชุดข้อมูล mouse.csv สำหรับการวิเคราะห์ต่อไป")

    # Metrics
    total_samples = df.shape[0]
    total_features = df.shape[1]
    missing_values = int(df.isnull().sum().sum())
    col1, col2, col3 = st.columns(3)
    col1.metric("จำนวนตัวอย่าง (rows)", total_samples)
    col2.metric("จำนวนคุณลักษณะ (columns)", total_features)
    col3.metric("ค่าที่หายไป (total missing)", missing_values)

    # Feature distributions
    st.subheader("การแจกแจงของแต่ละคุณลักษณะเชิงตัวเลข")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(4,2))
        ax.hist(df[col].dropna(), bins=20)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

# --- Data Explorer Page ---
elif page == "Data Explorer":
    st.title("Data Explorer: กรองและดาวน์โหลดข้อมูล")
    st.markdown("ใช้ฟิลเตอร์ด้านล่างเพื่อเลือกข้อมูลที่ต้องการ และดาวน์โหลดไฟล์ CSV ได้ทันที")
    with st.expander("▶️ ตั้งค่าการกรอง"):
        selected_cols = st.multiselect("เลือกคอลัมน์เพื่อแสดง", df.columns.tolist(), default=df.columns.tolist())
        filter_col = st.selectbox("เลือกคอลัมน์สำหรับกรอง (numeric):", num_cols)
        min_val, max_val = st.select_slider(
            "ช่วงค่าของ {}:".format(filter_col),
            options=sorted(df[filter_col].dropna().unique()),
            value=(df[filter_col].min(), df[filter_col].max())
        )

    # Apply filters
    filtered_df = df[selected_cols]
    mask = (df[filter_col] >= min_val) & (df[filter_col] <= max_val)
    filtered_df = filtered_df[mask]

    st.dataframe(filtered_df, use_container_width=True)
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 ดาวน์โหลด CSV ที่กรองแล้ว",
        data=csv,
        file_name='filtered_mouse_data.csv',
        mime='text/csv'
    )

# --- Analytics Page ---
elif page == "Analytics":
    st.title("Analytics: สหสัมพันธ์และการแสดงผลเชิงกราฟ")
    st.markdown("ดูความสัมพันธ์ระหว่างคุณลักษณะและกราฟเชิงลึกต่าง ๆ")

    # Correlation heatmap
    st.subheader("Heatmap ของ Correlation")
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(6,5))
    cax = ax.matshow(corr, vmin=-1, vmax=1)
    plt.xticks(range(len(num_cols)), num_cols, rotation=90)
    plt.yticks(range(len(num_cols)), num_cols)
    fig.colorbar(cax)
    st.pyplot(fig)

    # Pair Scatter
    st.subheader("Scatter Plot ระหว่างคุณลักษณะสองตัว")
    x_axis = st.selectbox("X-axis:", num_cols, index=0)
    y_axis = st.selectbox("Y-axis:", num_cols, index=1)
    fig2, ax2 = plt.subplots()
    ax2.scatter(df[x_axis], df[y_axis], alpha=0.6)
    ax2.set_xlabel(x_axis)
    ax2.set_ylabel(y_axis)
    st.pyplot(fig2)

# --- Prediction Page ---
elif page == "Prediction":
    st.title("Prediction: ทำนายประเภทดอกไม้จาก Iris Dataset")
    st.markdown("ใช้ KNN ในการทำนายประเภทดอกไม้ จากชุดข้อมูล iris-3.csv")

    # Load Iris data
    iris = load_data("./data/iris-3.csv")
    X = iris.drop('variety', axis=1)
    y = iris['variety']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train Model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation Metrics
    st.subheader("ประสิทธิภาพของโมเดล")
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(cm)
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # User Input for Prediction
    st.subheader("ทำนายข้อมูลใหม่")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pt_len = st.slider("petal length", float(df['petallength'].min()), float(df['petallength'].max()), float(df['petallength'].median()))
    with col2:
        pt_wd  = st.slider("petal width",  float(df['petalwidth'].min()),  float(df['petalwidth'].max()),  float(df['petalwidth'].median()))
    with col3:
        sp_len = st.number_input("sepal length", float(df['sepallength'].min()), float(df['sepallength'].max()), float(df['sepallength'].median()))
    with col4:
        sp_wd  = st.number_input("sepal width",  float(df['sepalwidth'].min()),  float(df['sepalwidth'].max()),  float(df['sepalwidth'].median()))

    if st.button("🔍 ทำนายผล"):    
        input_arr = np.array([[pt_len, pt_wd, sp_len, sp_wd]])
        result = model.predict(input_arr)[0]
        prob = model.predict_proba(input_arr).max()
        st.success(f"ผลการทำนาย: {result} (ความมั่นใจ {prob:.2%})")
        # Show image
        img_map = {'Setosa':'./img/iris1.jpg','Versicolor':'./img/iris2.jpg','Virginica':'./img/iris3.jpg'}
        st.image(img_map.get(result, './img/iris1.jpg'), use_column_width=True)

# --- About Page ---
else:
    st.title("About This App")
    st.markdown(
        """
        **Mouse Viral Infection Dashboard**
        
        - พัฒนาโดย: Your Name
        - ชุดข้อมูล: mouse.csv, iris-3.csv
        - เทคโนโลยี: Python, Streamlit, scikit-learn, pandas, matplotlib
        - เวอร์ชัน: 1.1.0
        """
    )
    st.markdown("---")
st.sidebar.markdown("\n---\nถูกพัฒนาโดย 💙 Your Name")