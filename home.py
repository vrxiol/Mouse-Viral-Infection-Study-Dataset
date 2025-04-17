import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Mouse Viral Infection Study",
    page_icon="🦠",
    layout="wide"
)

# --- Sidebar Navigation ---
st.sidebar.title("เมนูหลัก")
page = st.sidebar.radio(
    "เลือกหน้า:",
    ["Overview", "Expression Trends", "PCA & Clustering", "Infection Prediction", "Download Data", "About"]
)

# --- Load Data with Caching ---
@st.cache_data

def load_mouse_data(path):
    df = pd.read_csv(path)
    if 'Virus Present' in df.columns:
        df.rename(columns={'Virus Present': 'infected'}, inplace=True)
    return df

df = load_mouse_data("./data/mouse.csv")

# Identify numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'infected' in df.columns:
    df['infected'] = df['infected'].astype(int)

# --- Overview Page ---
st.image('./img/1.jpg')
if page == "Overview":
    st.title("ภาพรวมชุดข้อมูล Mouse Viral Infection")
    st.markdown(
        f"- จำนวนตัวอย่าง: **{df.shape[0]}** แถว  \n"
        f"- จำนวนคอลัมน์: **{df.shape[1]}**"
    )
    st.subheader("ตัวอย่างข้อมูล")
    st.dataframe(df.head(5), use_container_width=True)
    st.subheader("สถิติพื้นฐานของตัวแปร์เลขตัวเลข")
    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    st.subheader("Missing Values ต่อคอลัมน์")
    st.bar_chart(df.isnull().sum())

# --- Expression Trends Page ---
elif page == "Expression Trends":
    st.title("แนวโน้มการแสดงออกของยีน")
    selected_cols = st.multiselect(
        "เลือกตัวแปร์เลขตัวเลข:", numeric_cols, default=numeric_cols[:3]
    )
    if selected_cols:
        fig = px.line(df[selected_cols], title="Gene Expression Trends")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("โปรดเลือกตัวแปร์เลขอย่างน้อยหนึ่งตัว")

# --- PCA & Clustering Page ---
elif page == "PCA & Clustering":
    st.title("PCA & K-Means Clustering")
    data_numeric = df[numeric_cols].fillna(0)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(data_numeric)
    pca_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
    n_clusters = st.slider("จำนวนคลัสเตอร์:", 2, 8, 4)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    pca_df['Cluster'] = kmeans.fit_predict(data_numeric).astype(str)
    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', title="PCA - KMeans Clustering")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"Explained Variance: PC1={pca.explained_variance_ratio_[0]:.2f}, PC2={pca.explained_variance_ratio_[1]:.2f}")

# --- Infection Prediction Page ---
elif page == "Infection Prediction":
    st.title("ทำนายการติดเชื้อไวรัส")
    if 'infected' not in df.columns:
        st.error("ไม่มีคอลัมน์ 'infected' ในชุดข้อมูล")
    else:
        X = df[numeric_cols].drop(columns=['infected'], errors='ignore').fillna(0)
        y = df['infected']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model_name = st.selectbox("เลือกโมเดล:", ["Logistic Regression", "Random Forest"])
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        else:
            model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.subheader("Performance Metrics")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        st.subheader("ทำนายตัวอย่างใหม่")
        user_inputs = {}
        cols_container = st.container()
        for col in X.columns:
            user_inputs[col] = cols_container.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].median()))
        if st.button("🔍 ทำนาย"):
            input_df = pd.DataFrame([user_inputs])
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]
            label = 'ติดเชื้อ' if pred == 1 else 'ไม่ติดเชื้อ'
            st.success(f"ผลการทำนาย: **{label}** (ความน่าจะเป็น: {proba:.2%})")

# --- Download Page ---
elif page == "Download Data":
    st.title("ดาวน์โหลดชุดข้อมูล")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📅 ดาวน์โหลด CSV",
        data=csv,
        file_name='mouse_viral_infection.csv',
        mime='text/csv'
    )

# --- About Page ---
else:
    st.title("เกี่ยวกับแอป")
    st.markdown("""
        **Mouse Viral Infection Study App**  
        - พัฒนาโดย: อนุสรณ์ เถาะปีนาม  
        - ชุดข้อมูล: mouse.csv  
        - เทคโนโลยี: Streamlit, pandas, numpy, scikit-learn, matplotlib, plotly  
        - เวอร์ชัน: 1.4.0
    """)

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.write("Developed with 💙 อนุสรณ์ เถาะปีนาม")


