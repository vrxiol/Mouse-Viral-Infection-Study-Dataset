import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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
    ["Overview", "Expression Trends", "PCA & Clustering", "Download Data", "About"]
)

# --- Load Data with Caching ---
@st.cache_data
def load_mouse_data(path):
    return pd.read_csv(path)

df = load_mouse_data("./data/mouse.csv")

# Identify numeric columns automatically
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.tolist()

# --- Overview Page ---
if page == "Overview":
    st.title("ภาพรวมชุดข้อมูล Mouse Viral Infection")
    st.markdown(
        f"- จำนวนตัวอย่าง: **{df.shape[0]}** แถว  \n"
        f"- จำนวนคอลัมน์: **{df.shape[1]}**"
    )

    st.subheader("ตัวอย่างข้อมูล")
    st.dataframe(df.head(5), use_container_width=True)

    st.subheader("สถิติพื้นฐานของตัวแปรเชิงตัวเลข")
    st.dataframe(df_numeric.describe(), use_container_width=True)

    st.subheader("จำนวนค่าที่หายไปในแต่ละคอลัมน์")
    st.bar_chart(df.isnull().sum())

# --- Expression Trends Page ---
elif page == "Expression Trends":
    st.title("การเปลี่ยนแปลงของการแสดงออกของยีน")
    st.markdown("เลือกตัวแปรเชิงตัวเลขเพื่อดูการเปลี่ยนแปลงตามลำดับตัวอย่าง")

    selected_cols = st.multiselect(
        "เลือกคอลัมน์:", numeric_cols, default=numeric_cols[:3]
    )
    if selected_cols:
        trend_df = df_numeric[selected_cols]
        st.line_chart(trend_df)
    else:
        st.write("โปรดเลือกคอลัมน์อย่างน้อยหนึ่งตัว")

# --- PCA & Clustering Page ---
elif page == "PCA & Clustering":
    st.title("PCA และการจัดกลุ่ม (K-Means)")

    # PCA
    st.subheader("PCA: ลดมิติลงเหลือ 2 มิติ")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_numeric.fillna(0))
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])

    # Clustering parameters
    st.subheader("K-Means Clustering")
    n_clusters = st.slider("จำนวนกลุ่ม (clusters)", 2, 10, 4)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df_numeric.fillna(0))
    pca_df['Cluster'] = clusters.astype(str)

    # Plot PCA with clusters
    fig, ax = plt.subplots()
    for cluster in sorted(pca_df['Cluster'].unique()):
        mask = pca_df['Cluster'] == cluster
        ax.scatter(
            pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'],
            label=f"Cluster {cluster}", alpha=0.7
        )
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(title='Cluster')
    st.pyplot(fig)

    # Explained variance
    st.markdown(
        f"**Explained variance ratio:** PC1 = {pca.explained_variance_ratio_[0]:.2f}, "
        f"PC2 = {pca.explained_variance_ratio_[1]:.2f}"
    )

# --- Download Data Page ---
elif page == "Download Data":
    st.title("ดาวน์โหลดชุดข้อมูล")
    st.markdown("สามารถดาวน์โหลดชุดข้อมูล Mouse Viral Infection ได้ด้านล่าง")
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 ดาวน์โหลด CSV",
        data=csv,
        file_name='mouse_viral_infection.csv',
        mime='text/csv'
    )

# --- About Page ---
else:
    st.title("เกี่ยวกับแอปพลิเคชัน")
    st.markdown(
        """
        **Mouse Viral Infection Study App**  
        พัฒนาโดย: Your Name  
        เทคโนโลยี: Streamlit, pandas, numpy, scikit-learn, matplotlib  
        เวอร์ชัน: 1.2.0
        """
    )

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.write("Developed with 💙 by Your Name")
