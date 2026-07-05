"""
Crop Yield AI Dashboard
Kelompok: Revaldo Ramadana, Wirsan Wijoyo, Ahmad Rifky Perdana
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io

from sklearn.model_selection import train_test_split

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Yield · Data Mining",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
*, *::before, *::after { box-sizing: border-box; }
:root {
    --bg:       #F4F6FB;
    --surface:  #FFFFFF;
    --border:   #D4DAE8;
    --indigo:   #4F6EF7;
    --indigo-d: #3A55D4;
    --teal:     #0EA88C;
    --amber:    #E8960A;
    --rose:     #E53E3E;
    --violet:   #7C4DFF;
    --muted:    #64748B;
    --text:     #1A2340;
    --font-h:   'Syne', sans-serif;
    --font-b:   'DM Sans', sans-serif;
    --font-m:   'DM Mono', monospace;
}
html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-b) !important;
}
section[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 2px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
h1, h2, h3, h4, h5 {
    font-family: var(--font-h) !important;
    color: var(--text) !important;
    letter-spacing: -0.3px !important;
}
p, span, div, label { color: var(--muted) !important; }
code, .stCode { font-family: var(--font-m) !important; }
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px !important;
    transition: border-color .2s, transform .2s, box-shadow .2s;
}
[data-testid="stMetric"]:hover {
    border-color: var(--indigo);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(79,110,247,.12);
}
[data-testid="stMetricLabel"] p {
    font-family: var(--font-m) !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-h) !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    color: var(--indigo) !important;
}
.stButton > button {
    background: var(--indigo) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-h) !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    transition: all .2s !important;
    box-shadow: 0 2px 10px rgba(79,110,247,.3) !important;
}
.stButton > button:hover {
    background: var(--indigo-d) !important;
    box-shadow: 0 4px 20px rgba(79,110,247,.45) !important;
    transform: translateY(-1px) !important;
}
.stButton > button p { color: #fff !important; }
[data-testid="stNumberInput"] input {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 7px !important;
    font-family: var(--font-m) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 7px !important;
    color: var(--text) !important;
}
[data-testid="stDataFrame"] {
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    background: var(--surface) !important;
}
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: var(--text) !important; }
[data-testid="stTabs"] [role="tab"] {
    font-family: var(--font-h) !important;
    color: var(--muted) !important;
    font-size: 13px !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--indigo) !important;
    border-bottom: 2px solid var(--indigo) !important;
}
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }
[data-testid="stAlert"] { border-radius: 8px !important; }
.page-header {
    border-left: 4px solid var(--indigo);
    padding: 4px 0 4px 18px;
    margin-bottom: 12px;
}
.page-header h2 { margin: 0 !important; font-size: 22px !important; color: var(--text) !important; }
.page-header p  { margin: 4px 0 0 !important; font-size: 13px !important; color: var(--muted) !important; }
.section-tag {
    display: inline-block;
    background: rgba(79,110,247,.1);
    color: var(--indigo) !important;
    font-family: var(--font-m) !important;
    font-size: 10px !important;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 4px;
    margin-bottom: 10px;
    border: 1px solid rgba(79,110,247,.2);
}
.result-card {
    border-radius: 14px;
    padding: 28px;
    text-align: center;
    margin-top: 20px;
}
.result-card h2 { font-family: var(--font-h) !important; font-size: 28px !important; }
.model-card {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
    transition: border-color .2s, box-shadow .2s;
}
.model-card:hover {
    border-color: var(--indigo);
    box-shadow: 0 4px 16px rgba(79,110,247,.1);
}
.model-card-title {
    font-family: var(--font-h) !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    margin: 0 0 4px !important;
}
.model-card-sub {
    font-family: var(--font-m) !important;
    font-size: 10px !important;
    color: var(--muted) !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — Data & Model Loading
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_raw_data_from_bytes(file_bytes: bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.drop_duplicates(inplace=True)
    return df


@st.cache_data(show_spinner=False)
def load_raw_data_from_disk():
    path = "data/yield_df.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.drop_duplicates(inplace=True)
    return df


@st.cache_data(show_spinner=False)
def load_processed_data(_raw_df):
    from preprocessing import preprocess_data
    return preprocess_data(_raw_df)


@st.cache_resource(show_spinner=False)
def load_models():
    base = "models"
    needed = [
        "decision_tree_model.pkl",
        "naive_bayes_model.pkl",
        "scaler.pkl",
        "label_encoder.pkl",
        "feature_columns.pkl",
    ]
    for f in needed:
        if not os.path.exists(f"{base}/{f}"):
            return None
    result = {
        "dt":     joblib.load(f"{base}/decision_tree_model.pkl"),
        "nb":     joblib.load(f"{base}/naive_bayes_model.pkl"),
        "scaler": joblib.load(f"{base}/scaler.pkl"),
        "le":     joblib.load(f"{base}/label_encoder.pkl"),
        "cols":   joblib.load(f"{base}/feature_columns.pkl"),
        "cv":     joblib.load(f"{base}/cv_scores.pkl")
                  if os.path.exists(f"{base}/cv_scores.pkl") else None,
    }
    return result


def build_input_row(cols, scaler, year, rainfall, temp, pesticides,
                    selected_area, selected_item):
    """Buat satu baris input yang sudah discale dan di-encode."""
    input_row = {c: 0 for c in cols}
    numeric_only = {
        'Year': year,
        'average_rain_fall_mm_per_year': rainfall,
        'pesticides_tonnes': pesticides,
        'avg_temp': temp,
    }
    scaled = scaler.transform(pd.DataFrame([numeric_only]))
    for i, col in enumerate(numeric_only.keys()):
        if col in input_row:
            input_row[col] = scaled[0][i]

    if selected_area and selected_area != "(Tidak dipilih)":
        key = "Area_" + selected_area.lower().replace(" ", "_")
        if key in input_row:
            input_row[key] = 1
    if selected_item and selected_item != "(Tidak dipilih)":
        key = "Item_" + selected_item.lower().replace(" ", "_")
        if key in input_row:
            input_row[key] = 1

    return pd.DataFrame([input_row])[cols]


# ── Session state init ────────────────────────────────────────────────────────
if "uploaded_bytes" not in st.session_state:
    st.session_state["uploaded_bytes"] = None

# ── Resolve raw_df ────────────────────────────────────────────────────────────
if st.session_state["uploaded_bytes"] is not None:
    raw_df = load_raw_data_from_bytes(st.session_state["uploaded_bytes"])
else:
    raw_df = load_raw_data_from_disk()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:12px 0 20px;'>
        <div style='font-size:38px; line-height:1;'>&#127807;</div>
        <div style='font-family:"Syne",sans-serif; font-size:19px; font-weight:800;
                    color:#1A2340; letter-spacing:-0.5px; margin-top:8px;'>
            Crop Yield AI
        </div>
        <div style='font-family:"DM Mono",monospace; font-size:10px;
                    color:#64748B; letter-spacing:2px; text-transform:uppercase; margin-top:4px;'>
            Data Mining Dashboard
        </div>
    </div>
    <hr style='border-color:#D4DAE8; margin:0 0 16px;'>
    """, unsafe_allow_html=True)

    # ── Dataset Upload ────────────────────────────────────────────────────────
    st.markdown("""<div style='font-family:"DM Mono",monospace; font-size:10px; font-weight:600;
                text-transform:uppercase; letter-spacing:1.5px; color:#4F6EF7;
                margin-bottom:8px;'>Dataset</div>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload file CSV dataset",
        type=["csv"],
        help="Upload yield_df.csv. Data langsung diproses secara otomatis.",
        label_visibility="visible",
    )
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        if file_bytes != st.session_state.get("uploaded_bytes"):
            st.session_state["uploaded_bytes"] = file_bytes
            load_raw_data_from_bytes.clear()
            load_processed_data.clear()
            st.rerun()

    if raw_df is not None:
        st.markdown(f"""
        <div style='background:rgba(14,168,140,.08); border:1px solid rgba(14,168,140,.3);
                    border-radius:8px; padding:10px 14px; margin:10px 0 14px;
                    font-family:"DM Mono",monospace; font-size:10px; line-height:1.9;'>
            <span style='color:#0EA88C; font-weight:700; letter-spacing:1px;'>DATASET DIMUAT</span><br>
            <span style='color:#64748B;'>Records : {raw_df.shape[0]:,}</span><br>
            <span style='color:#64748B;'>Fitur   : {raw_df.shape[1]}</span><br>
            <span style='color:#64748B;'>Kelas   : Rendah · Sedang · Tinggi</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:rgba(229,62,62,.06); border:1px solid rgba(229,62,62,.25);
                    border-radius:8px; padding:10px 14px; margin:10px 0 14px;
                    font-family:"DM Mono",monospace; font-size:10px; color:#E53E3E;'>
            Dataset belum dimuat.<br>Upload file CSV di atas.
        </div>""", unsafe_allow_html=True)
        st.stop()

    st.markdown("<hr style='border-color:#D4DAE8; margin:8px 0 14px;'>", unsafe_allow_html=True)

    menu = st.radio(
        "Navigasi",
        ["Overview & EDA", "Training & Model", "Model Evaluation", "Testing"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#D4DAE8; margin:14px 0 10px;'>", unsafe_allow_html=True)

    models = load_models()
    if models:
        st.markdown("""
        <div style='font-family:"DM Mono",monospace; font-size:10px; font-weight:700;
                    text-transform:uppercase; letter-spacing:1.5px; color:#0EA88C;
                    margin-bottom:6px;'>Model Loaded</div>
        <div style='font-family:"DM Mono",monospace; font-size:10px; color:#64748B; line-height:1.8;'>
            DT &nbsp;: siap<br>
            NB &nbsp;: siap
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='font-family:"DM Mono",monospace; font-size:10px; font-weight:700;
                    text-transform:uppercase; letter-spacing:1.5px; color:#E53E3E;
                    margin-bottom:6px;'>Model Belum Tersedia</div>
        <div style='font-family:"DM Mono",monospace; font-size:10px; color:#64748B;'>
            Buka halaman Training untuk melatih model.
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:"DM Mono",monospace; font-size:9px; color:#A0AABF;
                text-align:center; line-height:1.8;'>
        Revaldo · Wirsan · Ahmad Rifky<br>Penggalian Data · 2025
    </div>""", unsafe_allow_html=True)


# ── Load processed data ───────────────────────────────────────────────────────
original_df, df_encoded, scaler_raw, le_raw = load_processed_data(raw_df)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW & EDA
# ══════════════════════════════════════════════════════════════════════════════

if menu == "Overview & EDA":
    import eda as eda_module

    st.markdown("""
    <div class='page-header'>
        <div class='section-tag'>Tahap 1–3</div>
        <h2>Dataset Overview &amp; Exploratory Data Analysis</h2>
        <p>Ringkasan data mentah, atribut, dan distribusi fitur sebelum pre-processing.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",    f"{raw_df.shape[0]:,}")
    c2.metric("Total Features",   raw_df.shape[1])
    c3.metric("Missing Values",   raw_df.isnull().sum().sum())
    c4.metric("Unique Countries", raw_df.iloc[:, 1].nunique() if raw_df.shape[1] > 1 else "—")

    st.markdown("---")
    st.markdown("### Data Preview (Raw — 10 Baris Pertama)")
    st.dataframe(raw_df.head(10), use_container_width=True)

    with st.expander("Struktur Atribut Data"):
        attr_df = pd.DataFrame({
            "Nama Atribut": ["Area","Item","Year","hg/ha_yield",
                             "average_rain_fall_mm_per_year","pesticides_tonnes","avg_temp"],
            "Deskripsi":    ["Nama negara/wilayah","Jenis komoditas tanaman",
                             "Tahun pencatatan data","Hasil panen (hg/ha) — Target",
                             "Rata-rata curah hujan tahunan (mm/tahun)",
                             "Jumlah penggunaan pestisida (tonnes)",
                             "Rata-rata suhu tahunan (°C)"],
            "Tipe Data":    ["Kategorikal","Kategorikal","Numerik","Numerik (Target)",
                             "Numerik","Numerik","Numerik"],
            "Status":       ["Dipertahankan","Dipertahankan","Dipertahankan",
                             "Target","Dipertahankan","Dipertahankan","Dipertahankan"],
        })
        st.dataframe(attr_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Vis 1–3 — Boxplot Distribusi Fitur Numerik")
    st.caption("Deteksi outlier pada curah hujan, pestisida, dan suhu rata-rata.")
    st.pyplot(eda_module.plot_boxplots(raw_df), use_container_width=True)
    st.info("**Interpretasi:** Ditemukan banyak nilai ekstrem terutama pada penggunaan "
            "**pestisida**. Hal ini mendasari keputusan Z-score Standardization.")

    st.markdown("---")
    st.markdown("### Vis 4 — Distribusi Hasil Panen (Target Awal)")
    st.caption("Histogram + KDE distribusi nilai hg/ha_yield sebelum diskritisasi.")
    st.pyplot(eda_module.plot_distribution(raw_df), use_container_width=True)
    st.info("**Interpretasi:** Distribusi **right-skewed** — diskritisasi (binning) wajib dilakukan "
            "sebelum klasifikasi.")

    st.markdown("---")
    st.markdown("### Vis 5 — Keseimbangan Kelas Target (Setelah Diskritisasi)")
    st.caption("Distribusi kelas Rendah · Sedang · Tinggi setelah Equal Frequency Binning.")
    st.pyplot(eda_module.plot_target_balance(original_df), use_container_width=True)
    st.info("**Interpretasi:** Setelah `pd.qcut(q=3)`, ketiga kelas masing-masing ≈33% — "
            "balanced untuk pelatihan model klasifikasi.")

    st.markdown("---")
    st.markdown("### Ringkasan Pre-processing")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Sebelum Scaling**")
        nc = ['Year','average_rain_fall_mm_per_year','pesticides_tonnes','avg_temp']
        stats = raw_df[nc].describe().loc[['mean','std']].T
        stats.columns = ['Mean (Before)','Std (Before)']
        st.dataframe(stats.round(2), use_container_width=True)
    with col_b:
        st.markdown("**Sesudah Scaling (Z-score)**")
        st.dataframe(pd.DataFrame({
            "Fitur": nc,
            "Mean (After)": ["≈ 0.00"]*4,
            "Std (After)":  ["≈ 1.00"]*4,
        }), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TRAINING & MODEL
# ══════════════════════════════════════════════════════════════════════════════

elif menu == "Training & Model":
    import eda as eda_module
    from sklearn.tree import plot_tree, export_text

    st.markdown("""
    <div class='page-header'>
        <div class='section-tag'>Tahap 4</div>
        <h2>Training Model — 80% Train / 20% Test</h2>
        <p>Pelatihan dua model klasifikasi: Decision Tree dan Naive Bayes.
           Visualisasi struktur pohon keputusan dan ringkasan parameter model.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Train button ─────────────────────────────────────────────────────────
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        do_train = st.button("Latih Semua Model Sekarang", use_container_width=True)
    with col_info:
        st.markdown("""
        <div style='background:rgba(79,110,247,.06); border:1px solid rgba(79,110,247,.2);
                    border-radius:8px; padding:12px 16px; margin-top:4px;'>
            <span style='font-family:"DM Mono",monospace; font-size:11px; color:#4F6EF7;'>
                Split: 80% training · 20% testing · stratified · random_state=42
            </span>
        </div>
        """, unsafe_allow_html=True)

    if do_train:
        with st.spinner("Melatih Decision Tree dan Naive Bayes..."): 
            from train_model import train_and_save
            os.makedirs("data", exist_ok=True)
            raw_df.to_csv("data/yield_df.csv", index=False)
            train_and_save()
            st.cache_resource.clear()
            models = load_models()
        st.success("Kedua model berhasil dilatih dan disimpan.")

    models = load_models()

    # ── Model Cards ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Ringkasan Model yang Telah Dilatih")

    if models is None:
        st.warning("Model belum tersedia. Klik tombol di atas untuk melatih model.")
    else:
        X_full = df_encoded.drop(columns=['hg/ha_yield','Yield_Class'])
        y_full = models["le"].transform(df_encoded['Yield_Class'])
        cols   = models["cols"]
        for c in cols:
            if c not in X_full.columns:
                X_full[c] = 0
        X_full = X_full[cols]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
        )

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("""
            <div class='model-card'>
                <div class='model-card-title'>Decision Tree</div>
                <div class='model-card-sub'>Criterion: Gini · Max Depth: 5</div>
            </div>""", unsafe_allow_html=True)
            acc_dt = models["dt"].score(X_te, y_te)
            st.metric("Test Accuracy", f"{acc_dt:.2%}")
            if models.get("cv") is not None:
                st.metric("CV Mean (5-Fold)", f"{models['cv'].mean():.2%}")

        with c2:
            st.markdown("""
            <div class='model-card'>
                <div class='model-card-title'>Naive Bayes</div>
                <div class='model-card-sub'>GaussianNB · Prior: Uniform</div>
            </div>""", unsafe_allow_html=True)
            acc_nb = models["nb"].score(X_te, y_te)
            st.metric("Test Accuracy", f"{acc_nb:.2%}")



        # ── Split Info ────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Pembagian Data Training & Testing")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Data",    f"{len(X_full):,}")
        s2.metric("Training (80%)", f"{len(X_tr):,}")
        s3.metric("Testing (20%)",  f"{len(X_te):,}")
        s4.metric("Jumlah Fitur",   f"{len(cols):,}")

        # ── Decision Tree Visualization ───────────────────────────────────────
        st.markdown("---")
        st.markdown("### Visualisasi Pohon Keputusan (Decision Tree)")
        st.caption("Pohon dilatih dengan max_depth=5 agar terbaca. Warna menunjukkan kelas dominan pada setiap node.")

        import matplotlib.pyplot as plt
        class_names = list(models["le"].classes_)

        tab_vis, tab_text, tab_imp = st.tabs([
            "Visualisasi Pohon",
            "Representasi Teks",
            "Feature Importance",
        ])

        with tab_vis:
            depth_opt = st.slider(
                "Tampilkan kedalaman pohon (max_depth tampil)",
                min_value=1, max_value=5, value=3,
                help="Kurangi depth untuk tampilan lebih bersih."
            )
            fig_tree, ax_tree = plt.subplots(
                figsize=(max(20, depth_opt * 8), max(8, depth_opt * 3))
            )
            fig_tree.patch.set_facecolor('#F4F6FB')
            ax_tree.set_facecolor('#F4F6FB')

            plot_tree(
                models["dt"],
                max_depth=depth_opt,
                feature_names=cols,
                class_names=class_names,
                filled=True,
                rounded=True,
                fontsize=8,
                impurity=True,
                proportion=False,
                ax=ax_tree,
            )
            plt.tight_layout()
            st.pyplot(fig_tree, use_container_width=True)

            st.markdown("""
            <div style='background:#FFFFFF; border:1.5px solid #D4DAE8; border-radius:8px;
                        padding:14px 18px; margin-top:8px;
                        font-family:"DM Mono",monospace; font-size:11px; color:#64748B; line-height:1.9;'>
                <b style='color:#1A2340;'>Cara membaca pohon:</b><br>
                Node berwarna biru  = dominan kelas <b>Rendah</b><br>
                Node berwarna oranye = dominan kelas <b>Sedang</b><br>
                Node berwarna hijau  = dominan kelas <b>Tinggi</b><br>
                Semakin gelap warnanya, semakin murni (impurity rendah) node tersebut.
            </div>
            """, unsafe_allow_html=True)

        with tab_text:
            dt_text = export_text(
                models["dt"],
                feature_names=list(cols),
                max_depth=5,
            )
            st.code(dt_text, language=None)

        with tab_imp:
            st.pyplot(
                eda_module.plot_feature_importance(models["dt"], cols),
                use_container_width=True
            )



        # ── Cross Validation ──────────────────────────────────────────────────
        if models.get("cv") is not None:
            st.markdown("---")
            st.markdown("### 5-Fold Cross Validation — Decision Tree")
            cv_s = models["cv"]
            st.pyplot(
                eda_module.plot_cross_validation(cv_s, cv_s.mean()),
                use_container_width=True
            )
            ca, cb, cc = st.columns(3)
            ca.metric("Mean CV Accuracy", f"{cv_s.mean():.2%}")
            cb.metric("Std Deviation",     f"± {cv_s.std():.4f}")
            cc.metric("Test Set Accuracy", f"{acc_dt:.2%}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

elif menu == "Model Evaluation":
    import eda as eda_module
    from evaluation import evaluate_model

    st.markdown("""
    <div class='page-header'>
        <div class='section-tag'>Tahap 5</div>
        <h2>Model Evaluation</h2>
        <p>Perbandingan performa Decision Tree dan Naive Bayes pada data test 20%.</p>
    </div>
    """, unsafe_allow_html=True)

    models = load_models()
    if models is None:
        st.warning("Model belum tersedia. Latih model dari halaman Training terlebih dahulu.")
        st.stop()

    with st.spinner("Menyiapkan data evaluasi..."):
        le   = models["le"]
        cols = models["cols"]

        X_full = df_encoded.drop(columns=['hg/ha_yield','Yield_Class'])
        y_full = le.transform(df_encoded['Yield_Class'])
        for col in cols:
            if col not in X_full.columns:
                X_full[col] = 0
        X_full = X_full[cols]

        _, X_test, _, y_test = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
        )
        dt_res  = evaluate_model(models["dt"], X_test, y_test)
        nb_res  = evaluate_model(models["nb"], X_test, y_test)
        class_labels = le.classes_

    # ── Metrik Ringkasan ──────────────────────────────────────────────────────
    st.markdown("### Perbandingan Metrik Evaluasi")

    rows = ["Akurasi","Precision (weighted)","Recall (weighted)","F1-Score (weighted)"]
    tbl_data = {
        "Metrik":        rows,
        "Decision Tree": [f"{dt_res['accuracy']:.2%}", f"{dt_res['precision']:.2%}",
                          f"{dt_res['recall']:.2%}",   f"{dt_res['f1']:.2%}"],
        "Naive Bayes":   [f"{nb_res['accuracy']:.2%}", f"{nb_res['precision']:.2%}",
                          f"{nb_res['recall']:.2%}",   f"{nb_res['f1']:.2%}"],
    }
    st.dataframe(pd.DataFrame(tbl_data), use_container_width=True, hide_index=True)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    kpi_cols = st.columns(2)
    for col_ui, (name, res) in zip(kpi_cols, [
        ("Decision Tree", dt_res), ("Naive Bayes", nb_res),
    ]):
        with col_ui:
            st.markdown(f"#### {name}")
            a, b = st.columns(2)
            a.metric("Accuracy",  f"{res['accuracy']:.2%}")
            b.metric("F1-Score",  f"{res['f1']:.2%}")
            c, d = st.columns(2)
            c.metric("Precision", f"{res['precision']:.2%}")
            d.metric("Recall",    f"{res['recall']:.2%}")

    st.markdown("---")

    # ── Vis 8: Metrik Bar ─────────────────────────────────────────────────────
    st.markdown("### Perbandingan 4 Metrik Evaluasi")
    st.pyplot(
        eda_module.plot_metrics_comparison(dt_res, nb_res),
        use_container_width=True
    )

    st.markdown("---")

    # ── Confusion Matrices ────────────────────────────────────────────────────
    st.markdown("### Confusion Matrix")
    cm_models = [("Decision Tree", dt_res, "Blues"),
                 ("Naive Bayes",   nb_res, "Greens")]
    cm_cols = st.columns(len(cm_models))
    for col_ui, (name, res, cmap) in zip(cm_cols, cm_models):
        with col_ui:
            st.markdown(f"**{name}**")
            st.pyplot(
                eda_module.plot_confusion_matrix(
                    res['confusion_matrix'], class_labels,
                    f"Confusion Matrix: {name}", cmap=cmap
                ), use_container_width=True
            )

    st.markdown("---")

    # ── F1 Per Kelas ──────────────────────────────────────────────────────────
    st.markdown("### F1-Score Per Kelas")
    tab_line, tab_bar = st.tabs(["Line Chart", "Bar Chart"])
    with tab_line:
        st.pyplot(
            eda_module.plot_f1_per_class_line(
                dt_res['f1_per_class'], nb_res['f1_per_class']
            ), use_container_width=True
        )
    with tab_bar:
        st.pyplot(
            eda_module.plot_f1_per_class_bar(
                dt_res['f1_per_class'], nb_res['f1_per_class']
            ), use_container_width=True
        )

    st.markdown("---")

    # ── Classification Report ─────────────────────────────────────────────────
    st.markdown("### Classification Report Lengkap")
    report_tabs = st.tabs(["Decision Tree", "Naive Bayes"])
    with report_tabs[0]:
        st.code(dt_res['classification_report'], language=None)
    with report_tabs[1]:
        st.code(nb_res['classification_report'], language=None)

    # ── NB Priors ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Naive Bayes — Class Priors")
    st.pyplot(eda_module.plot_nb_priors(models["nb"], class_labels), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — TESTING
# ══════════════════════════════════════════════════════════════════════════════

elif menu == "Testing":
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    st.markdown("""
    <div class='page-header'>
        <div class='section-tag'>Aplikasi</div>
        <h2>Testing — Prediksi Kelas Hasil Panen</h2>
        <p>Uji model dengan input tunggal (Single) atau unggah file CSV untuk prediksi massal (Batch).
           Setiap hasil dilengkapi visualisasi probabilitas, interpretasi, dan analisis konteks.</p>
    </div>
    """, unsafe_allow_html=True)

    models = load_models()
    if models is None:
        st.warning("Model belum tersedia. Latih model dari halaman Training terlebih dahulu.")
        st.stop()

    # ── Pilih Model ───────────────────────────────────────────────────────────
    model_choice_opts = ["Decision Tree", "Naive Bayes"]

    st.markdown("""
    <div style='font-family:"DM Mono",monospace; font-size:10px; font-weight:600;
                text-transform:uppercase; letter-spacing:1.5px; color:#4F6EF7;
                margin-bottom:8px;'>Pilih Model</div>
    """, unsafe_allow_html=True)
    model_choice   = st.selectbox("Model", model_choice_opts, label_visibility="collapsed")
    model_map      = {"Decision Tree": models["dt"], "Naive Bayes": models["nb"]}
    selected_model = model_map[model_choice]

    color_map     = {"Rendah": "#E53E3E", "Sedang": "#E8960A", "Tinggi": "#0EA88C"}
    color_bg_map  = {"Rendah": "#FFF5F5", "Sedang": "#FFFBF0", "Tinggi": "#F0FFF9"}
    icon_map      = {"Rendah": "⬇", "Sedang": "➡", "Tinggi": "⬆"}
    interp_map    = {
        "Rendah": "Hasil panen diprediksi **rendah**. Kondisi ini umumnya terjadi pada lahan dengan "
                  "curah hujan tidak optimal, suhu ekstrem, atau jenis komoditas yang kurang produktif "
                  "di wilayah tersebut. Pertimbangkan rotasi tanaman atau peningkatan input agronomi.",
        "Sedang": "Hasil panen diprediksi **sedang**. Model mendeteksi kondisi pertanian yang cukup "
                  "baik namun masih memiliki ruang untuk peningkatan. Optimasi penggunaan pestisida "
                  "dan pemilihan varietas unggul dapat meningkatkan produktivitas.",
        "Tinggi": "Hasil panen diprediksi **tinggi**. Kombinasi faktor iklim dan jenis komoditas "
                  "pada input ini menunjukkan kondisi pertanian yang sangat baik. Pertahankan "
                  "praktik agronomi yang ada dan pantau konsistensi hasilnya.",
    }

    tab_single, tab_batch = st.tabs(["Single — Prediksi Satu Data", "Batch — Prediksi Massal CSV"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB SINGLE
    # ════════════════════════════════════════════════════════════════════════
    with tab_single:

        area_cols_list = [c for c in models["cols"] if c.startswith("Area_")]
        item_cols_list = [c for c in models["cols"] if c.startswith("Item_")]
        area_options   = ["(Tidak dipilih)"] + sorted([
            c.replace("Area_","").replace("_"," ").title() for c in area_cols_list])
        item_options   = ["(Tidak dipilih)"] + sorted([
            c.replace("Item_","").replace("_"," ").title() for c in item_cols_list])

        # ── Input Form ────────────────────────────────────────────────────
        st.markdown("""
        <div style='font-family:"DM Mono",monospace; font-size:10px; font-weight:600;
                    text-transform:uppercase; letter-spacing:1.5px; color:#4F6EF7;
                    margin:16px 0 12px;'>Parameter Input</div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("""<div style='font-family:"DM Mono",monospace; font-size:11px;
                color:#1A2340; font-weight:600; margin-bottom:8px;
                border-left:3px solid #4F6EF7; padding-left:8px;'>Kondisi Iklim & Waktu</div>
            """, unsafe_allow_html=True)
            year     = st.number_input("Tahun", min_value=1990, max_value=2050, value=2010, step=1)
            rainfall = st.number_input("Curah Hujan (mm/tahun)",
                                       min_value=0.0, max_value=5000.0, value=1000.0, step=10.0)
            temp     = st.number_input("Suhu Rata-rata (°C)",
                                       min_value=-10.0, max_value=50.0, value=20.0, step=0.5)

        with col2:
            st.markdown("""<div style='font-family:"DM Mono",monospace; font-size:11px;
                color:#1A2340; font-weight:600; margin-bottom:8px;
                border-left:3px solid #0EA88C; padding-left:8px;'>Intervensi & Komoditas</div>
            """, unsafe_allow_html=True)
            pesticides    = st.number_input("Pestisida (tonnes)",
                                            min_value=0.0, max_value=1_000_000.0,
                                            value=10_000.0, step=100.0)
            selected_area = st.selectbox("Negara / Area", area_options, key="s_area")
            selected_item = st.selectbox("Komoditas Tanaman", item_options, key="s_item")

        run_single = st.button("Jalankan Prediksi", key="run_single", use_container_width=False)

        if run_single:
            with st.spinner("Menganalisis..."):
                try:
                    X_in = build_input_row(
                        models["cols"], models["scaler"],
                        year, rainfall, temp, pesticides,
                        selected_area, selected_item
                    )
                    pred_idx   = int(selected_model.predict(X_in)[0])
                    pred_prob  = selected_model.predict_proba(X_in)[0]
                    pred_label = models["le"].inverse_transform([pred_idx])[0]
                    color      = color_map[pred_label]
                    color_bg   = color_bg_map[pred_label]
                    classes    = list(models["le"].classes_)

                    st.markdown("---")

                    # ── HASIL UTAMA ────────────────────────────────────────
                    res_col, prob_col = st.columns([1, 1], gap="large")

                    with res_col:
                        st.markdown(f"""
                        <div style='background:{color_bg}; border:2px solid {color}40;
                                    border-radius:14px; padding:28px 24px; text-align:center;'>
                            <div style='font-family:"DM Mono",monospace; font-size:10px;
                                        letter-spacing:2px; text-transform:uppercase;
                                        color:{color}; margin-bottom:6px; font-weight:600;'>
                                Prediksi · {model_choice}
                            </div>
                            <div style='font-family:"Syne",sans-serif; font-size:52px;
                                        font-weight:800; color:{color}; line-height:1;
                                        margin:8px 0;'>{pred_label}</div>
                            <div style='font-family:"DM Mono",monospace; font-size:11px;
                                        color:{color}; opacity:.8; margin-top:6px;'>
                                Confidence : {pred_prob.max():.1%}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Input summary card
                        st.markdown(f"""
                        <div style='background:#FFFFFF; border:1.5px solid #D4DAE8;
                                    border-radius:10px; padding:16px 18px; margin-top:14px;
                                    font-family:"DM Mono",monospace; font-size:11px;
                                    line-height:2; color:#64748B;'>
                            <div style='font-family:"Syne",monospace; font-size:12px;
                                        font-weight:700; color:#1A2340; margin-bottom:6px;'>
                                Ringkasan Input
                            </div>
                            Tahun &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <b style='color:#1A2340'>{year}</b><br>
                            Curah Hujan: <b style='color:#1A2340'>{rainfall:,.0f} mm/tahun</b><br>
                            Pestisida &nbsp;: <b style='color:#1A2340'>{pesticides:,.0f} tonnes</b><br>
                            Suhu &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <b style='color:#1A2340'>{temp} °C</b><br>
                            Negara &nbsp;&nbsp;&nbsp;: <b style='color:#1A2340'>{selected_area}</b><br>
                            Komoditas : <b style='color:#1A2340'>{selected_item}</b>
                        </div>
                        """, unsafe_allow_html=True)

                    with prob_col:
                        # ── Chart: Probability Gauge Bars ──────────────────
                        fig_prob, ax_prob = plt.subplots(figsize=(5, 3.5))
                        fig_prob.patch.set_facecolor('#FFFFFF')
                        ax_prob.set_facecolor('#FFFFFF')

                        bar_colors = [color_map[c] for c in classes]
                        bar_alphas = [1.0 if c == pred_label else 0.25 for c in classes]
                        bars = ax_prob.barh(
                            classes,
                            [pred_prob[i] * 100 for i in range(len(classes))],
                            color=[color_map[c] for c in classes],
                            alpha=1.0,
                            height=0.5,
                            edgecolor='white',
                        )
                        # dim non-predicted bars
                        for bar, cls in zip(bars, classes):
                            if cls != pred_label:
                                bar.set_alpha(0.22)

                        for bar, prob, cls in zip(bars, pred_prob, classes):
                            x_pos = prob * 100
                            label_x = x_pos + 1.5 if x_pos < 80 else x_pos - 2
                            ha = 'left' if x_pos < 80 else 'right'
                            fw = 700 if cls == pred_label else 400
                            ax_prob.text(
                                label_x, bar.get_y() + bar.get_height() / 2,
                                f"{prob:.1%}", va='center', ha=ha,
                                fontsize=11, fontweight=fw,
                                color=color_map[cls] if cls == pred_label else '#64748B'
                            )

                        ax_prob.set_xlim(0, 105)
                        ax_prob.set_xlabel("Probabilitas (%)", fontsize=9, color='#64748B')
                        ax_prob.set_title("Distribusi Probabilitas Per Kelas",
                                         fontsize=11, fontweight='bold', color='#1A2340', pad=10)
                        for spine in ax_prob.spines.values():
                            spine.set_color('#D4DAE8')
                            spine.set_linewidth(0.8)
                        ax_prob.tick_params(colors='#64748B', labelsize=10)
                        ax_prob.grid(axis='x', color='#E8ECF3', linewidth=0.8, alpha=1.0)
                        ax_prob.set_axisbelow(True)
                        plt.tight_layout(pad=1.5)
                        st.pyplot(fig_prob, use_container_width=True)

                        # ── Chart: Probability Pie ────────────────────────
                        fig_pie, ax_pie = plt.subplots(figsize=(5, 3.0))
                        fig_pie.patch.set_facecolor('#FFFFFF')
                        pie_colors = [color_map[c] for c in classes]
                        pie_alphas = [1.0 if c == pred_label else 0.2 for c in classes]
                        wedge_props = dict(width=0.55, edgecolor='white', linewidth=2)
                        wedges, texts, autotexts = ax_pie.pie(
                            pred_prob,
                            labels=classes,
                            colors=pie_colors,
                            autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
                            startangle=90,
                            wedgeprops=wedge_props,
                            pctdistance=0.75,
                            textprops={'fontsize': 10, 'color': '#64748B'},
                        )
                        for wedge, cls in zip(wedges, classes):
                            if cls != pred_label:
                                wedge.set_alpha(0.2)
                        for autotext, cls in zip(autotexts, classes):
                            autotext.set_fontsize(9)
                            autotext.set_fontweight('bold' if cls == pred_label else 'normal')
                            autotext.set_color(color_map[cls] if cls == pred_label else '#64748B')
                        ax_pie.set_title("Komposisi Probabilitas",
                                        fontsize=10, fontweight='bold', color='#1A2340', pad=8)
                        plt.tight_layout(pad=1)
                        st.pyplot(fig_pie, use_container_width=True)

                    st.markdown("---")

                    # ── INTERPRETASI ──────────────────────────────────────
                    st.markdown(f"""
                    <div style='background:{color_bg}; border-left:4px solid {color};
                                border-radius:0 10px 10px 0; padding:16px 20px; margin-bottom:16px;'>
                        <div style='font-family:"DM Mono",monospace; font-size:10px;
                                    text-transform:uppercase; letter-spacing:1.5px;
                                    color:{color}; font-weight:600; margin-bottom:8px;'>
                            Interpretasi Hasil
                        </div>
                        <div style='font-size:13px; color:#1A2340; line-height:1.7;'>
                            {interp_map[pred_label].replace("**", "<b>").replace("**", "</b>")}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── ANALISIS KONTEKS INPUT ────────────────────────────
                    st.markdown("#### Analisis Konteks Parameter Input")

                    ctx_cols = st.columns(4)
                    # Curah hujan — referensi global avg ~1000 mm
                    rain_status = "Tinggi" if rainfall > 1500 else ("Rendah" if rainfall < 500 else "Normal")
                    rain_color  = "#0EA88C" if rain_status == "Tinggi" else ("#E53E3E" if rain_status == "Rendah" else "#E8960A")
                    ctx_cols[0].markdown(f"""
                    <div style='background:#FFFFFF; border:1.5px solid #D4DAE8; border-radius:10px;
                                padding:14px; text-align:center;'>
                        <div style='font-family:"DM Mono",monospace; font-size:9px; color:#64748B;
                                    text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;'>
                            Curah Hujan
                        </div>
                        <div style='font-family:"Syne",sans-serif; font-size:20px; font-weight:700;
                                    color:{rain_color};'>{rain_status}</div>
                        <div style='font-family:"DM Mono",monospace; font-size:10px; color:#64748B;
                                    margin-top:4px;'>{rainfall:,.0f} mm/thn</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Suhu — optimal crop ~15-25°C
                    temp_status = "Optimal" if 15 <= temp <= 28 else ("Terlalu Panas" if temp > 28 else "Terlalu Dingin")
                    temp_color  = "#0EA88C" if temp_status == "Optimal" else "#E53E3E"
                    ctx_cols[1].markdown(f"""
                    <div style='background:#FFFFFF; border:1.5px solid #D4DAE8; border-radius:10px;
                                padding:14px; text-align:center;'>
                        <div style='font-family:"DM Mono",monospace; font-size:9px; color:#64748B;
                                    text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;'>
                            Suhu
                        </div>
                        <div style='font-family:"Syne",sans-serif; font-size:20px; font-weight:700;
                                    color:{temp_color};'>{temp_status}</div>
                        <div style='font-family:"DM Mono",monospace; font-size:10px; color:#64748B;
                                    margin-top:4px;'>{temp} °C</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Pestisida — referensi median dataset ~37k tonnes
                    pest_status = "Tinggi" if pesticides > 50000 else ("Rendah" if pesticides < 5000 else "Sedang")
                    pest_color  = "#E8960A" if pest_status == "Tinggi" else ("#E53E3E" if pest_status == "Rendah" else "#0EA88C")
                    ctx_cols[2].markdown(f"""
                    <div style='background:#FFFFFF; border:1.5px solid #D4DAE8; border-radius:10px;
                                padding:14px; text-align:center;'>
                        <div style='font-family:"DM Mono",monospace; font-size:9px; color:#64748B;
                                    text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;'>
                            Pestisida
                        </div>
                        <div style='font-family:"Syne",sans-serif; font-size:20px; font-weight:700;
                                    color:{pest_color};'>{pest_status}</div>
                        <div style='font-family:"DM Mono",monospace; font-size:10px; color:#64748B;
                                    margin-top:4px;'>{pesticides:,.0f} t</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Confidence level
                    conf_val   = pred_prob.max()
                    conf_label = "Sangat Yakin" if conf_val > 0.80 else ("Cukup Yakin" if conf_val > 0.55 else "Tidak Yakin")
                    conf_color = "#0EA88C" if conf_val > 0.80 else ("#E8960A" if conf_val > 0.55 else "#E53E3E")
                    ctx_cols[3].markdown(f"""
                    <div style='background:#FFFFFF; border:1.5px solid #D4DAE8; border-radius:10px;
                                padding:14px; text-align:center;'>
                        <div style='font-family:"DM Mono",monospace; font-size:9px; color:#64748B;
                                    text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;'>
                            Keyakinan Model
                        </div>
                        <div style='font-family:"Syne",sans-serif; font-size:20px; font-weight:700;
                                    color:{conf_color};'>{conf_label}</div>
                        <div style='font-family:"DM Mono",monospace; font-size:10px; color:#64748B;
                                    margin-top:4px;'>{conf_val:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("---")

                    # ── PERBANDINGAN PREDIKSI SEMUA MODEL ────────────────
                    st.markdown("#### Perbandingan Prediksi Semua Model")
                    st.caption("Prediksi dan probabilitas dari ketiga model untuk input yang sama.")

                    all_models = [
                        ("Decision Tree", models["dt"]),
                        ("Naive Bayes",   models["nb"]),
                    ]


                    cmp_cols = st.columns(len(all_models))
                    for col_ui, (mname, mobj) in zip(cmp_cols, all_models):
                        try:
                            p_idx   = int(mobj.predict(X_in)[0])
                            p_prob  = mobj.predict_proba(X_in)[0]
                            p_label = models["le"].inverse_transform([p_idx])[0]
                            p_color = color_map[p_label]
                            is_selected = mname == model_choice
                            border_style = f"2.5px solid {p_color}" if is_selected else "1.5px solid #D4DAE8"

                            col_ui.markdown(f"""
                            <div style='background:#FFFFFF; border:{border_style};
                                        border-radius:12px; padding:16px; text-align:center;'>
                                <div style='font-family:"DM Mono",monospace; font-size:9px;
                                            text-transform:uppercase; letter-spacing:1px;
                                            color:#64748B; margin-bottom:6px;'>{mname}</div>
                                <div style='font-family:"Syne",sans-serif; font-size:22px;
                                            font-weight:800; color:{p_color};'>{p_label}</div>
                                <div style='font-family:"DM Mono",monospace; font-size:10px;
                                            color:{p_color}; margin-top:4px;'>{p_prob.max():.1%}</div>
                                <div style='margin-top:10px; font-family:"DM Mono",monospace;
                                            font-size:9px; color:#64748B; line-height:1.8;'>
                                    R: {p_prob[0]:.3f} &nbsp; S: {p_prob[1]:.3f} &nbsp; T: {p_prob[2]:.3f}
                                </div>
                                {'<div style="font-family:DM Mono,monospace;font-size:8px;background:' + p_color + '15;color:' + p_color + ';padding:2px 8px;border-radius:4px;margin-top:6px;letter-spacing:1px;">DIPILIH</div>' if is_selected else ''}
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception:
                            col_ui.warning(f"{mname} tidak tersedia.")

                    # ── RADAR / SPIDER CHART ─────────────────────────────
                    st.markdown("---")
                    st.markdown("#### Visualisasi Probabilitas — Semua Model")

                    fig_radar, ax_radar = plt.subplots(figsize=(10, 3.8))
                    fig_radar.patch.set_facecolor('#FFFFFF')
                    ax_radar.set_facecolor('#FFFFFF')

                    x_idx     = np.arange(len(classes))
                    bar_w     = 0.22
                    model_clr = ["#4F6EF7", "#0EA88C", "#F97316"]

                    for i, (mname, mobj) in enumerate(all_models):
                        try:
                            pp = mobj.predict_proba(X_in)[0]
                            offset = (i - len(all_models) / 2 + 0.5) * bar_w
                            bars_r = ax_radar.bar(
                                x_idx + offset, pp * 100, bar_w,
                                label=mname,
                                color=model_clr[i % len(model_clr)],
                                alpha=0.85,
                                edgecolor='white',
                                linewidth=0.8,
                                zorder=3,
                            )
                            for bar, val in zip(bars_r, pp):
                                ax_radar.text(
                                    bar.get_x() + bar.get_width() / 2,
                                    bar.get_height() + 0.8,
                                    f'{val:.1%}',
                                    ha='center', va='bottom',
                                    fontsize=8.5, color='#1A2340', fontweight='bold'
                                )
                        except Exception:
                            pass

                    ax_radar.set_xticks(x_idx)
                    ax_radar.set_xticklabels(classes, fontsize=11, color='#1A2340', fontweight='bold')
                    ax_radar.set_ylabel("Probabilitas (%)", fontsize=9, color='#64748B')
                    ax_radar.set_ylim(0, 115)
                    ax_radar.set_title(
                        f"Probabilitas Per Kelas — Semua Model (Input: {selected_item if selected_item != '(Tidak dipilih)' else 'tidak dipilih'})",
                        fontsize=11, fontweight='bold', color='#1A2340', pad=10
                    )
                    for spine in ax_radar.spines.values():
                        spine.set_color('#D4DAE8')
                    ax_radar.tick_params(colors='#64748B')
                    ax_radar.grid(axis='y', color='#E8ECF3', linewidth=0.8)
                    ax_radar.set_axisbelow(True)
                    ax_radar.legend(
                        facecolor='#FFFFFF', edgecolor='#D4DAE8',
                        labelcolor='#1A2340', fontsize=9, framealpha=1
                    )

                    # Highlight predicted class region
                    pred_x = classes.index(pred_label)
                    ax_radar.axvspan(pred_x - 0.4, pred_x + 0.4, color=color_map[pred_label],
                                    alpha=0.07, zorder=0)
                    plt.tight_layout(pad=1.5)
                    st.pyplot(fig_radar, use_container_width=True)


                    # ── CONFUSION MATRIX SINGLE ──────────────────────────
                    st.markdown("---")
                    st.markdown("#### Confusion Matrix — Konteks Posisi Prediksi Ini")
                    st.caption(
                        "Menunjukkan di mana posisi prediksi ini berada dalam performa model "
                        "secara keseluruhan pada data test 20%."
                    )

                    from sklearn.model_selection import train_test_split as _tts
                    from sklearn.metrics import confusion_matrix as _cm
                    from evaluation import evaluate_model as _eval

                    le_ev   = models["le"]
                    cols_ev = models["cols"]
                    X_full_ev = df_encoded.drop(columns=["hg/ha_yield","Yield_Class"])
                    y_full_ev = le_ev.transform(df_encoded["Yield_Class"])
                    for c in cols_ev:
                        if c not in X_full_ev.columns:
                            X_full_ev[c] = 0
                    X_full_ev = X_full_ev[cols_ev]
                    _, X_te_ev, _, y_te_ev = _tts(
                        X_full_ev, y_full_ev, test_size=0.2,
                        random_state=42, stratify=y_full_ev
                    )
                    ev_res   = _eval(selected_model, X_te_ev, y_te_ev)
                    cm_data  = ev_res["confusion_matrix"]
                    cl_names = list(le_ev.classes_)

                    cm_col, cm_info_col = st.columns([1.2, 1], gap="large")

                    with cm_col:
                        import matplotlib.pyplot as plt
                        import numpy as np
                        fig_cm_s, ax_cm_s = plt.subplots(figsize=(5, 4))
                        fig_cm_s.patch.set_facecolor('#FFFFFF')
                        ax_cm_s.set_facecolor('#FFFFFF')

                        cmap_name = {"Decision Tree":"Blues","Naive Bayes":"Greens"}
                        im_s = ax_cm_s.imshow(cm_data, interpolation='nearest',
                                              cmap=cmap_name.get(model_choice, "Blues"), aspect='auto')
                        cb_s = fig_cm_s.colorbar(im_s, ax=ax_cm_s, fraction=0.046, pad=0.04)
                        cb_s.ax.tick_params(colors='#64748B', labelsize=8)

                        ax_cm_s.set_xticks(range(len(cl_names)))
                        ax_cm_s.set_yticks(range(len(cl_names)))
                        ax_cm_s.set_xticklabels(cl_names, color='#1A2340', fontsize=10)
                        ax_cm_s.set_yticklabels(cl_names, color='#1A2340', fontsize=10)
                        ax_cm_s.set_xlabel("Predicted Label", color='#64748B', fontsize=9)
                        ax_cm_s.set_ylabel("True Label", color='#64748B', fontsize=9)
                        ax_cm_s.set_title(f"Confusion Matrix — {model_choice}",
                                         fontsize=11, fontweight='bold', color='#1A2340', pad=10)

                        thresh_s = cm_data.max() / 2
                        for i in range(cm_data.shape[0]):
                            for j in range(cm_data.shape[1]):
                                ax_cm_s.text(j, i, f"{cm_data[i,j]:,}",
                                            ha='center', va='center', fontsize=12, fontweight='bold',
                                            color='white' if cm_data[i,j] > thresh_s else '#1A2340')

                        # Highlight kolom prediksi saat ini
                        if pred_label in cl_names:
                            px = cl_names.index(pred_label)
                            ax_cm_s.add_patch(plt.Rectangle(
                                (px - 0.5, -0.5), 1, len(cl_names),
                                fill=False, edgecolor=color, linewidth=3, zorder=5
                            ))

                        for spine in ax_cm_s.spines.values():
                            spine.set_color('#D4DAE8')
                        ax_cm_s.tick_params(colors='#64748B')
                        plt.tight_layout(pad=1.5)
                        st.pyplot(fig_cm_s, use_container_width=True)
                        st.caption(
                            f"Kotak berwarna menandai kolom kelas prediksi '{pred_label}' pada data test."
                        )

                    with cm_info_col:
                        st.markdown("""
                        <div style='font-family:"DM Mono",monospace; font-size:10px; font-weight:600;
                                    text-transform:uppercase; letter-spacing:1.5px; color:#4F6EF7;
                                    margin-bottom:10px;'>Cara Membaca Confusion Matrix</div>
                        """, unsafe_allow_html=True)
                        st.markdown("""
                        <div style='background:#FFFFFF; border:1.5px solid #D4DAE8; border-radius:10px;
                                    padding:14px 16px; font-size:12px; color:#1A2340; line-height:1.8;'>
                            <b>Baris</b> = Kelas nyata (True Label)<br>
                            <b>Kolom</b> = Prediksi model (Predicted)<br>
                            <b>Diagonal</b> = Prediksi benar<br>
                            <b>Off-diagonal</b> = Kesalahan klasifikasi<br><br>
                            <span style='color:#64748B;'>Kotak berwarna pada confusion matrix
                            menunjukkan kolom kelas yang diprediksi untuk input ini.</span>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
                        st.markdown("""
                        <div style='font-family:"DM Mono",monospace; font-size:10px; font-weight:600;
                                    text-transform:uppercase; letter-spacing:1.5px; color:#4F6EF7;
                                    margin-bottom:10px;'>Metrik Model pada Data Test</div>
                        """, unsafe_allow_html=True)

                        m1, m2 = st.columns(2)
                        m1.metric("Akurasi",   f"{ev_res['accuracy']:.2%}")
                        m2.metric("F1-Score",  f"{ev_res['f1']:.2%}")
                        m3, m4 = st.columns(2)
                        m3.metric("Precision", f"{ev_res['precision']:.2%}")
                        m4.metric("Recall",    f"{ev_res['recall']:.2%}")

                        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
                        st.markdown("""
                        <div style='font-family:"DM Mono",monospace; font-size:10px; font-weight:600;
                                    text-transform:uppercase; letter-spacing:1.5px; color:#4F6EF7;
                                    margin-bottom:10px;'>F1-Score Per Kelas</div>
                        """, unsafe_allow_html=True)
                        for cls_n, f1_v in zip(cl_names, ev_res['f1_per_class']):
                            clr_f1 = color_map.get(cls_n, "#4F6EF7")
                            is_pred = cls_n == pred_label
                            st.markdown(f"""
                            <div style='display:flex; align-items:center; gap:10px;
                                        margin-bottom:6px; {'font-weight:700;' if is_pred else ''}'>
                                <span style='font-family:"DM Mono",monospace; font-size:11px;
                                             color:{clr_f1}; width:55px;'>{cls_n}</span>
                                <div style='flex:1; background:#F4F6FB; border-radius:4px;
                                            height:12px; overflow:hidden;'>
                                    <div style='width:{f1_v*100:.1f}%; background:{clr_f1};
                                                height:100%; opacity:{1.0 if is_pred else 0.45};'></div>
                                </div>
                                <span style='font-family:"DM Mono",monospace; font-size:10px;
                                             color:{clr_f1};'>{f1_v:.2%}</span>
                            </div>
                            """, unsafe_allow_html=True)

                    # ── PERBANDINGAN DATA ORIGINAL vs PREDIKSI (SINGLE) ──
                    st.markdown("---")
                    st.markdown("#### Perbandingan Data Original vs Hasil Prediksi")
                    st.caption(
                        "Tabel perbandingan antara parameter input yang dimasukkan (data original) "
                        "dengan hasil prediksi model beserta probabilitas masing-masing kelas."
                    )

                    # Tabel side-by-side: original | prediksi
                    orig_col, pred_col_t = st.columns(2, gap="large")

                    with orig_col:
                        st.markdown("""
                        <div style='font-family:"DM Mono",monospace; font-size:10px; font-weight:600;
                                    text-transform:uppercase; letter-spacing:1.5px; color:#1A2340;
                                    border-left:3px solid #4F6EF7; padding-left:8px;
                                    margin-bottom:10px;'>Data Original (Input)</div>
                        """, unsafe_allow_html=True)
                        orig_tbl = pd.DataFrame({
                            "Parameter":  ["Tahun","Curah Hujan (mm/thn)","Suhu (°C)",
                                           "Pestisida (tonnes)","Negara","Komoditas"],
                            "Nilai":      [str(year), f"{rainfall:,.1f}", str(temp),
                                          f"{pesticides:,.1f}",
                                          selected_area if selected_area != "(Tidak dipilih)" else "—",
                                          selected_item if selected_item != "(Tidak dipilih)" else "—"],
                        })
                        st.dataframe(orig_tbl, use_container_width=True, hide_index=True)

                    with pred_col_t:
                        st.markdown(f"""
                        <div style='font-family:"DM Mono",monospace; font-size:10px; font-weight:600;
                                    text-transform:uppercase; letter-spacing:1.5px; color:{color};
                                    border-left:3px solid {color}; padding-left:8px;
                                    margin-bottom:10px;'>Hasil Prediksi Model</div>
                        """, unsafe_allow_html=True)
                        pred_tbl = pd.DataFrame({
                            "Parameter":  ["Model","Kelas Prediksi","Confidence",
                                           "P(Rendah)","P(Sedang)","P(Tinggi)"],
                            "Nilai":      [model_choice, pred_label,
                                          f"{pred_prob.max():.2%}",
                                          f"{pred_prob[0]:.4f}",
                                          f"{pred_prob[1]:.4f}",
                                          f"{pred_prob[2]:.4f}"],
                        })
                        st.dataframe(pred_tbl, use_container_width=True, hide_index=True)

                    # Visualisasi perbandingan: input feature vs distribusi dataset
                    st.markdown("##### Posisi Input terhadap Distribusi Dataset (Feature Context)")
                    st.caption(
                        "Garis merah menunjukkan nilai input kamu dibandingkan distribusi "
                        "seluruh data training untuk setiap fitur numerik."
                    )

                    num_feats = [
                        ("Curah Hujan (mm/thn)", "average_rain_fall_mm_per_year", rainfall),
                        ("Suhu (°C)",             "avg_temp",                      temp),
                        ("Pestisida (tonnes)",    "pesticides_tonnes",             pesticides),
                    ]

                    fig_ctx, axes_ctx = plt.subplots(1, 3, figsize=(12, 3.5))
                    fig_ctx.patch.set_facecolor('#FFFFFF')

                    for ax_c, (feat_label, feat_col, feat_val) in zip(axes_ctx, num_feats):
                        ax_c.set_facecolor('#FFFFFF')
                        if feat_col in original_df.columns:
                            data_vals = original_df[feat_col].dropna()
                            ax_c.hist(data_vals, bins=30, color='#4F6EF7', alpha=0.55,
                                     edgecolor='white', linewidth=0.5)
                            ax_c.axvline(feat_val, color=color, linewidth=2.5,
                                        linestyle='--', label=f"Input: {feat_val:,.1f}", zorder=5)
                            # percentile
                            pct = (data_vals < feat_val).mean() * 100
                            ax_c.set_title(f"{feat_label}\n(Persentil: {pct:.0f}%)",
                                          fontsize=10, fontweight='bold', color='#1A2340', pad=6)
                            ax_c.set_xlabel("Nilai", fontsize=8, color='#64748B')
                            ax_c.set_ylabel("Frekuensi", fontsize=8, color='#64748B')
                            ax_c.legend(fontsize=8, facecolor='#FFFFFF',
                                       edgecolor='#D4DAE8', labelcolor='#1A2340')
                        for spine in ax_c.spines.values():
                            spine.set_color('#D4DAE8')
                        ax_c.tick_params(colors='#64748B', labelsize=7)
                        ax_c.grid(color='#E8ECF3', linewidth=0.6)
                        ax_c.set_axisbelow(True)

                    plt.tight_layout(pad=1.8)
                    st.pyplot(fig_ctx, use_container_width=True)
                    st.caption(
                        "Persentil menunjukkan seberapa tinggi nilai input dibandingkan "
                        "seluruh data dalam dataset. Persentil 80% artinya nilai input lebih "
                        "tinggi dari 80% data lainnya."
                    )


                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")

    # ════════════════════════════════════════════════════════════════════════
    # TAB BATCH
    # ════════════════════════════════════════════════════════════════════════
    with tab_batch:

        st.markdown("""
        <div style='background:rgba(79,110,247,.06); border:1px solid rgba(79,110,247,.2);
                    border-radius:8px; padding:14px 18px; margin-bottom:14px;
                    font-family:"DM Mono",monospace; font-size:11px; color:#64748B; line-height:1.9;'>
            <b style='color:#1A2340; font-size:12px;'>Format CSV yang diperlukan</b><br>
            Kolom wajib &nbsp;&nbsp;: <span style='color:#4F6EF7;'>Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp</span><br>
            Kolom opsional : <span style='color:#4F6EF7;'>Area, Item</span><br>
            Tidak perlu menyertakan kolom hg/ha_yield atau Yield_Class.
        </div>
        """, unsafe_allow_html=True)

        # Download template
        template_df = pd.DataFrame({
            "Area": [
                "albania","india","brazil","china","usa",
                "indonesia","australia","canada","mexico","france",
                "nigeria","kenya","argentina","thailand","germany",
                "pakistan","ukraine","myanmar","colombia","ethiopia",
                "spain","malaysia","egypt","turkey","vietnam",
                "philippines","iran","bangladesh","tanzania","morocco",
                "peru","algeria","sudan","angola","ghana",
                "mozambique","madagascar","cameroon","niger","mali",
                "zambia","senegal","zimbabwe","guinea","haiti",
                "bolivia","paraguay","uruguay","chile","ecuador",
                "nepal","sri lanka","cambodia","laos","uzbekistan",
                "kazakhstan","georgia","armenia","azerbaijan","moldova"
            ],
            "Item": [
                "maize","wheat","cassava","rice paddy","potatoes",
                "maize","wheat","potatoes","maize","wheat",
                "cassava","maize","soybeans","rice paddy","wheat",
                "rice paddy","wheat","rice paddy","maize","maize",
                "wheat","rice paddy","wheat","maize","rice paddy",
                "rice paddy","wheat","rice paddy","maize","wheat",
                "potatoes","wheat","sorghum","maize","cassava",
                "maize","rice paddy","maize","millet","sorghum",
                "maize","maize","maize","cassava","maize",
                "potatoes","soybeans","soybeans","wheat","maize",
                "rice paddy","rice paddy","rice paddy","rice paddy","wheat",
                "wheat","maize","wheat","wheat","maize"
            ],
            "Year": [
                2010,2015,2012,2008,2018,
                2011,2016,2013,2009,2017,
                2010,2014,2012,2010,2016,
                2013,2015,2011,2017,2012,
                2014,2016,2010,2018,2015,
                2011,2013,2009,2016,2014,
                2010,2012,2015,2017,2013,
                2011,2008,2016,2014,2012,
                2010,2015,2013,2009,2017,
                2012,2011,2016,2014,2010,
                2013,2015,2012,2011,2016,
                2014,2010,2017,2015,2013
            ],
            "average_rain_fall_mm_per_year": [
                1485.0,1083.0,1761.0,645.0,715.0,
                2702.0,534.0,537.0,752.0,867.0,
                1150.0,630.0,591.0,1622.0,700.0,
                494.0,565.0,2090.0,2612.0,848.0,
                636.0,2875.0,51.0,593.0,1821.0,
                2348.0,228.0,2666.0,855.0,346.0,
                1648.0,89.0,416.0,1010.0,1187.0,
                1032.0,1513.0,1604.0,151.0,282.0,
                1020.0,686.0,657.0,1651.0,1440.0,
                1190.0,1130.0,1282.0,308.0,946.0,
                1800.0,1712.0,1904.0,1778.0,225.0,
                341.0,1180.0,575.0,560.0,527.0
            ],
            "pesticides_tonnes": [
                121.0,90500.0,54000.0,1800000.0,450000.0,
                48000.0,15200.0,92000.0,32000.0,73000.0,
                3200.0,980.0,41000.0,21000.0,44000.0,
                15000.0,37000.0,13000.0,9000.0,1100.0,
                38000.0,31000.0,14000.0,23000.0,31000.0,
                14000.0,22000.0,12000.0,820.0,4500.0,
                6200.0,5100.0,750.0,3400.0,1100.0,
                720.0,1800.0,2900.0,480.0,620.0,
                1300.0,690.0,810.0,540.0,290.0,
                7800.0,5600.0,4200.0,27000.0,3100.0,
                11000.0,9800.0,8500.0,7200.0,6400.0,
                18000.0,2700.0,4800.0,5900.0,3300.0
            ],
            "avg_temp": [
                16.37,25.0,25.4,15.0,8.5,
                26.3,21.6,1.7,21.0,11.8,
                26.8,17.6,17.3,27.3,9.5,
                22.0,9.3,27.5,24.4,21.7,
                17.2,27.0,22.0,13.9,25.9,
                26.6,17.4,25.1,22.5,17.8,
                13.2,22.6,28.4,24.2,25.7,
                23.8,18.9,24.5,28.1,27.9,
                19.4,28.7,20.6,25.3,25.1,
                10.5,23.4,17.8,11.2,22.0,
                24.6,27.8,26.9,25.4,12.1,
                8.4,15.7,12.3,13.8,10.2
            ],
        })
        csv_template = template_df.to_csv(index=False).encode("utf-8")
        dl_col, _ = st.columns([1, 3])
        with dl_col:
            st.download_button(
                "Unduh Template CSV (60 baris — variatif area, komoditas, iklim)",
                data=csv_template,
                file_name="template_batch.csv",
                mime="text/csv",
            )

        batch_file = st.file_uploader(
            "Upload file CSV untuk prediksi batch",
            type=["csv"],
            key="batch_upload",
            label_visibility="visible",
        )

        if batch_file is not None:
            try:
                batch_df = pd.read_csv(batch_file)

                st.markdown(f"**Preview data — {len(batch_df)} baris (menampilkan 5 baris pertama):**")
                st.dataframe(batch_df.head(5), use_container_width=True)

                run_batch = st.button("Jalankan Prediksi Batch", key="run_batch")

                if run_batch:
                    with st.spinner(f"Memproses {len(batch_df)} baris..."):
                        results = []
                        for _, row in batch_df.iterrows():
                            try:
                                yr   = float(row.get("Year", 2010))
                                rf   = float(row.get("average_rain_fall_mm_per_year", 1000))
                                tmp  = float(row.get("avg_temp", 20))
                                pst  = float(row.get("pesticides_tonnes", 1000))
                                area = str(row.get("Area", "")).strip()
                                item = str(row.get("Item", "")).strip()

                                X_in = build_input_row(
                                    models["cols"], models["scaler"],
                                    yr, rf, tmp, pst, area, item
                                )
                                pred_idx   = int(selected_model.predict(X_in)[0])
                                pred_prob  = selected_model.predict_proba(X_in)[0]
                                pred_label = models["le"].inverse_transform([pred_idx])[0]

                                results.append({
                                    "No":            len(results) + 1,
                                    "Area":          area or "—",
                                    "Item":          item or "—",
                                    "Year":          int(yr),
                                    "Curah Hujan":   round(rf, 1),
                                    "Pestisida (t)": round(pst, 1),
                                    "Suhu (°C)":     round(tmp, 2),
                                    "Prediksi":      pred_label,
                                    "Confidence":    f"{pred_prob.max():.1%}",
                                    "P(Rendah)":     round(float(pred_prob[0]), 4),
                                    "P(Sedang)":     round(float(pred_prob[1]), 4),
                                    "P(Tinggi)":     round(float(pred_prob[2]), 4),
                                })
                            except Exception as row_err:
                                results.append({
                                    "No": len(results) + 1,
                                    "Area": str(row.get("Area","")),"Item": str(row.get("Item","")),
                                    "Prediksi": "ERROR", "Confidence": "—",
                                    "Error": str(row_err)
                                })

                    result_df = pd.DataFrame(results)
                    pred_df   = result_df[result_df["Prediksi"] != "ERROR"] if "ERROR" not in result_df.get("Prediksi","") else result_df

                    st.markdown("---")

                    # ── KPI CARDS ──────────────────────────────────────────
                    st.markdown("""
                    <div style='font-family:"DM Mono",monospace; font-size:10px; font-weight:600;
                                text-transform:uppercase; letter-spacing:1.5px; color:#4F6EF7;
                                margin-bottom:12px;'>Ringkasan Hasil Prediksi Batch</div>
                    """, unsafe_allow_html=True)

                    dist = result_df["Prediksi"].value_counts()
                    total_valid = len(result_df[result_df["Prediksi"] != "ERROR"])

                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    kpi1.metric("Total Data", f"{len(result_df):,} baris")
                    kpi2.metric("Kelas Rendah",
                                f"{dist.get('Rendah', 0):,} baris",
                                f"{dist.get('Rendah',0)/total_valid*100:.1f}%" if total_valid else "0%")
                    kpi3.metric("Kelas Sedang",
                                f"{dist.get('Sedang', 0):,} baris",
                                f"{dist.get('Sedang',0)/total_valid*100:.1f}%" if total_valid else "0%")
                    kpi4.metric("Kelas Tinggi",
                                f"{dist.get('Tinggi', 0):,} baris",
                                f"{dist.get('Tinggi',0)/total_valid*100:.1f}%" if total_valid else "0%")

                    st.markdown("---")

                    # ── VISUALISASI BATCH ──────────────────────────────────
                    if total_valid > 0:
                        vis1, vis2 = st.columns(2, gap="large")

                        with vis1:
                            # Pie chart distribusi kelas
                            st.markdown("##### Distribusi Kelas Prediksi")
                            fig_pie_b, ax_pie_b = plt.subplots(figsize=(5, 4))
                            fig_pie_b.patch.set_facecolor('#FFFFFF')
                            pie_labels = []
                            pie_values = []
                            pie_colors_b = []
                            for kls in ["Rendah", "Sedang", "Tinggi"]:
                                cnt = dist.get(kls, 0)
                                if cnt > 0:
                                    pie_labels.append(f"{kls}\n({cnt} baris)")
                                    pie_values.append(cnt)
                                    pie_colors_b.append(color_map[kls])

                            if pie_values:
                                wedges_b, texts_b, autotexts_b = ax_pie_b.pie(
                                    pie_values,
                                    labels=pie_labels,
                                    colors=pie_colors_b,
                                    autopct='%1.1f%%',
                                    startangle=90,
                                    wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2),
                                    pctdistance=0.75,
                                    textprops={'fontsize': 10, 'color': '#1A2340'},
                                )
                                for at in autotexts_b:
                                    at.set_fontsize(10)
                                    at.set_fontweight('bold')
                            ax_pie_b.set_title(f"Distribusi Prediksi — {model_choice}",
                                              fontsize=11, fontweight='bold', color='#1A2340', pad=8)
                            plt.tight_layout()
                            st.pyplot(fig_pie_b, use_container_width=True)

                        with vis2:
                            # Bar chart confidence distribution
                            st.markdown("##### Distribusi Confidence Score")
                            conf_vals = []
                            for _, r in result_df.iterrows():
                                c_str = r.get("Confidence", "0%")
                                try:
                                    conf_vals.append(float(str(c_str).replace('%','')) / 100)
                                except Exception:
                                    pass

                            if conf_vals:
                                fig_conf, ax_conf = plt.subplots(figsize=(5, 4))
                                fig_conf.patch.set_facecolor('#FFFFFF')
                                ax_conf.set_facecolor('#FFFFFF')
                                n, bins, patches = ax_conf.hist(
                                    conf_vals, bins=min(20, max(5, len(conf_vals)//2)),
                                    color='#4F6EF7', alpha=0.75, edgecolor='white', linewidth=0.8
                                )
                                # Color bars by confidence zone
                                for patch, left in zip(patches, bins[:-1]):
                                    if left >= 0.80:
                                        patch.set_facecolor('#0EA88C')
                                        patch.set_alpha(0.8)
                                    elif left >= 0.55:
                                        patch.set_facecolor('#E8960A')
                                        patch.set_alpha(0.8)
                                    else:
                                        patch.set_facecolor('#E53E3E')
                                        patch.set_alpha(0.8)

                                ax_conf.axvline(np.mean(conf_vals), color='#1A2340',
                                               linestyle='--', linewidth=1.5,
                                               label=f'Mean: {np.mean(conf_vals):.1%}')
                                ax_conf.set_xlabel("Confidence Score", fontsize=9, color='#64748B')
                                ax_conf.set_ylabel("Jumlah Data", fontsize=9, color='#64748B')
                                ax_conf.set_title("Distribusi Confidence Score",
                                                 fontsize=11, fontweight='bold', color='#1A2340', pad=8)
                                for spine in ax_conf.spines.values():
                                    spine.set_color('#D4DAE8')
                                ax_conf.tick_params(colors='#64748B', labelsize=8)
                                ax_conf.grid(color='#E8ECF3', linewidth=0.8, alpha=1.0)
                                ax_conf.set_axisbelow(True)
                                ax_conf.legend(fontsize=9, facecolor='#FFFFFF',
                                              edgecolor='#D4DAE8', labelcolor='#1A2340')
                                # Legend zones
                                from matplotlib.patches import Patch
                                legend_els = [
                                    Patch(facecolor='#0EA88C', alpha=0.8, label='Sangat Yakin (>80%)'),
                                    Patch(facecolor='#E8960A', alpha=0.8, label='Cukup Yakin (55–80%)'),
                                    Patch(facecolor='#E53E3E', alpha=0.8, label='Kurang Yakin (<55%)'),
                                ]
                                ax_conf.legend(handles=legend_els, fontsize=8,
                                              facecolor='#FFFFFF', edgecolor='#D4DAE8',
                                              labelcolor='#1A2340', loc='upper left')
                                plt.tight_layout()
                                st.pyplot(fig_conf, use_container_width=True)

                        # ── CHART: PREDIKSI PER AREA / ITEM ───────────────
                        if "Area" in result_df.columns and result_df["Area"].nunique() > 1:
                            st.markdown("##### Prediksi Per Negara / Area")
                            fig_area, ax_area = plt.subplots(
                                figsize=(10, max(3.5, result_df["Area"].nunique() * 0.45))
                            )
                            fig_area.patch.set_facecolor('#FFFFFF')
                            ax_area.set_facecolor('#FFFFFF')

                            area_dist = result_df.groupby(["Area","Prediksi"]).size().unstack(fill_value=0)
                            for kls in ["Rendah","Sedang","Tinggi"]:
                                if kls not in area_dist.columns:
                                    area_dist[kls] = 0
                            area_dist = area_dist[["Rendah","Sedang","Tinggi"]]

                            bottom = np.zeros(len(area_dist))
                            for kls in ["Rendah","Sedang","Tinggi"]:
                                bars_a = ax_area.barh(
                                    area_dist.index, area_dist[kls],
                                    left=bottom, color=color_map[kls],
                                    alpha=0.8, label=kls, edgecolor='white', linewidth=0.6
                                )
                                bottom += area_dist[kls].values

                            ax_area.set_xlabel("Jumlah Prediksi", fontsize=9, color='#64748B')
                            ax_area.set_title("Distribusi Prediksi per Area",
                                             fontsize=11, fontweight='bold', color='#1A2340', pad=8)
                            for spine in ax_area.spines.values():
                                spine.set_color('#D4DAE8')
                            ax_area.tick_params(colors='#64748B', labelsize=9)
                            ax_area.grid(axis='x', color='#E8ECF3', linewidth=0.8)
                            ax_area.set_axisbelow(True)
                            ax_area.legend(fontsize=9, facecolor='#FFFFFF',
                                          edgecolor='#D4DAE8', labelcolor='#1A2340')
                            plt.tight_layout()
                            st.pyplot(fig_area, use_container_width=True)

                        # ── CHART: CONFIDENCE PER KELAS (STRIP + BOX) ─────────
                        st.markdown("##### Confidence Score Per Kelas Prediksi")

                        # Build confidence values per class
                        conf_by_class = {}
                        all_conf_vals = []
                        for kls in ["Rendah", "Sedang", "Tinggi"]:
                            subset = result_df[result_df["Prediksi"] == kls]["Confidence"]
                            vals = []
                            for v in subset:
                                try:
                                    vals.append(float(str(v).replace("%","")) / 100)
                                except Exception:
                                    pass
                            conf_by_class[kls] = vals
                            all_conf_vals.extend(vals)

                        present_classes = [k for k in ["Rendah","Sedang","Tinggi"] if conf_by_class[k]]
                        n_classes = len(present_classes)

                        if n_classes > 0:
                            # Decide: boxplot only if each class has >= 4 pts, else strip+bar
                            use_box = all(len(conf_by_class[k]) >= 4 for k in present_classes)

                            fig_conf2, axes_c2 = plt.subplots(1, 2, figsize=(12, 4))
                            fig_conf2.patch.set_facecolor('#FFFFFF')

                            # LEFT: strip plot (jitter) — always works regardless of n
                            ax_strip = axes_c2[0]
                            ax_strip.set_facecolor('#FFFFFF')
                            np.random.seed(42)
                            for xi, kls in enumerate(present_classes):
                                vals = conf_by_class[kls]
                                clr  = color_map[kls]
                                # jitter x
                                jitter = np.random.uniform(-0.15, 0.15, len(vals))
                                ax_strip.scatter(
                                    [xi + j for j in jitter], vals,
                                    color=clr, alpha=0.8, s=55,
                                    edgecolors='white', linewidths=0.8, zorder=4
                                )
                                # mean line
                                if vals:
                                    ax_strip.hlines(
                                        np.mean(vals), xi - 0.3, xi + 0.3,
                                        color=clr, linewidth=2.5, zorder=5
                                    )
                                    ax_strip.annotate(
                                        f"mean={np.mean(vals):.1%}",
                                        xy=(xi, np.mean(vals)),
                                        xytext=(xi + 0.32, np.mean(vals)),
                                        fontsize=8, color=clr, fontweight="bold",
                                        va="center"
                                    )
                            ax_strip.set_xticks(range(len(present_classes)))
                            ax_strip.set_xticklabels(present_classes, fontsize=11,
                                                     color="#1A2340", fontweight="bold")
                            ax_strip.yaxis.set_major_formatter(
                                plt.FuncFormatter(lambda y, _: f"{y:.0%}")
                            )
                            ax_strip.set_ylabel("Confidence Score", fontsize=9, color="#64748B")
                            ax_strip.set_title("Sebaran Confidence per Kelas (Strip Plot)",
                                              fontsize=11, fontweight="bold", color="#1A2340", pad=10)
                            for spine in ax_strip.spines.values():
                                spine.set_color("#D4DAE8")
                            ax_strip.tick_params(colors="#64748B", labelsize=9)
                            ax_strip.grid(axis="y", color="#E8ECF3", linewidth=0.8)
                            ax_strip.set_axisbelow(True)

                            # RIGHT: mean confidence bar chart per class
                            ax_bar_c = axes_c2[1]
                            ax_bar_c.set_facecolor('#FFFFFF')
                            means_c = [np.mean(conf_by_class[k]) * 100 for k in present_classes]
                            clrs_c  = [color_map[k] for k in present_classes]
                            bars_c  = ax_bar_c.bar(
                                present_classes, means_c,
                                color=[c + "CC" for c in clrs_c],
                                edgecolor=clrs_c, linewidth=1.8, width=0.5, zorder=3
                            )
                            for bar_c, val_c, clr_c in zip(bars_c, means_c, clrs_c):
                                ax_bar_c.text(
                                    bar_c.get_x() + bar_c.get_width() / 2,
                                    bar_c.get_height() + 0.5,
                                    f"{val_c:.1f}%",
                                    ha="center", va="bottom", fontsize=11,
                                    fontweight="bold", color=clr_c
                                )
                            ax_bar_c.set_ylabel("Mean Confidence (%)", fontsize=9, color="#64748B")
                            ax_bar_c.set_ylim(0, 115)
                            ax_bar_c.set_title("Rata-rata Confidence per Kelas",
                                              fontsize=11, fontweight="bold", color="#1A2340", pad=10)
                            for spine in ax_bar_c.spines.values():
                                spine.set_color("#D4DAE8")
                            ax_bar_c.tick_params(colors="#64748B", labelsize=11)
                            ax_bar_c.grid(axis="y", color="#E8ECF3", linewidth=0.8)
                            ax_bar_c.set_axisbelow(True)

                            plt.tight_layout(pad=2)
                            st.pyplot(fig_conf2, use_container_width=True)
                            st.caption(
                                "Garis horizontal pada strip plot = nilai rata-rata confidence. "
                                "Setiap titik = satu baris data. Kelas dengan titik padat & tinggi "
                                "berarti model sangat yakin pada prediksi tersebut."
                            )

                        # ── CHART: PROBABILITAS HEATMAP + SCATTER 3-AXIS ──────
                        st.markdown("##### Peta Probabilitas Tiga Kelas Per Baris Data")

                        try:
                            prob_r = result_df["P(Rendah)"].astype(float).values
                            prob_s = result_df["P(Sedang)"].astype(float).values
                            prob_t = result_df["P(Tinggi)"].astype(float).values
                            preds  = result_df["Prediksi"].values
                            n_rows = len(result_df)

                            fig_prob_map, axes_pm = plt.subplots(1, 2, figsize=(13, 4.5))
                            fig_prob_map.patch.set_facecolor('#FFFFFF')

                            # LEFT: heatmap-style stacked bar — probability per row
                            ax_hm = axes_pm[0]
                            ax_hm.set_facecolor('#FFFFFF')
                            x_idx_hm = np.arange(n_rows)
                            ax_hm.bar(x_idx_hm, prob_r, color=color_map["Rendah"],
                                     alpha=0.85, label="Rendah", zorder=3)
                            ax_hm.bar(x_idx_hm, prob_s, bottom=prob_r,
                                     color=color_map["Sedang"], alpha=0.85, label="Sedang", zorder=3)
                            ax_hm.bar(x_idx_hm, prob_t, bottom=prob_r + prob_s,
                                     color=color_map["Tinggi"], alpha=0.85, label="Tinggi", zorder=3)
                            ax_hm.set_xlabel("Indeks Baris Data", fontsize=9, color="#64748B")
                            ax_hm.set_ylabel("Probabilitas", fontsize=9, color="#64748B")
                            ax_hm.set_title("Komposisi Probabilitas Tiap Baris (Stacked Bar)",
                                           fontsize=11, fontweight="bold", color="#1A2340", pad=8)
                            ax_hm.set_ylim(0, 1.05)
                            ax_hm.legend(fontsize=8, facecolor="#FFFFFF",
                                        edgecolor="#D4DAE8", labelcolor="#1A2340",
                                        loc="upper right")
                            for spine in ax_hm.spines.values():
                                spine.set_color("#D4DAE8")
                            ax_hm.tick_params(colors="#64748B", labelsize=8)
                            ax_hm.grid(axis="y", color="#E8ECF3", linewidth=0.8)
                            ax_hm.set_axisbelow(True)

                            # RIGHT: P(Rendah) vs P(Tinggi) scatter, size=P(Sedang)
                            ax_sc3 = axes_pm[1]
                            ax_sc3.set_facecolor('#FFFFFF')
                            for kls in ["Rendah","Sedang","Tinggi"]:
                                mask = preds == kls
                                if mask.sum() > 0:
                                    sizes = np.clip(prob_s[mask] * 600 + 30, 30, 400)
                                    ax_sc3.scatter(
                                        prob_r[mask], prob_t[mask],
                                        c=color_map[kls], s=sizes,
                                        alpha=0.80, label=kls,
                                        edgecolors="white", linewidths=0.8, zorder=4
                                    )
                                    # annotate each point with row number
                                    idxs = np.where(mask)[0]
                                    for idx in idxs:
                                        ax_sc3.annotate(
                                            str(idx + 1),
                                            (prob_r[idx], prob_t[idx]),
                                            fontsize=7, color="#64748B",
                                            xytext=(3, 3), textcoords="offset points"
                                        )
                            ax_sc3.set_xlabel("P(Rendah)", fontsize=10, color="#64748B")
                            ax_sc3.set_ylabel("P(Tinggi)", fontsize=10, color="#64748B")
                            ax_sc3.set_title("Scatter P(Rendah) vs P(Tinggi) - ukuran titik = P(Sedang)",
                                            fontsize=11, fontweight="bold", color="#1A2340", pad=8)
                            ax_sc3.set_xlim(-0.05, 1.05)
                            ax_sc3.set_ylim(-0.05, 1.05)
                            ax_sc3.legend(fontsize=8, facecolor="#FFFFFF",
                                         edgecolor="#D4DAE8", labelcolor="#1A2340")
                            for spine in ax_sc3.spines.values():
                                spine.set_color("#D4DAE8")
                            ax_sc3.tick_params(colors="#64748B", labelsize=9)
                            ax_sc3.grid(color="#E8ECF3", linewidth=0.8)
                            ax_sc3.set_axisbelow(True)

                            plt.tight_layout(pad=2)
                            st.pyplot(fig_prob_map, use_container_width=True)
                            st.caption(
                                "Kiri: Setiap batang = satu baris data; tinggi tiap segmen = "
                                "probabilitas kelas tersebut. Kanan: Angka pada titik = nomor baris. "
                                "Titik di sudut kiri-bawah = P(Tinggi) dan P(Rendah) sama-sama rendah "
                                "(P(Sedang) dominan)."
                            )
                        except Exception as e_pm:
                            st.warning(f"Chart probabilitas tidak tersedia: {e_pm}")

                    # ── CONFUSION MATRIX BATCH ────────────────────────────
                    st.markdown("---")
                    st.markdown("#### Confusion Matrix — Performa Model pada Data Test")
                    st.caption(
                        "Confusion matrix berikut menunjukkan performa model yang digunakan "
                        "secara keseluruhan pada data test 20%. Gunakan sebagai referensi "
                        "seberapa andal prediksi batch ini."
                    )

                    from sklearn.model_selection import train_test_split as _tts2
                    from evaluation import evaluate_model as _eval2

                    le_b   = models["le"]
                    cols_b = models["cols"]
                    Xf_b   = df_encoded.drop(columns=["hg/ha_yield","Yield_Class"])
                    yf_b   = le_b.transform(df_encoded["Yield_Class"])
                    for c in cols_b:
                        if c not in Xf_b.columns:
                            Xf_b[c] = 0
                    Xf_b = Xf_b[cols_b]
                    _, Xte_b, _, yte_b = _tts2(
                        Xf_b, yf_b, test_size=0.2, random_state=42, stratify=yf_b
                    )
                    ev_b     = _eval2(selected_model, Xte_b, yte_b)
                    cm_b     = ev_b["confusion_matrix"]
                    cl_b     = list(le_b.classes_)

                    cm_b_col, cm_b_info = st.columns([1.2, 1], gap="large")

                    with cm_b_col:
                        cmap_b = {"Decision Tree":"Blues","Naive Bayes":"Greens"}
                        fig_cmb, ax_cmb = plt.subplots(figsize=(5, 4))
                        fig_cmb.patch.set_facecolor('#FFFFFF')
                        ax_cmb.set_facecolor('#FFFFFF')
                        im_b = ax_cmb.imshow(cm_b, interpolation='nearest',
                                             cmap=cmap_b.get(model_choice,"Blues"), aspect='auto')
                        cb_b = fig_cmb.colorbar(im_b, ax=ax_cmb, fraction=0.046, pad=0.04)
                        cb_b.ax.tick_params(colors='#64748B', labelsize=8)
                        ax_cmb.set_xticks(range(len(cl_b)))
                        ax_cmb.set_yticks(range(len(cl_b)))
                        ax_cmb.set_xticklabels(cl_b, color='#1A2340', fontsize=10)
                        ax_cmb.set_yticklabels(cl_b, color='#1A2340', fontsize=10)
                        ax_cmb.set_xlabel("Predicted Label", color='#64748B', fontsize=9)
                        ax_cmb.set_ylabel("True Label", color='#64748B', fontsize=9)
                        ax_cmb.set_title(f"Confusion Matrix — {model_choice}",
                                        fontsize=11, fontweight='bold', color='#1A2340', pad=10)
                        thresh_b = cm_b.max() / 2
                        for i in range(cm_b.shape[0]):
                            for j in range(cm_b.shape[1]):
                                ax_cmb.text(j, i, f"{cm_b[i,j]:,}",
                                           ha='center', va='center', fontsize=12, fontweight='bold',
                                           color='white' if cm_b[i,j] > thresh_b else '#1A2340')
                        for spine in ax_cmb.spines.values():
                            spine.set_color('#D4DAE8')
                        ax_cmb.tick_params(colors='#64748B')
                        plt.tight_layout(pad=1.5)
                        st.pyplot(fig_cmb, use_container_width=True)

                    with cm_b_info:
                        st.markdown("""
                        <div style='font-family:"DM Mono",monospace; font-size:10px; font-weight:600;
                                    text-transform:uppercase; letter-spacing:1.5px; color:#4F6EF7;
                                    margin-bottom:10px;'>Metrik Model (Data Test 20%)</div>
                        """, unsafe_allow_html=True)
                        mb1, mb2 = st.columns(2)
                        mb1.metric("Akurasi",   f"{ev_b['accuracy']:.2%}")
                        mb2.metric("F1-Score",  f"{ev_b['f1']:.2%}")
                        mb3, mb4 = st.columns(2)
                        mb3.metric("Precision", f"{ev_b['precision']:.2%}")
                        mb4.metric("Recall",    f"{ev_b['recall']:.2%}")

                        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                        st.markdown("""
                        <div style='font-family:"DM Mono",monospace; font-size:10px; font-weight:600;
                                    text-transform:uppercase; letter-spacing:1.5px; color:#4F6EF7;
                                    margin-bottom:10px;'>F1-Score Per Kelas</div>
                        """, unsafe_allow_html=True)
                        for cls_nb, f1_vb in zip(cl_b, ev_b['f1_per_class']):
                            clr_fb = color_map.get(cls_nb, "#4F6EF7")
                            st.markdown(f"""
                            <div style='display:flex; align-items:center; gap:10px; margin-bottom:6px;'>
                                <span style='font-family:"DM Mono",monospace; font-size:11px;
                                             color:{clr_fb}; width:55px;'>{cls_nb}</span>
                                <div style='flex:1; background:#F4F6FB; border-radius:4px;
                                            height:12px; overflow:hidden;'>
                                    <div style='width:{f1_vb*100:.1f}%; background:{clr_fb};
                                                height:100%; opacity:0.8;'></div>
                                </div>
                                <span style='font-family:"DM Mono",monospace; font-size:10px;
                                             color:{clr_fb};'>{f1_vb:.2%}</span>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div style='background:#F4F6FB; border:1px solid #D4DAE8;
                                    border-radius:8px; padding:12px 14px;
                                    font-family:"DM Mono",monospace; font-size:10px;
                                    color:#64748B; line-height:1.8;'>
                            Diagonal = prediksi benar<br>
                            Off-diagonal = kesalahan klasifikasi<br>
                            Nilai besar di diagonal = model andal
                        </div>
                        """, unsafe_allow_html=True)

                    # ── PERBANDINGAN ORIGINAL vs PREDIKSI (BATCH) ─────────
                    st.markdown("---")
                    st.markdown("#### Perbandingan Data Original vs Hasil Prediksi Batch")
                    st.caption(
                        "Tabel berikut menyandingkan data input asli (dari file CSV yang diupload) "
                        "dengan hasil prediksi model. Ditampilkan 5 baris pertama sebagai sampel."
                    )

                    # Kolom original dari batch_df (5 baris)
                    cmp_left, cmp_right = st.columns(2, gap="large")

                    with cmp_left:
                        st.markdown("""
                        <div style='font-family:"DM Mono",monospace; font-size:10px; font-weight:600;
                                    text-transform:uppercase; letter-spacing:1.5px; color:#1A2340;
                                    border-left:3px solid #4F6EF7; padding-left:8px;
                                    margin-bottom:10px;'>Data Original (Input CSV)</div>
                        """, unsafe_allow_html=True)
                        orig_cols_show = [c for c in ["Area","Item","Year",
                                          "average_rain_fall_mm_per_year",
                                          "pesticides_tonnes","avg_temp"]
                                         if c in batch_df.columns]
                        st.dataframe(batch_df[orig_cols_show].head(5),
                                    use_container_width=True, hide_index=True)

                    with cmp_right:
                        st.markdown("""
                        <div style='font-family:"DM Mono",monospace; font-size:10px; font-weight:600;
                                    text-transform:uppercase; letter-spacing:1.5px; color:#0EA88C;
                                    border-left:3px solid #0EA88C; padding-left:8px;
                                    margin-bottom:10px;'>Hasil Prediksi Model</div>
                        """, unsafe_allow_html=True)
                        pred_cols_show = ["No","Area","Item","Prediksi","Confidence",
                                         "P(Rendah)","P(Sedang)","P(Tinggi)"]
                        pred_cols_show = [c for c in pred_cols_show if c in result_df.columns]

                        def _hp(val):
                            cmap2 = {"Rendah":"background:#FFF5F5;color:#E53E3E;font-weight:600",
                                     "Sedang":"background:#FFFBF0;color:#E8960A;font-weight:600",
                                     "Tinggi":"background:#F0FFF9;color:#0EA88C;font-weight:600"}
                            return cmap2.get(val, "")

                        st.dataframe(
                            result_df[pred_cols_show].head(5).style.applymap(
                                _hp, subset=["Prediksi"]
                            ),
                            use_container_width=True, hide_index=True
                        )

                    # Visualization: Original feature vs Prediksi (scatter per item/area)
                    if "Item" in result_df.columns and result_df["Item"].nunique() > 1:
                        st.markdown("##### Distribusi Prediksi per Komoditas (Item)")
                        fig_item, ax_item = plt.subplots(
                            figsize=(10, max(3.5, result_df["Item"].nunique() * 0.5))
                        )
                        fig_item.patch.set_facecolor('#FFFFFF')
                        ax_item.set_facecolor('#FFFFFF')
                        item_dist = result_df.groupby(["Item","Prediksi"]).size().unstack(fill_value=0)
                        for kls in ["Rendah","Sedang","Tinggi"]:
                            if kls not in item_dist.columns:
                                item_dist[kls] = 0
                        item_dist = item_dist[["Rendah","Sedang","Tinggi"]]
                        bot_i = np.zeros(len(item_dist))
                        for kls in ["Rendah","Sedang","Tinggi"]:
                            ax_item.barh(item_dist.index, item_dist[kls],
                                        left=bot_i, color=color_map[kls],
                                        alpha=0.8, label=kls,
                                        edgecolor='white', linewidth=0.6)
                            bot_i += item_dist[kls].values
                        ax_item.set_xlabel("Jumlah Prediksi", fontsize=9, color='#64748B')
                        ax_item.set_title("Distribusi Prediksi per Komoditas",
                                         fontsize=11, fontweight='bold', color='#1A2340', pad=8)
                        for spine in ax_item.spines.values():
                            spine.set_color('#D4DAE8')
                        ax_item.tick_params(colors='#64748B', labelsize=9)
                        ax_item.grid(axis='x', color='#E8ECF3', linewidth=0.8)
                        ax_item.set_axisbelow(True)
                        ax_item.legend(fontsize=9, facecolor='#FFFFFF',
                                      edgecolor='#D4DAE8', labelcolor='#1A2340')
                        plt.tight_layout()
                        st.pyplot(fig_item, use_container_width=True)

                    # Scatter: curah hujan vs suhu, warna = prediksi
                    if "Curah Hujan" in result_df.columns and "Suhu (°C)" in result_df.columns:
                        st.markdown("##### Sebaran Input: Curah Hujan vs Suhu (warna = kelas prediksi)")
                        fig_sc2, ax_sc2 = plt.subplots(figsize=(9, 4))
                        fig_sc2.patch.set_facecolor('#FFFFFF')
                        ax_sc2.set_facecolor('#FFFFFF')
                        for kls in ["Rendah","Sedang","Tinggi"]:
                            sub2 = result_df[result_df["Prediksi"] == kls]
                            if len(sub2) > 0:
                                try:
                                    ax_sc2.scatter(
                                        sub2["Curah Hujan"].astype(float),
                                        sub2["Suhu (°C)"].astype(float),
                                        c=color_map[kls], label=kls,
                                        alpha=0.75, s=70, edgecolors='white',
                                        linewidths=0.8, zorder=3
                                    )
                                except Exception:
                                    pass
                        ax_sc2.set_xlabel("Curah Hujan (mm/thn)", fontsize=10, color='#64748B')
                        ax_sc2.set_ylabel("Suhu (°C)", fontsize=10, color='#64748B')
                        ax_sc2.set_title("Pola Input: Curah Hujan vs Suhu — dikelompokkan per Kelas Prediksi",
                                        fontsize=11, fontweight='bold', color='#1A2340', pad=8)
                        for spine in ax_sc2.spines.values():
                            spine.set_color('#D4DAE8')
                        ax_sc2.tick_params(colors='#64748B', labelsize=9)
                        ax_sc2.grid(color='#E8ECF3', linewidth=0.8)
                        ax_sc2.set_axisbelow(True)
                        ax_sc2.legend(fontsize=9, facecolor='#FFFFFF',
                                     edgecolor='#D4DAE8', labelcolor='#1A2340')
                        plt.tight_layout()
                        st.pyplot(fig_sc2, use_container_width=True)


                    # ── TABEL HASIL LENGKAP ────────────────────────────────
                    st.markdown("---")
                    st.markdown("##### Tabel Hasil Prediksi Lengkap")

                    def highlight_pred(val):
                        cmap = {"Rendah": "background:#FFF5F5;color:#E53E3E;font-weight:600",
                                "Sedang": "background:#FFFBF0;color:#E8960A;font-weight:600",
                                "Tinggi": "background:#F0FFF9;color:#0EA88C;font-weight:600"}
                        return cmap.get(val, "")

                    styled_df = result_df.style.applymap(
                        highlight_pred, subset=["Prediksi"]
                    )
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)

                    st.markdown(f"""
                    <div style='background:#F0FFF9; border:1px solid #0EA88C40;
                                border-radius:8px; padding:12px 16px; margin-top:10px;
                                font-family:"DM Mono",monospace; font-size:11px; color:#64748B;'>
                        <b style='color:#1A2340;'>Statistik Batch</b> &nbsp;|&nbsp;
                        Total: {len(result_df)} baris &nbsp;|&nbsp;
                        Model: {model_choice} &nbsp;|&nbsp;
                        Rendah: {dist.get('Rendah',0)} ({dist.get('Rendah',0)/max(total_valid,1)*100:.1f}%) &nbsp;|&nbsp;
                        Sedang: {dist.get('Sedang',0)} ({dist.get('Sedang',0)/max(total_valid,1)*100:.1f}%) &nbsp;|&nbsp;
                        Tinggi: {dist.get('Tinggi',0)} ({dist.get('Tinggi',0)/max(total_valid,1)*100:.1f}%)
                    </div>
                    """, unsafe_allow_html=True)

                    # ── DOWNLOAD ───────────────────────────────────────────
                    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                    csv_out = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Unduh Hasil Prediksi (.csv)",
                        data=csv_out,
                        file_name=f"hasil_prediksi_{model_choice.lower().replace(' ','_')}.csv",
                        mime="text/csv",
                    )

            except Exception as e:
                st.error(f"Gagal memproses file: {e}")