import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ==========================================
# 0. KONFIGURASI HALAMAN UTAMA (Hanya boleh 1 kali & paling atas)
# ==========================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

# ==========================================
# 1. LOAD SAVED OBJECTS
# ==========================================
@st.cache_resource
def load_objects():
    model = joblib.load('model_churn_terbaik.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    selected_features = joblib.load('top_features.pkl')
    return model, scaler, feature_cols, selected_features

try:
    model, scaler, feature_cols, selected_features = load_objects()
except Exception as e:
    st.error(f"Error memuat model atau objek preprocessing: {e}")
    st.stop()

# ==========================================
# 1B. DEFAULT VALUES UNTUK FITUR YANG TIDAK MUNCUL DI FORM
# ==========================================
# Idealnya nilai ini adalah MEDIAN dari dataset training (bukan 0), karena
# mengisi fitur numerik dengan 0 sering kali merepresentasikan "pelanggan
# paling buruk" pada fitur tersebut, yang akan membuat model bias ke arah Churn.
#
# Jika Anda memiliki file 'feature_defaults.pkl' (dict {nama_fitur: median})
# yang disimpan saat training, file tersebut akan otomatis dipakai dan
# nilai di bawah ini hanya berfungsi sebagai fallback.
FALLBACK_DEFAULT_VALUES = {
    'age': 35,
    'total_visits': 10,
    'avg_session_time': 6.0,
    'total_spent': 150.0,
    'days_since_signup': 200,
    'support_tickets': 1,
    'satisfaction_score': 3.5,
    'nps_score': 6,
    'lifetime_value': 600.0,
    'last_3_month_purchase_freq': 3,
    'days_since_last_purchase': 15,
    'is_premium_user': 0,
    # Fitur yang sebelumnya HILANG dari form (akar masalah utama):
    'marketing_spend_per_user': 25.0,
    'avg_order_value': 75.0,
    'email_open_rate': 25.0,
    'email_click_rate': 5.0,
    'pages_per_session': 4.0,
    'delivery_delay_days': 1,
}

@st.cache_resource
def load_default_values():
    """Coba muat median asli dari training (jika ada file opsionalnya).
    Tidak mengubah model/scaler, hanya sumber nilai default untuk fitur
    yang tidak ditanyakan pada form input."""
    try:
        saved_defaults = joblib.load('feature_defaults.pkl')
        merged = dict(FALLBACK_DEFAULT_VALUES)
        merged.update(saved_defaults)
        return merged
    except Exception:
        return dict(FALLBACK_DEFAULT_VALUES)

DEFAULT_VALUES = load_default_values()

# ==========================================
# 2. SIDEBAR & HALAMAN UTAMA
# ==========================================

# Custom CSS untuk mempercantik tampilan tanpa gambar/emoji
st.markdown("""
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 16px;
        color: #6B7280;
        margin-bottom: 30px;
    }
    .result-box-churn {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .result-box-retain {
        background-color: #D1FAE5;
        border-left: 5px solid #10B981;
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Informasi Proyek")
    st.markdown("---")
    st.write("**Mata Kuliah:**")
    st.write("Bengkel Koding - Data Science")
    st.write("**Nama Mahasiswa:**")
    st.write("SHIYANAH")
    st.write("**NIM:**")
    st.write("A11.2023.15459")
    st.markdown("---")
    st.caption("Aplikasi ini menggunakan Machine Learning untuk memprediksi probabilitas pelanggan meninggalkan layanan (Churn).")

# Header Utama
st.markdown('<div class="main-title">Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Sistem Evaluasi dan Prediksi Retensi Pelanggan Berbasis Machine Learning</div>', unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# 3. FORM INPUT DINAMIS
# ==========================================
def is_feature_used(raw_feature_name, selected_list):
    return any(raw_feature_name in f for f in selected_list)

input_data = {}

st.write("Silakan lengkapi parameter metrik pelanggan di bawah ini untuk melakukan prediksi. Form ini hanya menampilkan fitur yang memiliki signifikansi tinggi (Top Features).")

with st.form("input_form"):
    st.markdown("#### Parameter Demografi & Interaksi")
    col1, col2 = st.columns(2)

    with col1:
        if is_feature_used('age', selected_features):
            input_data['age'] = st.number_input("Umur Pelanggan", min_value=15, max_value=100, value=30, step=1)

        if is_feature_used('gender', selected_features):
            input_data['gender'] = st.selectbox("Gender", ["Male", "Female", "Other"])

        if is_feature_used('country', selected_features):
            input_data['country'] = st.selectbox("Negara", ["India", "Germany", "USA", "UK", "Other"])

        if is_feature_used('total_visits', selected_features):
            input_data['total_visits'] = st.number_input("Total Kunjungan", min_value=0, value=5, step=1)

        if is_feature_used('avg_session_time', selected_features):
            input_data['avg_session_time'] = st.number_input("Rata-rata Waktu Sesi (Menit)", min_value=0.0, value=5.0)

        if is_feature_used('total_spent', selected_features):
            input_data['total_spent'] = st.number_input("Total Pengeluaran ($)", min_value=0.0, value=100.0)

        if is_feature_used('days_since_signup', selected_features):
            input_data['days_since_signup'] = st.number_input("Jumlah Hari Sejak Signup", min_value=0, value=100, step=1)

        # --- Fitur tambahan (sebelumnya hilang dari form) ---
        if is_feature_used('marketing_spend_per_user', selected_features):
            input_data['marketing_spend_per_user'] = st.number_input("Biaya Marketing per User ($)", min_value=0.0, value=25.0)

        if is_feature_used('avg_order_value', selected_features):
            input_data['avg_order_value'] = st.number_input("Rata-rata Nilai Order ($)", min_value=0.0, value=75.0)

        if is_feature_used('delivery_delay_days', selected_features):
            input_data['delivery_delay_days'] = st.number_input("Keterlambatan Pengiriman (Hari)", min_value=0, value=1, step=1)

    with col2:
        if is_feature_used('is_premium_user', selected_features):
            is_premium = st.selectbox("Pengguna Premium?", ["Tidak", "Ya"])
            input_data['is_premium_user'] = 1 if is_premium == "Ya" else 0

        if is_feature_used('support_tickets', selected_features):
            input_data['support_tickets'] = st.number_input("Jumlah Tiket Support", min_value=0, value=0, step=1)

        if is_feature_used('satisfaction_score', selected_features):
            input_data['satisfaction_score'] = st.number_input("Skor Kepuasan (1-5)", min_value=1.0, max_value=5.0, value=3.0)

        if is_feature_used('nps_score', selected_features):
            input_data['nps_score'] = st.number_input("NPS Score (0-10)", min_value=0, max_value=10, value=5, step=1)

        if is_feature_used('lifetime_value', selected_features):
            input_data['lifetime_value'] = st.number_input("Lifetime Value", min_value=0.0, value=500.0)

        if is_feature_used('last_3_month_purchase_freq', selected_features):
            input_data['last_3_month_purchase_freq'] = st.number_input("Frekuensi Pembelian 3 Bulan Terakhir", min_value=0, value=2, step=1)

        if is_feature_used('days_since_last_purchase', selected_features):
            input_data['days_since_last_purchase'] = st.number_input("Jumlah Hari Sejak Pembelian Terakhir", min_value=0, value=30, step=1)

        # --- Fitur tambahan (sebelumnya hilang dari form) ---
        if is_feature_used('email_open_rate', selected_features):
            input_data['email_open_rate'] = st.number_input("Email Open Rate (%)", min_value=0.0, max_value=100.0, value=25.0)

        if is_feature_used('email_click_rate', selected_features):
            input_data['email_click_rate'] = st.number_input("Email Click Rate (%)", min_value=0.0, max_value=100.0, value=5.0)

        if is_feature_used('pages_per_session', selected_features):
            input_data['pages_per_session'] = st.number_input("Halaman per Sesi", min_value=0.0, value=4.0)

        if is_feature_used('subscription_type', selected_features):
            input_data['subscription_type'] = st.selectbox("Tipe Langganan", ["Monthly", "Yearly", "Other"])

    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button("Jalankan Prediksi", use_container_width=True)

# ==========================================
# 4. PREPROCESSING & INFERENSI
# ==========================================
def build_model_input(input_data, feature_cols, defaults):
    """
    Membangun satu baris fitur yang identik strukturnya dengan data training:
    1. Mulai dari nilai default (median training jika tersedia) untuk SEMUA
       kolom di feature_columns.pkl -- bukan 0 -- supaya fitur yang tidak
       ditanyakan pada form tidak membuat prediksi bias ke arah Churn.
       Kolom one-hot (dummy) yang tidak punya default eksplisit dimulai dari 0,
       karena itu memang representasi "bukan kategori ini" pada one-hot encoding.
    2. Timpa nilai fitur numerik langsung dari input form.
    3. Untuk fitur kategorikal, set kolom dummy yang sesuai (mis. 'gender_Male')
       menjadi 1 -- dengan cara ini TIDAK menggunakan pd.get_dummies pada
       DataFrame satu baris, yang akan salah karena drop_first menghapus
       seluruh kolom dummy saat hanya ada satu kategori dalam data.
    4. Kembalikan DataFrame dengan urutan kolom PERSIS feature_columns.pkl.
    """
    row = {}
    for col in feature_cols:
        row[col] = defaults.get(col, 0)

    for key, value in input_data.items():
        if key in feature_cols:
            # Fitur numerik / biner yang namanya langsung match dengan feature_columns
            row[key] = value
        else:
            # Fitur kategorikal mentah (mis. gender, country, subscription_type)
            # dicocokkan ke kolom one-hot hasil training, format: "{key}_{value}"
            dummy_col = f"{key}_{value}"
            if dummy_col in feature_cols:
                row[dummy_col] = 1
            # Jika value adalah kategori baseline yang di-drop saat training
            # (drop_first=True), maka memang tidak ada kolom untuk di-set --
            # baris tetap 0 di semua dummy kategori itu, sesuai definisi baseline.

    df_row = pd.DataFrame([row], columns=feature_cols)
    return df_row


if submit_button:
    df_input_aligned = build_model_input(input_data, feature_cols, DEFAULT_VALUES)

    # Urutan pipeline tetap identik dengan training:
    # reindex (sudah terjamin oleh build_model_input) -> scaling -> feature selection -> predict
    input_scaled = scaler.transform(df_input_aligned)
    df_scaled = pd.DataFrame(input_scaled, columns=feature_cols)
    df_final = df_scaled[selected_features]

    prediction = model.predict(df_final)

    st.markdown("---")
    st.markdown("### Hasil Analisis Sistem")

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(df_final)[0][1] * 100

        col_res1, col_res2 = st.columns([2, 1])

        with col_res1:
            if prediction[0] == 1:
                st.markdown("""
                <div class="result-box-churn">
                    <h4 style="color: #991B1B; margin:0;">Status: Berpotensi Churn</h4>
                    <p style="color: #7F1D1D; margin-top:5px; margin-bottom:0;">Sistem mendeteksi risiko tinggi pelanggan akan meninggalkan layanan. Disarankan untuk segera melakukan tindakan retensi, seperti memberikan penawaran khusus atau menghubungi pelanggan.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-box-retain">
                    <h4 style="color: #065F46; margin:0;">Status: Tidak Churn (Aman)</h4>
                    <p style="color: #047857; margin-top:5px; margin-bottom:0;">Sistem memprediksi pelanggan akan tetap bertahan. Pertahankan kualitas layanan untuk menjaga kepuasan pelanggan.</p>
                </div>
                """, unsafe_allow_html=True)

        with col_res2:
            st.metric(label="Probabilitas Churn", value=f"{prob:.2f}%")

    else:
        if prediction[0] == 1:
            st.markdown("""
            <div class="result-box-churn">
                <h4 style="color: #991B1B; margin:0;">Status: Berpotensi Churn</h4>
                <p style="color: #7F1D1D; margin-top:5px; margin-bottom:0;">Sistem mendeteksi risiko tinggi pelanggan akan meninggalkan layanan.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-box-retain">
                <h4 style="color: #065F46; margin:0;">Status: Tidak Churn (Aman)</h4>
                <p style="color: #047857; margin-top:5px; margin-bottom:0;">Sistem memprediksi pelanggan akan tetap bertahan.</p>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9CA3AF; font-size: 12px;'>Sistem Ujian Akhir Semester | Fakultas Ilmu Komputer</p>", unsafe_allow_html=True)