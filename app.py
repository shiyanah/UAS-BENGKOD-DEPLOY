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
            
    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button("Jalankan Prediksi", use_container_width=True)

# ==========================================
# 4. PREPROCESSING & INFERENSI
# ==========================================
if submit_button:
    df_input = pd.DataFrame([input_data])
    df_input = pd.get_dummies(df_input, drop_first=True)
    df_input_aligned = df_input.reindex(columns=feature_cols, fill_value=0)
    
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