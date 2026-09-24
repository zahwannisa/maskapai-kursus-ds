import streamlit as st
import pandas as pd
import joblib

# 1. PERSIAPAN MUAT ARTEFAK DENGAN CACHE
# Menggunakan @st.cache_resource agar artefak hanya dimuat SATU KALI 
# meskipun pengguna berkali-kali mengubah nilai input (menghindari load berulang)
@st.cache_resource
def load_artefak():
    scaler = joblib.load('scaler_maskapai.joblib')
    model_kmeans = joblib.load('kmeans_maskapai.joblib')
    return scaler, model_kmeans

scaler, model_kmeans = load_artefak()

# 2. UI STREAMLIT
st.title("✈️ Prediksi Cluster Delay Maskapai")
st.write("Aplikasi dasbor interaktif untuk memprediksi kelompok penerbangan berdasarkan Waktu dan Jarak.")

st.markdown("---") # Menambahkan garis pembatas

# Membuat layout 2 kolom agar tampilan lebih rapi
col1, col2 = st.columns(2)

# Menggunakan widget input angka
with col1:
    time_val = st.number_input("Time (Waktu Keterlambatan/Penerbangan)", value=100.0)

with col2:
    length_val = st.number_input("Length (Jarak/Durasi Penerbangan)", value=500.0)


# 3. LOGIKA PREDIKSI    
# Tombol untuk triger prediksi
if st.button("Prediksi Cluster"):
    # Bungkus input menjadi DataFrame
    input_data = pd.DataFrame([[time_val, length_val]], columns=['Time', 'Length'])
    # PRA-PEMROSESAN KRUSIAL: Gunakan scaler dari artefak
    scaled_input = scaler.transform(input_data)
    # Lakukan prediksi
    cluster_result = model_kmeans.predict(scaled_input)[0]
    # Tampilkan hasil dengan kotak hijau yang elegan
    st.success(f"Penerbangan ini masuk ke dalam: **Cluster {cluster_result}**")
    # (Opsional) Menampilkan metrik seperti dashboard
    st.metric(label="Status Prediksi", value="Berhasil", delta="Selesai")