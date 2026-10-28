import time
import pandas as pd
import streamlit as st
from groq import Groq

from classifier import NAMA_FITUR, classify, load_config

# ---------- 0. Page Config ----------
st.set_page_config(
    page_title="Prediksi Status Akademik (LLM)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CSS Styling: Clean, Minimalist, Enterprise Typography ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #0f172a;
}

/* Header Section */
.header-container {
    margin-bottom: 28px;
    padding-bottom: 20px;
    border-bottom: 1px solid #e2e8f0;
}
.system-tag {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #475569;
    background-color: #f1f5f9;
    border: 1px solid #e2e8f0;
    padding: 3px 10px;
    border-radius: 4px;
    margin-bottom: 8px;
}
.app-heading {
    font-size: 1.85rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.025em;
    margin: 0 0 6px 0;
}
.app-subheading {
    font-size: 0.95rem;
    color: #64748b;
    margin: 0;
    line-height: 1.5;
}

/* Card Container */
.clean-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 20px 24px;
    margin-top: 18px;
    margin-bottom: 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}

/* Custom Status Pills */
.status-pill {
    display: inline-flex;
    align-items: center;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: -0.01em;
}
.pill-aman {
    background-color: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
}
.pill-waspada {
    background-color: #fffbeb;
    color: #92400e;
    border: 1px solid #fde68a;
}
.pill-berisiko {
    background-color: #fef2f2;
    color: #991b1b;
    border: 1px solid #fecaca;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #f8fafc;
    border-right: 1px solid #e2e8f0;
}

/* Sleek Buttons */
.stButton>button {
    background-color: #0f172a !important;
    color: #ffffff !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border: none !important;
    padding: 8px 18px !important;
    transition: all 0.15s ease-in-out !important;
}
.stButton>button:hover {
    background-color: #1e293b !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.08) !important;
}

/* Clean Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
    border-bottom: 1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 500;
    color: #64748b;
    padding: 10px 4px;
    border: none;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    font-weight: 700 !important;
    color: #0f172a !important;
    border-bottom: 2px solid #0f172a !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- 1. Muat Konfigurasi Evaluasi & Siapkan Client ----------
cfg = load_config("config.json")
FITUR = list(NAMA_FITUR)

try:
    API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("Kredensial GROQ_API_KEY belum dikonfigurasi pada environment.")
    st.stop()

@st.cache_resource
def get_client():
    return Groq(api_key=API_KEY)

client = get_client()

@st.cache_data(show_spinner=False)
def prediksi(ipk, kehadiran, jam_belajar, tugas_terkumpul, nilai_uts, aktif_organisasi):
    data = {
        "ipk": ipk, "kehadiran": kehadiran, "jam_belajar": jam_belajar,
        "tugas_terkumpul": tugas_terkumpul, "nilai_uts": nilai_uts,
        "aktif_organisasi": aktif_organisasi
    }
    return classify(client, cfg, data)

# ---------- 2. Sidebar: Spesifikasi Model & Parameter Kontrol ----------
ev = cfg["evaluasi"]
with st.sidebar:
    st.markdown("### Spesifikasi Model")
    st.markdown(f"**Arsitektur:** `{cfg['model']}`")
    st.markdown(f"**Penyedia:** Groq Cloud LPU")
    st.markdown(f"**Few-Shot Prompt:** {len(cfg['few_shot'])} contoh evaluasi")
    
    st.markdown("---")
    st.markdown("### Metrik Data Uji (Test Set)")
    st.metric("F1-Score Macro", f"{ev['f1_macro']:.3f}")
    if "f1_dnn_pembanding" in ev:
        st.metric("F1 Macro DNN (Pembanding)", f"{ev['f1_dnn_pembanding']:.3f}")
    
    st.caption(f"Dievaluasi pada {ev['n']} sampel data uji terkontrol ({ev['tanggal']}).")
    
    with st.expander("Konfigurasi System Prompt"):
        st.code(cfg["system_prompt"], language=None)

# ---------- 3. Header Utama ----------
st.markdown("""
<div class="header-container">
    <div class="system-tag">AI Model Service • Modul 2B</div>
    <h1 class="app-heading">Klasifikasi Status Akademik Mahasiswa</h1>
    <p class="app-subheading">Sistem inferensi prediktif status akademik berbasis Large Language Model (Groq LLM Qwen 3.8 27B).</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Evaluasi Mahasiswa Tunggal", "Pemrosesan Batch CSV"])

# ---------- 4. Tab 1: Prediksi Data Tunggal ----------
with tab1:
    st.markdown("#### Formulir Parameter Akademik")
    st.caption("Masukkan indikator performa perkuliahan mahasiswa untuk melakukan evaluasi otomatis.")
    
    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)
        with col1:
            ipk = st.number_input("Indeks Prestasi Kumulatif (IPK)", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
            kehadiran = st.slider("Persentase Kehadiran Kuliah (%)", 0, 100, 80)
            jam_belajar = st.slider("Alokasi Jam Belajar Mandiri (Jam/Minggu)", 0, 40, 8)
        with col2:
            tugas = st.slider("Rasio Pengumpulan Tugas (%)", 0, 100, 80)
            uts = st.slider("Nilai Ujian Tengah Semester (UTS)", 0, 100, 70)
            organisasi = st.radio("Partisipasi Kegiatan Organisasi", ["Ya", "Tidak"], horizontal=True)
            
        kirim = st.form_submit_button("Jalankan Evaluasi Prediksi", use_container_width=True)

    if kirim:
        with st.spinner("Memproses inferensi via Groq API..."):
            hasil = prediksi(ipk, kehadiran, jam_belajar, tugas, uts, organisasi)
        
        status_class = "pill-aman" if hasil == "Aman" else ("pill-waspada" if hasil == "Waspada" else "pill-berisiko")
        desc_text = (
            "Performa akademik berada pada jalur aman dan memenuhi syarat kelulusan standar."
            if hasil == "Aman" else (
                "Performa akademik memerlukan pemantauan berkala guna mencegah penurunan indeks prestasi."
                if hasil == "Waspada" else
                "Indikator akademik berada di bawah ambang batas standar, memerlukan intervensi bimbingan akademik intensif."
            )
        )
        
        st.markdown(f"""
        <div class="clean-card">
            <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; margin-bottom: 8px;">Hasil Klasifikasi Model</div>
            <div style="margin-bottom: 12px;">
                <span class="status-pill {status_class}">Status: {hasil}</span>
            </div>
            <div style="font-size: 0.9rem; color: #334155; line-height: 1.5;">{desc_text}</div>
        </div>
        """, unsafe_allow_html=True)

# ---------- 5. Tab 2: Prediksi Batch CSV ----------
with tab2:
    st.markdown("#### Evaluasi Batch Data Mahasiswa")
    st.caption("Unggah berkas CSV berisi kolom: " + ", ".join(f"`{c}`" for c in FITUR) + " (dibatasi 30 baris per transaksi).")
    
    file = st.file_uploader("Pilih Berkas CSV", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        kurang = [c for c in FITUR if c not in df.columns]
        if kurang:
            st.error(f"Format tidak sesuai. Kolom berikut tidak ditemukan: {', '.join(kurang)}")
        else:
            df = df.head(30)
            st.markdown(f"**Pratinjau Data ({len(df)} baris terdeteksi):**")
            st.dataframe(df.head(5), use_container_width=True)
            
            if st.button("Jalankan Prediksi Batch", use_container_width=True):
                bar = st.progress(0.0)
                hasil = []
                for i, (_, b) in enumerate(df.iterrows(), start=1):
                    hasil.append(prediksi(
                        float(b["ipk"]), int(b["kehadiran"]), int(b["jam_belajar"]),
                        int(b["tugas_terkumpul"]), int(b["nilai_uts"]), str(b["aktif_organisasi"])
                    ))
                    bar.progress(i / len(df))
                    time.sleep(0.5)
                
                df["Prediksi_Status"] = hasil
                
                st.markdown("---")
                st.markdown("#### Tabel Hasil Klasifikasi")
                st.dataframe(df, use_container_width=True)
                
                col_r1, col_r2 = st.columns([1, 2])
                with col_r1:
                    st.markdown("**Distribusi Frekuensi:**")
                    st.dataframe(df["Prediksi_Status"].value_counts(), use_container_width=True)
                
                with col_r2:
                    csv_data = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Unduh Hasil Evaluasi (CSV)",
                        csv_data,
                        file_name="hasil_evaluasi_status_akademik.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
