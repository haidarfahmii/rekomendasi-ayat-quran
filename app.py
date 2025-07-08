import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import jaccard
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =====================================================================================
# FUNGSI-FUNGSI UTAMA (DI LUAR BAGIAN UI)
# =====================================================================================

# 1. Fungsi untuk memuat data & melatih model (hanya berjalan sekali)
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv('./data_clean_v1.csv')
    df['clean_text'] = df['clean_text'].fillna('')
    vsm = TfidfVectorizer(
        lowercase=True,
        smooth_idf=True,
        sublinear_tf=True,
        ngram_range=(1, 2),
        max_df=0.90,
        min_df=2
    )
    quran_matrix = vsm.fit_transform(df['clean_text'])
    return vsm, quran_matrix, df

# 2. Fungsi untuk membersihkan dan memproses teks input baru
@st.cache_resource # Cache stemmer agar tidak dibuat berulang kali
def get_stemmer():
    return StemmerFactory().create_stemmer()

def preprocess_text(text, stemmer):
    """
    Fungsi ini menerapkan semua langkah cleaning dari notebook ke teks baru.
    """
    # Case folding dan cleaning dasar
    temp = text.lower()
    temp = re.sub("@[A-Za-z0-9_]+",'', temp)
    temp = re.sub("#[A-Za-z0-9_]+",'', temp)
    temp = re.sub(r'http\\S+', ' ', temp)
    temp = re.sub('[()!?]', '', temp)
    temp = re.sub('\\[.*?\\]','', temp)
    temp = re.sub("[^a-z0-9\\s]",'', temp)
    temp = re.sub(r'[0-9]',' ', temp)
    
    # Stemming (Anda bisa tambahkan stopword removal di sini jika perlu)
    stemmed_text = stemmer.stem(temp)
    return stemmed_text

# =====================================================================================
# BAGIAN UTAMA APLIKASI (USER INTERFACE)
# =====================================================================================

st.set_page_config(
    page_title="Sistem Rekomendasi Ayat Al-Qur'an",
    page_icon="📖",
    layout="wide"
)

st.title("📖 Sistem Rekomendasi Ayat Al-Qur'an")
st.markdown("Aplikasi ini membantu Anda menemukan ayat Al-Qur'an yang relevan dengan pertanyaan atau curhatan Anda menggunakan metode *Jaccard Similarity*.")

# Muat model dan data yang sudah di-cache
try:
    with st.spinner("Model sedang disiapkan, mohon tunggu..."):
        vsm_model, quran_matrix, df_quran = load_and_train_model()
        stemmer = get_stemmer()
    st.success("Model siap digunakan!", icon="✅")
except FileNotFoundError:
    st.error("File 'data_clean_v1.csv' tidak ditemukan. Pastikan file tersebut berada di folder yang sama dengan aplikasi.")
    st.stop() # Hentikan eksekusi jika file tidak ada

# Input dari pengguna
user_input = st.text_area(
    "Tuliskan pertanyaan atau curhatanmu di sini:",
    height=150,
    placeholder="Contoh: Mengapa saya selalu gagal padahal sudah berusaha keras?"
)

if st.button("🔍 Cari Ayat"):
    if user_input:
        with st.spinner("Sedang memproses dan mencari ayat yang relevan..."):
            # 1. Bersihkan input pengguna menggunakan fungsi preprocess_text
            clean_input = preprocess_text(user_input, stemmer)
            
            # 2. Ubah input bersih menjadi vektor menggunakan model yang sudah ada
            input_vector = vsm_model.transform([clean_input])
            
            # 3. Hitung Jaccard distance
            scores = [jaccard(input_vector.toarray()[0], quran_matrix[i].toarray()[0]) for i in range(quran_matrix.shape[0])]
            
            # 4. Simpan skor ke DataFrame dan urutkan
            df_quran['Skor'] = scores
            results_df = df_quran.sort_values(by='Skor', ascending=True).head(5)

            # 5. Tampilkan hasil
            st.subheader("✨ Rekomendasi Ayat Untuk Anda:")
            for index, row in results_df.iterrows():
                with st.container(border=True):
                    st.markdown(f"### **Surah {row['surah']} Ayat {row['ayat']}**")
                    st.markdown(f"> _{row['short']}_")
                    st.success(f"**Skor Jaccard:** {row['Skor']:.4f} (Semakin besar semakin relevan)")
    else:
        st.warning("Mohon masukkan pertanyaan atau curhatan Anda terlebih dahulu.")