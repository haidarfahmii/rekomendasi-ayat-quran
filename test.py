import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import braycurtis, jaccard
from tqdm import tqdm

def load_data(file_path='./data_clean_v1.csv'):
    # Baca file CSV
    final = pd.read_csv(file_path)
    final['clean_text'] = final['clean_text'].fillna('')

    # TF-IDF Vectorizer
    vsm = TfidfVectorizer(lowercase=True, smooth_idf=True, sublinear_tf=True,
                          ngram_range=(1, 2), max_df=0.90, min_df=2)

    quran = vsm.fit_transform(final['clean_text'])

    return final, quran, vsm

def generate_recommendations_from_feedback(feedback_file, method='cosine', file_path='./data_clean_v1.csv', top_n=5):
    import os

    # Baca data keluh kesah mahasiswa
    df_feedback = pd.read_csv(feedback_file)

    # Pastikan kolom yang sesuai
    if 'keluh_kesah' not in df_feedback.columns:
        raise ValueError("Kolom 'keluh_kesah' tidak ditemukan di file survei.")

    # Load data Quran dan TF-IDF sekali saja
    final, quran, vsm = load_data(file_path)

    all_recommendations = []

    for idx, row in tqdm(df_feedback.iterrows(), total=len(df_feedback)):
        keluh = row['keluh_kesah']
        input_vector = vsm.transform([keluh])

        if method == 'cosine':
            scores = [cosine_similarity(input_vector, quran[i])[0][0] for i in range(quran.shape[0])]
            final['Skor'] = scores
            sorted_result = final[['surah', 'ayat', 'short', 'Skor']].sort_values(by='Skor', ascending=False).head(top_n)
        elif method == 'braycurtis':
            scores = [braycurtis(input_vector.toarray()[0], quran[i].toarray()[0]) for i in range(quran.shape[0])]
            final['Skor'] = scores
            sorted_result = final[['surah', 'ayat', 'short', 'Skor']].sort_values(by='Skor', ascending=True).head(top_n)
        elif method == 'jaccard':
            scores = [jaccard(input_vector.toarray()[0], quran[i].toarray()[0]) for i in range(quran.shape[0])]
            final['Skor'] = scores
            sorted_result = final[['surah', 'ayat', 'short', 'Skor']].sort_values(by='Skor', ascending=True).head(top_n)
        else:
            raise ValueError("Metode tidak dikenali.")

        # Tambahkan kolom 'keluh_kesah' dan index baris
        sorted_result['keluh_kesah'] = keluh
        sorted_result['id_keluhan'] = idx
        sorted_result['metode'] = method

        all_recommendations.append(sorted_result)

    # Gabungkan seluruh hasil
    final_recommendation_df = pd.concat(all_recommendations, ignore_index=True)

    return final_recommendation_df


# File berisi keluhan mahasiswa
feedback_file = './keluh_kesah.csv'

# Metode yang diinginkan: jaccard
hasil = generate_recommendations_from_feedback(feedback_file, method='jaccard')

# Tampilkan hasil
print(hasil[["surah", "ayat", "Skor"]])