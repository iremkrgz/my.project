import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Log verilerini pandas DataFrame olarak yükleyin
log_data = pd.read_csv('log_data.csv')

# Log verilerini TF-IDF ile vektörize edin
vectorizer = TfidfVectorizer(stop_words='turkish')  # Türkçe durdurma kelimeleri için 'turkish'
vectorized_data = vectorizer.fit_transform(log_data['text'])

# Kullanıcı sorusuna dayalı olarak ilgili log kayıtlarını getiren fonksiyon
def retrieve_log_records(question):
    # Kullanıcı sorusunu vektörize edin
    question_vector = vectorizer.transform([question])
    
    # Soruyla log kayıtları arasındaki kosinüs benzerliğini hesaplayın
    similarities = cosine_similarity(question_vector, vectorized_data)
    
    # En yüksek benzerliğe sahip ilk N log kaydını alın
    top_n = 5
    indices = np.argsort(-similarities[0])[:top_n]
    return log_data.iloc[indices]

# Alınan log kayıtlarına göre yanıt oluşturan fonksiyon
def generate_response(log_records):
    # Basit bir şablon kullanarak yanıt oluşturun
    response = "Loglarımızdan bazı olası çözümler şunlardır:\n"
    for _, row in log_records.iterrows():
        response += f"- {row['text']}\n"
    return response

# Soru-Cevap sistemini tanımlayan fonksiyon
def qa_system(question):
    # İlgili log kayıtlarını getirin
    log_records = retrieve_log_records(question)
    
    # Alınan log kayıtlarına göre yanıt oluşturun
    response = generate_response(log_records)
    
    return response

# Soru-Cevap sistemini test edin
question = "Bu hatayı nasıl düzeltirim?"
response = qa_system(question)
print(response)
