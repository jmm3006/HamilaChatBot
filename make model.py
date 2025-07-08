import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Excel faylni o‘qish
df = pd.read_excel("answer_and_questions.xlsx")  # Faylda 'questions' va 'answers' ustunlari bo‘lishi shart

# Bo‘sh satrlarni olib tashlash
df = df.dropna(subset=['questions', 'answers'])

# TF-IDF modelni o‘rnatish
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['questions'])

# Model va ma’lumotlarni saqlash
joblib.dump(vectorizer, "model.pkl")               # TF-IDF model
joblib.dump(tfidf_matrix, "question_vectors.pkl")  # Savollarni vektor ko‘rinishda

print(" Model, matrix, and dataset exported successfully!")