import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

#  Yuklash
vectorizer = joblib.load("model.pkl")
tfidf_matrix = joblib.load("question_vectors.pkl")
df = pd.read_csv("qa_clean.csv")

# Savolga javob beruvchi funksiya
def get_answer(user_question):
    user_vec = vectorizer.transform([user_question])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    best_match_index = similarity.argmax()

    return df['answers'].iloc[best_match_index]

# Test
print(get_answer("Why do I have elevated glucose levels?"))