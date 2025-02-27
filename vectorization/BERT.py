# Векторизация предложений с BERT
# BERT/Sentence-BERT (S-BERT) создает контекстные векторы.
# Используется в продвинутых NLP-задачах, включая поиск и чат-ботов.

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = [
    'Как купить телефон?',
    'Как забронировать отель?',
    'Где посмотреть кино?',
    'Где купить продукты?'
]
vectors = model.encode(sentences)

print(vectors.shape)  # (4, 384) — 4 предложения в 384-мерном пространстве
