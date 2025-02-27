# Word2Vec (векторизация с учетом контекста)
# обучает модели, чтобы слова с похожими значениями имели близкие векторы.

from gensim.models import Word2Vec

sentences = [
    ['купить', 'телефон'],
    ['забронировать', 'отель'],
    ['смотреть', 'кино'],
    ['купить', 'продукты'],
]
model = Word2Vec(sentences, vector_size=50, window=5, min_count=1)

vector = model.wv['телефон']
print(vector)  # Векторное представление слова 'телефон'
vector = model.wv['купить']
print(vector)
