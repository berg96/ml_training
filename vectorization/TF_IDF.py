# TF-IDF (Term Frequency - Inverse Document Frequency) —
# учитывает, как часто слово встречается в документе и в корпусе в целом.

from sklearn.feature_extraction.text import TfidfVectorizer

texts = ['Купить телефон', 'Забронировать отель', 'Смотреть кино', 'Купить продукты']
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

print(vectorizer.get_feature_names_out())  # ['забронировать', 'кино', 'купить', 'отель', 'смотреть', 'телефон']
print(X.toarray())  # Векторное представление текста
