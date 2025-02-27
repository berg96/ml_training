from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = ['Купить телефон', 'Забронировать отель', 'Смотреть кино']
labels = [0, 1, 2]  # 0 - покупки, 1 - путешествия, 2 - развлечения

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

new_text = ['Заказать смартфон']
prediction = model.predict(vectorizer.transform(new_text))
print(prediction)  # [0] → 'покупки'
print(model.predict(vectorizer.transform(['Купить продукты'])))
print(model.predict(vectorizer.transform(['Смотреть в окно'])))
model.partial_fit(vectorizer.transform(['Полететь на самолете']), 1)
print(model.predict(vectorizer.transform(['Поехать на море'])))
