from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import numpy as np

texts = ['Купить телефон', 'Забронировать отель', 'Смотреть кино']
labels = [0, 1, 2]  # 0 - покупки, 1 - путешествия, 2 - развлечения

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = SGDClassifier(loss='log_loss')  # Используем логистическую регрессию
model.fit(X, labels)


test_text = ['Где купить смартфон?']
test_vector = vectorizer.transform(test_text)
prediction = model.predict(test_vector)
print(prediction)  # [0] → покупки
print(model.predict(vectorizer.transform(['Купить продукты'])))
print(model.predict(vectorizer.transform(['Смотреть в окно'])))

# Дообучаем модель
model.partial_fit(
    vectorizer.transform(['Полететь на самолете']),
    [1],
    classes=np.array([0, 1, 2])
)

print(model.predict(vectorizer.transform(['Поехать на море'])))
