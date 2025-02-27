from transformers import pipeline

classifier = pipeline('zero-shot-classification', model="cointegrated/rubert-base-cased-nli-threeway")
categories = ["Покупка", "медицина", "финансы", "путешествие"]
result = classifier('Я хочу купить новый ноутбук.', candidate_labels=categories)
print(result)  # {'sequence': 'Я хочу купить новый ноутбук.', 'labels': ['Покупка', 'финансы', 'путешествие', 'медицина'],
# 'scores': [0.8950983881950378, 0.06867236644029617, 0.02188030444085598, 0.014348926953971386]}
print(classifier('Я полечу на море', candidate_labels=categories))
