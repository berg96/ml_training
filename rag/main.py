import ollama
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Модель для векторизации текста
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Примеры документов
documents = [
    "Офис работает с 9:00 до 18:00.",
    "Техподдержка доступна по телефону +7 (999) 123-45-67.",
    "Вы можете записаться на встречу через сайт компании."
]

# Векторизация документов
vectors = np.array([model.encode(doc) for doc in documents], dtype="float32")

# Создание FAISS индекса
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

# Сохранение индекса
faiss.write_index(index, "knowledge_base.index")


def search_knowledge_base(query, top_k=1):
    query_vector = np.array([model.encode(query)], dtype="float32")

    # Загружаем FAISS индекс
    index = faiss.read_index("knowledge_base.index")

    # Ищем ближайшие векторы
    distances, indices = index.search(query_vector, top_k)

    # Возвращаем найденные документы
    return [documents[i] for i in indices[0]]


query = "Когда работает офис?"
found_docs = search_knowledge_base(query)
print(found_docs)  # ['Офис работает с 9:00 до 18:00.']


def generate_response(query):
    found_docs = search_knowledge_base(query)

    prompt = f"""
    Ты — умный помощник. Ответь на вопрос пользователя на основе этой информации:
    {found_docs}

    Вопрос: {query}
    """

    response = ollama.chat(model="llama3.2",
                           messages=[{"role": "user", "content": prompt}])
    return response["message"]


print(generate_response("Какое расписание работы офиса?"))
a = input()