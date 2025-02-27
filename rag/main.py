import ollama
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Модель для векторизации текста
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Примеры документов
documents = [
    "Офис работает с 9:00 до 18:00.",
    "Техподдержка доступна по телефону +7 (999) 123-45-67.",
    "Вы можете записаться на встречу через сайт компании."
]

# Векторизация документов
vectors = model.encode(documents)

# Создание FAISS индекса
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

# Сохранение индекса
faiss.write_index(index, "knowledge_base.index")


def search_knowledge_base(query, top_k=1):
    query_vector = model.encode([query])

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

def generate_response_transformers(query):
    found_docs = search_knowledge_base(query)
    llm = pipeline("text-generation", model="utter-project/EuroLLM-1.7B-Instruct")
    prompt = f"""
        Ты — умный помощник. Ответь на вопрос пользователя на основе этой информации:
        {found_docs}

        Вопрос: {query}
        """
    response = llm(prompt, max_length=100)
    return response[0]["generated_text"]


print(generate_response("Нужен телефон техподдержки"))
print(generate_response_transformers("Время работы офиса"))