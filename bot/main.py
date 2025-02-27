import os

import openai
import asyncio

import spacy
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import Command
from dotenv import load_dotenv
from ollama import chat
from openai import OpenAI
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

load_dotenv()

# Твой API-ключ OpenAI и Telegram
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TELEGRAM_BOT_TOKEN = os.getenv('BOT_TOKEN')

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
client = OpenAI(
    api_key=OPENAI_API_KEY
)


# Функция запроса к OpenAI GPT
async def ask_gpt(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": "Ты опытный Telegram-бот, который отвечает лаконично и полезно."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return response["choices"][0]["message"]["content"]


async def take_topics(message: str) -> list[str]:
    nlp = spacy.load("ru_core_news_sm")
    doc = nlp(message)
    return [token.text for token in doc.ents]


async def ask_llama(topics: list[str], prompt: str) -> str:
    prompt = """
    Ты — умный помощник в Telegram. Твои ответы должны быть:
    1️⃣ Короткими (не более 3 предложений).
    2️⃣ Дружелюбными и простыми.
    3️⃣ Четкими и без сложных терминов.

    Вопрос: {user_message}
    """
    response = chat(
        model='llama3.2',
        messages=[
            {
                'role': 'user',
                'content': f'Ты эксперт в {", ".join(topics)[:-2]}. {prompt}',
            },
            # {
            #     'role': 'user',
            #     'content': prompt.format(user_prompt)
            # }
    ])
    return response['message']['content']


# Обработчик команды /start
@dp.message(Command("start"))
async def start_command(message: Message):
    await message.answer(
        "Привет! Отправь мне любой вопрос, и я попробую ответить с помощью GPT-4.")



@dp.message(Command("help"))
async def help_command(message: Message):
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Как работать с SQL?",
                              callback_data="sql")],
        [InlineKeyboardButton(text="Как создать Telegram-бота?",
                              callback_data="telegram_bot")],
        [InlineKeyboardButton(text="Советы по Python",
                              callback_data="python_tips")]
    ])
    await message.answer("Выбери один из популярных вопросов:",
                         reply_markup=keyboard)


@dp.callback_query()
async def handle_callback(callback: types.CallbackQuery):
    topics = {
        "sql": "Как сделать JOIN в SQL?",
        "telegram_bot": "Как создать Telegram-бота на Python?",
        "python_tips": "Дай советы по написанию чистого кода на Python."
    }
    question = topics.get(callback.data, "Напиши мне свой вопрос!")

    response = await ask_llama(topics=[callback.data], prompt=question)

    await callback.message.answer(response)
    await callback.answer()  # Закрываем всплывающее уведомление


# Обработчик пользовательских сообщений
@dp.message()
async def handle_message(message: Message):
    user_prompt = message.text
    await message.answer("Обрабатываю запрос... ⏳")

    response = await ask_llama(topics=await take_topics(user_prompt), prompt=user_prompt)

    await message.answer(response)


# Запуск бота
async def main():
    print("Бот запущен...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
