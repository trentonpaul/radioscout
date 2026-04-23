from telegram import Bot
from telegram.error import TimedOut
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()
bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
chat_id = os.getenv("TELEGRAM_CHAT_ID")

def send_message(message):
    asyncio.run(send_notification(message))

async def send_notification(message, retries=5, base_delay=1):
    bot = Bot(token=bot_token)
    for attempt in range(retries):
        try:
            await bot.send_message(chat_id=chat_id, text=message, parse_mode="HTML")
            break  # Success (exit the loop)
        except TimedOut:
            wait_time = base_delay * (2 ** attempt)  # Exponential backoff
            print(f"Attempt {attempt + 1} failed due to timeout. Retrying in {wait_time} seconds...")
            if attempt < retries - 1:
                await asyncio.sleep(wait_time)
            else:
                print("All retry attempts failed.")

