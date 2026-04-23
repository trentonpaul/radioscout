# RadioScout

Welcome to **RadioScout**, a simple tool that listens to an online audio stream (such as a radio station) and transcribes it in real time using OpenAI's [Whisper](https://openai.com/index/whisper/).

RadioScout continuously monitors the transcription for specific target phrases and sends instant Telegram alerts when they are detected.

Note: This project is designed to be modular and easily-modifiable to integrate with your own streams, APIs, and preferences.

---

## Requirements

This project depends on the following Python packages:

```bash
pip install openai-whisper ffmpeg-python numpy torch python-dotenv python-telegram-bot
```

In addition, make sure you have **FFmpeg** installed on your system:

- **macOS:**  
  ```bash
  brew install ffmpeg
  ```
- **Ubuntu/Linux:**  
  ```bash
  sudo apt install ffmpeg
  ```
- **Windows:**  
  Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to your system PATH.

> ℹ️ **Note:**  
> `telegram_bot` is a custom module used for sending notifications via Telegram.  
> Ensure your `telegram_bot.py` is properly set up to read your environment variables.

> ⚡ **Performance Tip:**  
> It is highly recommended to run this project with a CUDA-compatible GPU.  
> Whisper transcription is significantly faster on a machine with **GPU acceleration** enabled through PyTorch + CUDA.

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/radioscout.git
   cd radioscout
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Install **FFmpeg** if you haven't already (see [Requirements](#requirements)).

4. Create a `.env` file with your stream URL and Telegram bot credentials (see below).

5. Run the main script:

   ```bash
   python radioscout.py
   ```

---

## Environment Variables

Create a `.env` file in the project root with the following contents:

```dotenv
TELEGRAM_BOT_TOKEN="YOUR TOKEN HERE"
TELEGRAM_CHAT_ID="YOUR TELEGRAM CHAT ID HERE"
RADIO_URL="YOUR STREAM URL HERE"
TARGET_PHRASES='["PHRASE_ONE", "PHRASE_TWO"]'
```

- `TELEGRAM_BOT_TOKEN` — your Telegram bot token from BotFather.
- `TELEGRAM_CHAT_ID` — the chat ID where notifications should be sent.
- `RADIO_URL` — the URL of the online audio stream you want to monitor.
- `TARGET_PHRASES` — a JSON array of phrases to detect in the live transcription.

---

## Example Output

When a target phrase is detected, you will see output similar to:

```text
📝: new music playing now on your favorite station!

🔥 DETECTED: 'PHRASE_ONE' inside: new music playing now on your favorite station!
```

Simultaneously, this message gets transmitted to your Telegram bot.

---

## License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).

---

## Notes

- RadioScout is designed for live-stream audio monitoring and transcription.
- It uses overlapping audio chunks to avoid cutting off words during transcription.
- If the stream connection drops, the system will automatically attempt to reconnect.

---
