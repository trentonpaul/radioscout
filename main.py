import json
import logging
import os
import shutil
import sys
import time
import concurrent.futures

import ffmpeg
import numpy as np
import torch
import whisper
from dotenv import load_dotenv
from telegram_bot import send_message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

load_dotenv()


def load_config():
    stream_url = os.getenv("RADIO_URL")
    if not stream_url:
        raise ValueError("RADIO_URL environment variable is required.")

    raw_phrases = os.getenv("TARGET_PHRASES")
    if not raw_phrases:
        raise ValueError("TARGET_PHRASES environment variable is required.")

    try:
        key_phrases = json.loads(raw_phrases)
    except json.JSONDecodeError as exc:
        raise ValueError("TARGET_PHRASES must be valid JSON.") from exc

    if not isinstance(key_phrases, list) or not all(isinstance(p, str) for p in key_phrases):
        raise ValueError("TARGET_PHRASES must be a JSON array of strings.")

    message_template = os.getenv(
        "MESSAGE_TEMPLATE"
    )
    # Decode escape sequences like \n, \r, \t
    message_template = message_template.encode('utf-8').decode('unicode_escape')

    return stream_url, key_phrases, message_template


def ensure_ffmpeg_available():
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError("ffmpeg is not installed or not available on PATH.")


class RadioStreamTranscriber:
    def __init__(self, stream_url, key_phrases, message_template, model_name="tiny.en"):
        self.stream_url = stream_url
        self.key_phrases = [phrase.lower() for phrase in key_phrases]
        self.message_template = message_template
        self.model_name = model_name
        self.model = self._load_model()
        self.seconds_per_chunk = 10
        self.overlap_seconds = 2
        self.sample_rate = 16000
        self.samples_overlap = self.sample_rate * self.overlap_seconds
        self.process = None
        self.retry_delay = 1
        self.retry_limit = 5

    def _load_model(self):
        try:
            logger.info("Loading Whisper model: %s", self.model_name)
            return whisper.load_model(self.model_name)
        except Exception as exc:
            logger.exception("Failed to load Whisper model")
            raise RuntimeError("Unable to initialize speech model.") from exc

    def on_phrase_detected(self, phrase, full_text):
        message = self.message_template.format(phrase=phrase, text=full_text)
        try:
            send_message(message)
        except Exception:
            logger.exception("Failed to send Telegram alert for phrase '%s'", phrase)
        logger.info(message)

    def get_radio_stream(self, url):
        try:
            logger.info("Starting ffmpeg stream for URL: %s", url)
            process = (
                ffmpeg
                .input(url)
                .output("pipe:", format="wav", acodec="pcm_s16le", ac=1, ar=self.sample_rate)
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )
            return process
        except ffmpeg.Error as exc:
            logger.exception("Failed to start ffmpeg process")
            return None

    def convert_audio_to_numpy(self, audio_bytes):
        try:
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            audio = audio.astype(np.float32) / 32768.0
            return torch.from_numpy(audio)
        except Exception:
            logger.exception("Failed to convert audio bytes to numpy tensor")
            raise

    def safe_transcribe(self, audio_tensor, timeout=20):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.model.transcribe, audio_tensor)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                logger.warning("Transcription timeout. Skipping this chunk.")
                return None
            except Exception:
                logger.exception("Unexpected error during transcription")
                return None

    def restart_stream(self):
        self.close_process()
        for attempt in range(1, self.retry_limit + 1):
            logger.info("Attempt %d/%d to restart stream", attempt, self.retry_limit)
            self.process = self.get_radio_stream(self.stream_url)
            if self.process is not None:
                self.retry_delay = 1
                return True
            time.sleep(self.retry_delay)
            self.retry_delay = min(self.retry_delay * 2, 30)
        return False

    def close_process(self):
        if self.process is None:
            return
        try:
            if self.process.poll() is None:
                self.process.kill()
                logger.info("Killed existing ffmpeg process")
        except Exception:
            logger.exception("Error while killing ffmpeg process")
        finally:
            self.process = None

    def transcribe_radio_stream(self):
        previous_audio = torch.tensor([])
        self.process = self.get_radio_stream(self.stream_url)

        if self.process is None:
            raise RuntimeError("Unable to open radio stream.")

        try:
            while True:
                logger.debug("Reading audio chunk")
                expected_bytes = self.sample_rate * 2 * self.seconds_per_chunk
                audio_chunk = self.process.stdout.read(expected_bytes)

                if not audio_chunk or len(audio_chunk) < expected_bytes:
                    logger.warning("Stream hiccup detected; restarting ffmpeg")
                    if not self.restart_stream():
                        raise RuntimeError("Unable to restart ffmpeg after repeated failures.")
                    previous_audio = torch.tensor([])
                    continue

                current_audio = self.convert_audio_to_numpy(audio_chunk)
                combined_audio = torch.cat((previous_audio, current_audio), dim=0)

                logger.info("Transcribing audio chunk")
                result = self.safe_transcribe(combined_audio)
                if result is None:
                    continue

                text = result.get("text", "").lower()
                logger.info("Transcription result: %s", text)

                logger.debug("Scanning for key phrases")
                for phrase in self.key_phrases:
                    if phrase in text:
                        self.on_phrase_detected(phrase, text)

                previous_audio = current_audio[-self.samples_overlap:] if len(current_audio) >= self.samples_overlap else current_audio
                time.sleep(0.2)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received; stopping")
        except Exception:
            logger.exception("Unexpected failure in transcription loop")
            raise
        finally:
            self.close_process()


if __name__ == "__main__":
    ensure_ffmpeg_available()
    stream_url, key_phrases, message_template = load_config()
    transcriber = RadioStreamTranscriber(stream_url, key_phrases, message_template)
    transcriber.transcribe_radio_stream()
