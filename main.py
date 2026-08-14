import datetime
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
    """Load and validate configuration from environment variables.

    Returns:
        tuple: (stream_url:str, key_phrases:list[str], message_template:str, parsed_times:list[datetime.time], padding_minutes:float)
    """
    stream_url = os.getenv("RADIO_URL")
    if not stream_url:
        raise ValueError("RADIO_URL environment variable is required and cannot be empty.")

    raw_phrases = os.getenv("TARGET_PHRASES", "[]")
    # Accept either JSON array or comma-separated string
    key_phrases = None
    if raw_phrases.strip().startswith("["):
        try:
            key_phrases = json.loads(raw_phrases)
        except json.JSONDecodeError as exc:
            raise ValueError("TARGET_PHRASES starts with '[' so must be valid JSON array of strings.") from exc
    else:
        # comma separated
        key_phrases = [p.strip() for p in raw_phrases.split(",") if p.strip()]

    if not isinstance(key_phrases, list) or not all(isinstance(p, str) for p in key_phrases):
        raise ValueError("TARGET_PHRASES must be an array of strings or a comma-separated list of phrases.")

    raw_scheduled_times = os.getenv("PROCESS_TIMES", "")
    if not raw_scheduled_times:
        parsed_times = []
    else:
        try:
            if raw_scheduled_times.strip().startswith("["):
                scheduled_times = json.loads(raw_scheduled_times)
            else:
                scheduled_times = [t.strip() for t in raw_scheduled_times.split(",") if t.strip()]
        except json.JSONDecodeError as exc:
            raise ValueError("PROCESS_TIMES must be JSON array or comma-separated list of HH:MM strings.") from exc

        if not isinstance(scheduled_times, list) or not all(isinstance(t, str) for t in scheduled_times):
            raise ValueError("PROCESS_TIMES must be a JSON array of strings or a comma-separated list of HH:MM values.")

        parsed_times = []
        for time_str in scheduled_times:
            try:
                parsed_times.append(datetime.datetime.strptime(time_str, "%H:%M").time())
            except ValueError as exc:
                raise ValueError(f"PROCESS_TIMES entries must use HH:MM format. Invalid value: {time_str}") from exc

    raw_padding = os.getenv("PROCESS_PADDING_MINUTES", "5")
    try:
        padding_minutes = float(raw_padding)
    except ValueError as exc:
        raise ValueError("PROCESS_PADDING_MINUTES must be a number.") from exc

    if padding_minutes < 0:
        raise ValueError("PROCESS_PADDING_MINUTES must be a non-negative value.")

    message_template = os.getenv("MESSAGE_TEMPLATE", "{phrase}: {text}")
    # Decode escape sequences like \n, \r, \t for user-friendly config
    try:
        message_template = message_template.encode("utf-8").decode("unicode_escape")
    except Exception:
        # If decoding fails, fall back to raw value
        pass

    return stream_url, key_phrases, message_template, parsed_times, padding_minutes


def ensure_ffmpeg_available():
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError("ffmpeg is not installed or not available on PATH.")


class RadioStreamTranscriber:
    def __init__(self, stream_url, key_phrases, message_template, scheduled_times, padding_minutes, model_name="base.en"):
        self.stream_url = stream_url
        self.key_phrases = [phrase.lower() for phrase in key_phrases]
        self.message_template = message_template
        self.scheduled_times = scheduled_times
        self.padding_minutes = padding_minutes
        self.model_name = model_name
        self.model = self._load_model()
        self.seconds_per_chunk = 10
        self.overlap_seconds = 2
        self.sample_rate = 16000
        self.samples_overlap = int(self.sample_rate * self.overlap_seconds)
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

    def is_within_processing_window(self, now=None):
        if now is None:
            now = datetime.datetime.now()

        current_datetime = now
        padding_delta = datetime.timedelta(minutes=self.padding_minutes)

        for scheduled_time in self.scheduled_times:
            for day_offset in (0, -1, 1):
                scheduled_datetime = datetime.datetime.combine(
                    now.date() + datetime.timedelta(days=day_offset),
                    scheduled_time,
                )
                if abs(current_datetime - scheduled_datetime) <= padding_delta:
                    return True

        return False

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
            # PCM16 little-endian -> float32 in [-1, 1]
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            return audio
        except Exception:
            logger.exception("Failed to convert audio bytes to numpy tensor")
            raise

    def safe_transcribe(self, audio_tensor, timeout=20):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Whisper accepts numpy arrays or file paths; pass numpy audio directly
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
        # Main loop: only open and read the ffmpeg stream during scheduled windows.
        byte_buffer = bytearray()
        previous_audio = np.zeros((0,), dtype=np.float32)
        expected_bytes = int(self.sample_rate * 2 * self.seconds_per_chunk)  # 2 bytes per sample (pcm16)

        try:
            while True:
                # If outside any processing window, ensure stream is closed and wait.
                if not self.is_within_processing_window():
                    if self.process is not None:
                        logger.info("Outside processing window; closing stream to conserve resources")
                        self.close_process()
                    # Poll until a window opens; sleep to avoid busy loop
                    while not self.is_within_processing_window():
                        time.sleep(5)
                    # Once window opens, continue to open stream below

                # Ensure the ffmpeg process is started for the active window
                if self.process is None:
                    self.process = self.get_radio_stream(self.stream_url)
                    if self.process is None:
                        logger.warning("Unable to open stream at window start; retrying in 5s")
                        time.sleep(5)
                        continue
                    byte_buffer.clear()
                    previous_audio = np.zeros((0,), dtype=np.float32)

                # Read and process frames while still inside the processing window
                while self.is_within_processing_window():
                    try:
                        chunk = self.process.stdout.read(4096)
                    except Exception:
                        chunk = None

                    if not chunk:
                        logger.warning("Stream hiccup detected; restarting ffmpeg")
                        if not self.restart_stream():
                            raise RuntimeError("Unable to restart ffmpeg after repeated failures.")
                        byte_buffer.clear()
                        previous_audio = np.zeros((0,), dtype=np.float32)
                        continue

                    byte_buffer.extend(chunk)

                    # Process complete frames from buffer
                    while len(byte_buffer) >= expected_bytes:
                        # If the window ended mid-processing, break to outer loop to close stream
                        if not self.is_within_processing_window():
                            logger.debug("Window ended during frame processing; breaking to outer loop")
                            break

                        frame_bytes = bytes(byte_buffer[:expected_bytes])
                        del byte_buffer[:expected_bytes]

                        current_audio = self.convert_audio_to_numpy(frame_bytes)

                        # Prepend overlap from previous frame
                        if previous_audio.size > 0:
                            combined_audio = np.concatenate((previous_audio, current_audio), axis=0)
                        else:
                            combined_audio = current_audio

                        logger.info("Transcribing audio frame (%.2fs)", len(combined_audio) / float(self.sample_rate))
                        result = self.safe_transcribe(combined_audio)
                        if result is None:
                            previous_audio = current_audio[-self.samples_overlap:] if len(current_audio) >= self.samples_overlap else current_audio
                            continue

                        text = result.get("text", "").lower()
                        logger.info("Transcription result: %s", text)

                        logger.debug("Scanning for key phrases")
                        for phrase in self.key_phrases:
                            if phrase in text:
                                self.on_phrase_detected(phrase, text)

                        # Save last overlap samples for next combined frame
                        previous_audio = current_audio[-self.samples_overlap:] if len(current_audio) >= self.samples_overlap else current_audio

                    # small sleep to yield
                    time.sleep(0.01)

                # end of window: loop will close process at top

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received; stopping")
        except Exception:
            logger.exception("Unexpected failure in transcription loop")
            raise
        finally:
            self.close_process()


if __name__ == "__main__":
    ensure_ffmpeg_available()
    stream_url, key_phrases, message_template, scheduled_times, padding_minutes = load_config()
    transcriber = RadioStreamTranscriber(
        stream_url,
        key_phrases,
        message_template,
        scheduled_times,
        padding_minutes,
    )
    transcriber.transcribe_radio_stream()
