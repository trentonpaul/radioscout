import whisper
import ffmpeg
import numpy as np
import torch
import time
import concurrent.futures
from telegram_bot import send_message
from dotenv import load_dotenv
import os
import json

load_dotenv()

class RadioStreamTranscriber:
    def __init__(self, stream_url, key_phrases):
        self.stream_url = stream_url
        self.key_phrases = key_phrases
        self.model = whisper.load_model("tiny.en")
        # Configuration constants
        self.seconds_per_chunk = 10
        self.overlap_seconds = 2
        self.sample_rate = 16000
        self.samples_per_chunk = self.sample_rate * self.seconds_per_chunk
        self.samples_overlap = self.sample_rate * self.overlap_seconds

    def on_phrase_detected(self, phrase, full_text):
        send_message(f"📻 The Mix ALERT: CALL (312-233-1019). '{phrase}' found inside this text: {full_text}")
        print(f"📻 The Mix ALERT: CALL (312-233-1019). '{phrase}' found inside this text: {full_text}")
        # send_message(f"🔥 DETECTED: '{phrase}' inside: {full_text}")
        # print(f"🔥 DETECTED: '{phrase}' inside: {full_text}")

    def get_radio_stream(self, url):
        process = (
            ffmpeg
            .input(url)
            .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar='16000')
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        return process

    def convert_audio_to_numpy(self, audio_bytes):
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0
        return torch.from_numpy(audio)

    def safe_transcribe(self, audio_tensor, timeout=20):
        # Safely transcribe audio with a timeout to prevent getting stuck.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.model.transcribe, audio_tensor)
            try:
                result = future.result(timeout=timeout)
                return result
            except concurrent.futures.TimeoutError:
                print("⚠️ Transcription timeout. Skipping this chunk...")
                return None

    def transcribe_radio_stream(self):
        previous_audio = torch.tensor([])

        process = self.get_radio_stream(self.stream_url)

        try:
            while True:
                # Read audio chunk
                print("⏳ Reading audio chunk...")
                audio_chunk = process.stdout.read(self.sample_rate * 2 * self.seconds_per_chunk)

                if not audio_chunk or len(audio_chunk) < self.sample_rate * 2 * self.seconds_per_chunk:
                    print("⚠️ Stream hiccup detected. Restarting ffmpeg...")
                    process.kill()
                    time.sleep(1)
                    process = self.get_radio_stream(self.stream_url)
                    continue

                current_audio = self.convert_audio_to_numpy(audio_chunk)

                # Concatenate previous overlap with current audio
                combined_audio = torch.cat((previous_audio, current_audio), dim=0)

                print("🔊 Processing audio chunk...")

                result = self.safe_transcribe(combined_audio)

                if result is None:
                    # Skip this chunk if transcription failed
                    continue

                text = result['text'].lower()

                print("📝", text)

                print("🔍 Searching for key phrases...")
                # Detect key phrases
                for phrase in self.key_phrases:
                    if phrase.lower() in text:
                        self.on_phrase_detected(phrase, text)

                # Save last N seconds for next overlap
                if len(current_audio) >= self.samples_overlap:
                    previous_audio = current_audio[-self.samples_overlap:]
                else:
                    previous_audio = current_audio

                time.sleep(0.2)

        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            process.kill()

# Load configuration
stream_url = os.getenv("RADIO_URL")
key_phrases = json.loads(os.getenv("TARGET_PHRASES"))

# Instantiate and run
transcriber = RadioStreamTranscriber(stream_url, key_phrases)
transcriber.transcribe_radio_stream()
