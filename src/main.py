import pyaudio
import numpy as np
import time

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # Sampling rate
CHUNK = RATE // 2  # 500ms chunks
VAD_THRESHOLD = 1000  # Simple threshold for voice activity detection
HOLDOFF_TIME = 2  # 2 seconds

class AudioProcessing:
    def __init__(self):
        self.audio_interface = pyaudio.PyAudio()
        self.audio_stream = self.audio_interface.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        self.audio_buffer = []
        self.holdoff_start_time = None

    def capture_audio(self):
        return np.fromstring(self.audio_stream.read(CHUNK), dtype=np.int16)

    def transition_condition(self, chunk):
        return np.mean(np.abs(chunk)) > VAD_THRESHOLD

    def process_chunk(self, chunk):
        if self.transition_condition(chunk):
            self.audio_buffer.append(chunk)
            self.holdoff_start_time = time.time()
        elif self.holdoff_start_time:
            self.audio_buffer.append(chunk)
            if time.time() - self.holdoff_start_time > HOLDOFF_TIME:
                self.speech_to_text(self.audio_buffer)
                self.audio_buffer = []
                self.holdoff_start_time = None


    def speech_to_text(self, audio_buffer):
        # Placeholder function
        print('speech_to_text function called')

    def run(self):
        while True:
            chunk = self.capture_audio()
            self.process_chunk(chunk)

if __name__ == '__main__':
    AudioProcessing().run()
