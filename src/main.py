import pyaudio
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
import torch
import queue
import threading
import multiprocessing
import whisper
import ffmpeg
import scipy.signal as signal
from scipy.signal import butter, lfilter
import os
torch.set_num_threads(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def get_audio_interface():
    """Presents available audio interfaces to the user and returns the selected one."""
    audio_interface = pyaudio.PyAudio()

    # Get info of all available audio interfaces
    info = audio_interface.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')

    # Create a list of devices and print them
    devices = []
    for i in range(num_devices):
        device = audio_interface.get_device_info_by_host_api_device_index(0, i)
        if device.get('maxInputChannels') > 0:  # Device supports audio input
            devices.append(device)
            print(f"{len(devices)}. {device.get('name')}")

    # Get user's choice
    choice = int(input('Please choose an audio interface: '))
    device_info = devices[choice - 1]

    # Check if the chosen device supports any of the specified sample rates
    rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 96000]  # Common sample rates
    for rate in rates:
        try:
            if audio_interface.is_format_supported(rate, input_device=device_info['index'], 
                                                   input_channels=1, input_format=pyaudio.paInt16):
                logging.debug(f"Chosen device supports {rate}Hz sample rate.")
                return device_info['index'], rate  # Return the chosen device and the supported rate
        except ValueError as e:
            continue

    print("None of the standard sample rates is supported by the chosen device.")
    print("Please choose a different device.")
    exit(1)

class AudioPlot:
    def __init__(self, rate: int):
        self.RATE = rate
        self.PLOT_WINDOW_SIZE = rate * 15
        self.previous_speech_detected = False

        self.audio_window = np.zeros(self.PLOT_WINDOW_SIZE)
        self.VAD_signal = np.zeros(self.PLOT_WINDOW_SIZE)

        self.init_plots()

    def init_plots(self):
        self.fig, self.ax = plt.subplots()
        self.audio_signal_line, = self.ax.plot(self.audio_window)
        self.VAD_signal_line, = self.ax.plot(self.VAD_signal, color='r')

        self.update_plot(np.zeros(self.PLOT_WINDOW_SIZE), False)
        self.fig.show()

    def update_data(self, chunk, is_speech_detected: bool, change_idx: int = 0):
        self.audio_window = np.roll(self.audio_window, -len(chunk))
        self.audio_window[-len(chunk):] = chunk

        self.VAD_signal = np.roll(self.VAD_signal, -len(chunk))

        if self.previous_speech_detected != is_speech_detected:
            self.VAD_signal[-len(chunk):-change_idx] = self.previous_speech_detected
            self.VAD_signal[-change_idx:] = is_speech_detected
            self.previous_speech_detected = is_speech_detected
        else:
            self.VAD_signal[-len(chunk):] = is_speech_detected

    def update_plot(self, chunk, is_speech_detected: bool, change_idx: int = 0):
        self.update_data(chunk, is_speech_detected, change_idx)
        
        self.ax.set_ylim(-self.audio_window.max(), self.audio_window.max())
        self.audio_signal_line.set_ydata(self.audio_window)
        self.VAD_signal_line.set_ydata(self.VAD_signal * (self.audio_window.max() // 2))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

class SpeechSegment:
    def __init__(self, model: whisper.Whisper, sample_rate):
        self.model = model
        self.sample_rate = sample_rate
        self.chunks: np.NDArray[np.int16] = np.array([])

    def add_chunk(self, chunk):
        # self.chunks = np.append(self.chunks, chunk)
        self.model.embed_audio()

    def get_audio(self):
        return (self.chunks.astype(np.float32) / 32768.0)
    
    def get_duration(self):
        return len(self.chunks) / self.sample_rate
    
    def get_sample_count(self):
        return len(self.chunks)
    
    def convert_to_text(self) -> str | None:

        if len(self.chunks) == 0:
            return None
        
        audio = whisper.pad_or_trim(self.get_audio())
        result = self.model.transcribe(audio)
        return result['text']

    def clear(self):
        self.chunks = np.array([])

def tts_proc_fun(q: multiprocessing.Queue, tts_proc_ready_sem: multiprocessing.Semaphore):
    stt_model = whisper.load_model("tiny.en")
    speech_segment = SpeechSegment(stt_model, sample_rate=16000)
    tts_proc_ready_sem.release()

    while True:
        segment = q.get()
        speech_segment.add_chunk(segment)
        text = speech_segment.convert_to_text()
        speech_segment.clear()
        if text is not None:
            print(text)

def process_audio(q: multiprocessing.Queue, plot_queue: multiprocessing.Queue, tts_proc_q: multiprocessing.Queue, processing_ready_sem: threading.Semaphore):
    """
    Processes audio from the queue.
    """
    vad_model_dir = "./models"
    vad_repo_dirname = "snakers4_silero-vad_master" 
    vad_repo_path = os.path.join(vad_model_dir, vad_repo_dirname)
    if os.path.isdir(vad_repo_path):
        # Load the existing model
        print("Loading model from disk...")
        model, utils = torch.hub.load(repo_or_dir=vad_repo_path,
                                        model='silero_vad',
                                        source='local',
                                        force_reload=False,
                                        onnx=False)
    else:
        # Download the model and save it to disk
        print("Downloading model...")
        torch.hub.set_dir(vad_model_dir)
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=True,
                                      onnx=False)
    
    (_, # get_speech_ts
    _, # save_audio
    _, # read_audio
    VADIterator,
    _) = utils # collect_chunks

    vad_iterator = VADIterator(model, sampling_rate=16000)

    noise_floor: int = 300
    chunk_start_sample = 0
    speech_is_active = False
    absolute_change_idx = 0
    previous_chunk = np.array([])
    segment = np.array([])

    processing_ready_sem.release()

    while True:
        chunk = q.get()

        preprocess_chunk = chunk.copy()
        preprocess_chunk = butter_bandpass_filter(preprocess_chunk, 85, 255, 16000, order=5)
        preprocess_chunk = np.where(np.abs(chunk) > noise_floor, chunk, 0)
        logging.debug(f"Chunk start sample: {chunk_start_sample}")

        # Process the chunk
        speech_dict = vad_iterator(preprocess_chunk)
        if speech_dict is not None:
            if speech_dict.get('start') is not None:
                logging.info("Speech detected @ sample: " + str(speech_dict['start']))
                absolute_change_idx = speech_dict['start']
                speech_is_active = True
                segment = np.append(segment, previous_chunk)
                segment = np.append(segment, chunk)
            elif speech_dict.get('end') is not None:
                logging.info("Speech ended @ sample: " + str(speech_dict['end']))
                absolute_change_idx = speech_dict['end']
                speech_is_active = False
                segment = np.append(segment, chunk)
                tts_proc_q.put(segment)
                segment = np.array([])
        else:
            absolute_change_idx = 0
            if speech_is_active:
                segment = np.append(segment, chunk)

        relative_change_idx = absolute_change_idx - chunk_start_sample
        plot_queue.put((preprocess_chunk, speech_is_active, relative_change_idx))
        chunk_start_sample += len(chunk)
        previous_chunk = chunk

def capture_audio(q: multiprocessing.Queue):
    """
    Captures audio from the specified input device and puts it in a queue.
    """
    audio_interface = pyaudio.PyAudio()
    input_device_info = audio_interface.get_default_input_device_info()
    logging.info(f'Input device info: {input_device_info}')

    def stream_callback(in_data, frame_count, time_info, status_flags):
        q.put(np.frombuffer(in_data, dtype=np.int16))
        return (None, pyaudio.paContinue)

    input_stream = audio_interface.open(
        format=pyaudio.paInt16, channels=1, rate=16000, input=True, 
        frames_per_buffer=8000, input_device_index=input_device_info['index'], 
        stream_callback=stream_callback
    )

    logging.info('Starting stream...')
    input_stream.start_stream()
    while input_stream.is_active():
        time.sleep(1)
    logging.info('Capturing audio...')

def run_audio_plotter(q: multiprocessing.Queue):
    audio_plot = AudioPlot(rate=16000)
    while True:
        chunk, speech_is_active, relative_change_idx = q.get()
        logging.debug(f'Plotting chunk of length {len(chunk)}')
        audio_plot.update_plot(chunk, speech_is_active, relative_change_idx)

if __name__ == '__main__':
    manager = multiprocessing.Manager()
    vad_proc_q = manager.Queue()
    plot_proc_q = manager.Queue()
    tts_proc_q = manager.Queue()

    tts_proc_sem = manager.Semaphore(0)
    tts_proc = multiprocessing.Process(target=tts_proc_fun, args=(tts_proc_q, tts_proc_sem))
    tts_proc.start()
    tts_proc_sem.acquire()

    vad_proc_sem = manager.Semaphore(0)
    vad_proc = multiprocessing.Process(target=process_audio, args=(vad_proc_q, plot_proc_q, tts_proc_q, vad_proc_sem))
    vad_proc.start()
    vad_proc_sem.acquire()

    print('Ready to process audio')
    capture_proc = multiprocessing.Process(target=capture_audio, args=(vad_proc_q,))
    capture_proc.start()
    
    run_audio_plotter(plot_proc_q)
