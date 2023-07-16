import pyaudio
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
import torch
import queue
import threading
torch.set_num_threads(1)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
    def __init__(self, rate: int, chunk_size: int):
        self.RATE = rate
        self.CHUNK = chunk_size
        self.PLOT_WINDOW_SIZE = rate * 15
        self.previous_speech_detected = False

        self.window = np.zeros(self.PLOT_WINDOW_SIZE)  # For plotting
        self.VAD_signal = np.zeros(self.PLOT_WINDOW_SIZE)  # For plotting

        # Set up plot
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots()
        self.line1, = self.ax.plot(self.window)
        self.line2, = self.ax.plot(self.VAD_signal, color='r')  # VAD plot, in red
        self.update_plot(np.zeros(self.CHUNK), False)  # Initialize plot with zeros


    def update_plot(self, chunk, is_speech_detected: bool, speech_active_change_idx: int = 0):
        # Update x-axis range
        self.window = np.roll(self.window, -len(chunk))
        self.window[-len(chunk):] = chunk

        # roll the VAD signal to the left by the number of samples since the last chunk
        self.VAD_signal = np.roll(self.VAD_signal, -len(chunk))

        # Now update the VAD plot
        # If speech detected state has changed in this chunk, update the plot
        if self.previous_speech_detected != is_speech_detected:
            self.VAD_signal[-len(chunk):speech_active_change_idx] = self.previous_speech_detected
            self.VAD_signal[speech_active_change_idx:] = is_speech_detected
            self.previous_speech_detected = is_speech_detected
        else:
            self.VAD_signal[-len(chunk):] = is_speech_detected
            

        # Update the plotted data
        self.line1.set_ydata(self.window)
        self.line2.set_ydata(self.VAD_signal*(self.window.max() // 2))
        self.ax.set_ylim(-self.window.max(), self.window.max())

        # Draw the new data
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def process_audio(q: queue.Queue, sample_rate: int, chunk_size: int, processing_ready_sem: threading.Semaphore):
    """
    Processes audio from the queue.
    """

    audio_plot = AudioPlot(sample_rate, chunk_size)

    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=False)
    
    (get_speech_timestamps,
    save_audio,
    read_audio,
    VADIterator,
    collect_chunks) = utils

    vad_iterator = VADIterator(model, sampling_rate=sample_rate)

    # model.reset_states()
    # speech_probability_threshold: float = 0.5
    noise_floor: int = 300
    chunk_start_sample = 0
    speech_is_active = False
    speech_active_change_idx = 0

    processing_ready_sem.release()

    while True:
        chunk = q.get()
        chunk = np.where(np.abs(chunk) > noise_floor, chunk, 0)
        logging.debug(f"Chunk start sample: {chunk_start_sample}")

        # Update the indexes in the chunk to be relative to the start of the audio
        np.roll(chunk, -chunk_start_sample)

        # Process the chunk
        speech_dict = vad_iterator(chunk)
        if speech_dict is not None:
            if speech_dict.get('start') is not None:
                logging.debug("Speech detected")
                speech_active_change_idx = speech_dict['start']
                speech_is_active = True
            elif speech_dict.get('end') is not None:
                logging.debug("Speech ended")
                speech_active_change_idx = speech_dict['end']
                speech_is_active = False
            speech_active_change_idx -= chunk_start_sample
        else:
            speech_active_change_idx = 0

        audio_plot.update_plot(chunk, speech_is_active, speech_active_change_idx)
        chunk_start_sample += len(chunk)

def capture_audio(input_device_idx: int, sample_rate: int, chunk_size: int, q: queue.Queue):
    """
    Captures audio from the specified input device and puts it in a queue.
    """
    audio_interface = pyaudio.PyAudio()
    audio_stream = audio_interface.open(
        format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, 
        frames_per_buffer=chunk_size, input_device_index=input_device_idx
    )
    try:
        while True:
            chunk = np.frombuffer(audio_stream.read(chunk_size), dtype=np.int16)
            q.put(chunk)
    finally:
        try:
            audio_stream.stop_stream()
        except OSError:
            pass
        audio_stream.close()
        audio_interface.terminate()

if __name__ == '__main__':
    chosen_interface, rate = get_audio_interface()
    print(f'Chosen interface: {chosen_interface}')
    print(f'Chosen sample rate: {rate}')

    chunk_size = rate // 2 # 500 ms

    q = queue.Queue()
    processing_ready_sem = threading.Semaphore(0)
    processing_thread = threading.Thread(target=process_audio, args=(q, rate, chunk_size, processing_ready_sem))
    processing_thread.start()
    processing_ready_sem.acquire()

    print('Ready to process audio')
    capture_audio(chosen_interface, rate, chunk_size, q)
