import pyaudio
import numpy as np
import time
import logging
import matplotlib.pyplot as plt

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
                                                   input_channels=CHANNELS, input_format=FORMAT):
                logging.debug(f"Chosen device supports {rate}Hz sample rate.")
                return device_info['index'], rate  # Return the chosen device and the supported rate
        except ValueError as e:
            continue

    print("None of the standard sample rates is supported by the chosen device.")
    print("Please choose a different device.")
    exit(1)

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
VAD_THRESHOLD = 1000  # Simple threshold for voice activity detection
HOLDOFF_TIME = 2  # 2 seconds

class AudioProcessing:
    def __init__(self, input_device: int, rate: int):
        """Initializes audio interface and stream. Creates an empty audio buffer."""
        # Constants
        self.RATE = rate
        self.CHUNK = rate // 2  # 500ms chunks
        self.PLOT_WINDOW_SIZE = rate * 15  # 15 seconds of audio data is shown in the plot

        self.audio_interface = pyaudio.PyAudio()
        self.audio_stream = self.audio_interface.open(
            format=FORMAT, channels=CHANNELS, rate=rate, input=True, 
            frames_per_buffer=self.CHUNK, input_device_index=input_device
        )
        self.audio_buffer = []  # Buffer to hold audio frames
        self.holdoff_start_time = None  # Time when the last voice activity was detected

        self.window = np.zeros(self.PLOT_WINDOW_SIZE)  # For plotting
        self.VAD_signal = np.zeros(self.PLOT_WINDOW_SIZE)  # For plotting

        # Set up plot
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots()
        self.line1, = self.ax.plot(self.window)
        self.line2, = self.ax.plot(self.VAD_signal, color='r')  # VAD plot, in red
        self.update_plot([0 * 512])

    def capture_audio(self):
        return np.fromstring(self.audio_stream.read(self.CHUNK), dtype=np.int16)

    def transition_condition(self, chunk):
        return np.mean(np.abs(chunk)) > VAD_THRESHOLD

    def process_chunk(self, chunk):
        if self.transition_condition(chunk):
            logging.debug("Voice activity detected")
            self.audio_buffer.append(chunk)
            self.holdoff_start_time = time.time()
        elif self.holdoff_start_time:
            self.audio_buffer.append(chunk)
            if time.time() - self.holdoff_start_time > HOLDOFF_TIME:
                logging.debug("Speaker has stopped talking")
                self.speech_to_text(self.audio_buffer)
                self.audio_buffer = []
                self.holdoff_start_time = None
        self.update_plot(chunk)

    def update_plot(self, chunk):
        # Update x-axis range
        self.window = np.roll(self.window, -len(chunk))
        self.window[-len(chunk):] = chunk

        # Update the plotted data
        self.line1.set_ydata(self.window)
        self.line2.set_ydata(self.VAD_signal*(self.window.max() // 2))
        self.ax.set_ylim(-self.window.max(), self.window.max())

        # Now update the VAD plot
        self.VAD_signal = np.roll(self.VAD_signal, -len(chunk))
        if self.holdoff_start_time:
            self.VAD_signal[-len(chunk):] = 1
        else:
            self.VAD_signal[-len(chunk):] = 0

        # Draw the new data
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def speech_to_text(self, audio_buffer):
        # Placeholder function
        print('speech_to_text function called')

    def run(self):
        while True:
            chunk = self.capture_audio()
            logging.debug('Captured audio chunk')
            self.process_chunk(chunk)

if __name__ == '__main__':
    chosen_interface, rate = get_audio_interface()
    AudioProcessing(chosen_interface, rate).run()
