import sounddevice as sd
import numpy as np
import whisper
import webrtcvad
import collections
import contextlib

# Initialize the speech to text model
model = whisper.load_model("base")

# Initialize VAD
vad = webrtcvad.Vad(3)  # set aggressiveness from 0 to 3

# Function to convert frames to text
def frames_to_text(frames):
    mel = whisper.log_mel_spectrogram(frames).to(model.device)
    result = whisper.decode(model, mel, whisper.DecodingOptions())
    return result.text

# A ring buffer is a non-overflowing queue fixed in size
buffer = collections.deque(maxlen=30)

# Parameters for the audio stream
sample_rate = 16000  # Sample rate in Hz
frame_duration_ms = 30  # Duration of a frame in milliseconds
frame_size = int(sample_rate * (frame_duration_ms / 1000.0))  # Size of a frame in samples

# Initialize a buffer for storing frames
frame_buffer = np.array([], dtype=np.int16)

# state 0 waiting for 'hello robot', 1 listening for commands
state = 0
text = ''

# Callback function for the sounddevice stream
def callback(indata, frames, time, status):
    global text, state, frame_buffer
    if status:
        print(status)

     # Ensure the audio is mono and 16-bit PCM
    # indata = np.squeeze(indata)  # This converts `indata` to a 1D array
    indata = np.mean(indata, axis=1)
    indata = (indata * np.iinfo(np.int16).max).astype(np.int16)

    frame_buffer = np.concatenate((frame_buffer, indata))

    while len(frame_buffer) >= frame_size:
        # Take a frame from the buffer
        frame = frame_buffer[:frame_size]
        frame_buffer = frame_buffer[frame_size:]

        # Process the frame with VAD
        vad_res = vad.is_speech(frame.tobytes(), sample_rate=sample_rate)
        volume_norm = np.linalg.norm(frame) * 10
        buffer.append((volume_norm, vad_res))

        if sum([int(b[1]) for b in buffer]) > 0:
            if state == 0:
                text += frames_to_text(indata.flatten().tobytes())
                if 'hello robot' in text:
                    print('Hello Robot Detected')
                    text = ''
                    state = 1
            elif state == 1:
                command = frames_to_text(indata.flatten().tobytes())
                if command:
                    print(command)
                    # Reset state to 0 to listen for the next 'hello robot'
                    state = 0
        else:
            text = ''
            state = 0



# Callback function for the sounddevice stream
# def callback(indata, frames, time, status):
#     global text, state
#     if status:
#         print(status)

#     # Ensure the audio is mono and 16-bit
#     indata = indata.astype(np.int16)
#     if indata.shape[1] != 1:
#         indata = np.mean(indata, axis=1)
#     indata = (indata * np.iinfo(np.int16).max).astype(np.int16)

#     volume_norm = np.linalg.norm(indata) * 10
#     print("Indata length: ", len(indata))
#     vad_res = vad.is_speech(indata.tobytes(), sample_rate=16000)
#     print("VAD result: ", vad_res)
#     buffer.append((volume_norm, vad_res))

#     if sum([int(b[1]) for b in buffer]) > 0:
#         if state == 0:
#             text += frames_to_text(indata.flatten().tobytes())
#             if 'hello robot' in text:
#                 print('Hello Robot Detected')
#                 text = ''
#                 state = 1
#         elif state == 1:
#             command = frames_to_text(indata.flatten().tobytes())
#             if command:
#                 print(command)
#                 # Reset state to 0 to listen for the next 'hello robot'
#                 state = 0
#     else:
#         text = ''
#         state = 0

# Setting up the stream with the callback
with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate):
    print("Listening...")
    while True:  # Keep running indefinitely
        sd.sleep(10000)
