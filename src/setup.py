import torch
torch.set_num_threads(1)
from IPython.display import Audio
from pprint import pprint

SAMPLING_RATE = 16000

# download example
torch.hub.download_url_to_file('https://models.silero.ai/vad_models/en.wav', 'en_example.wav')

USE_ONNX = False # change this to True if you want to test onnx model
# if USE_ONNX:
#     import sys
#     !{sys.executable} -m pip install onnxruntime

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=USE_ONNX)

(get_speech_timestamps,
save_audio,
read_audio,
VADIterator,
collect_chunks) = utils