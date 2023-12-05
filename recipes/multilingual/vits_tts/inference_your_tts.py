# YourTTS_luganda_speech_synthesis - Python API Usage

### Imports

import os

import numpy as np
import pandas as pd

import torch

from TTS.tts.utils.synthesis import synthesis
from TTS.utils.audio import AudioProcessor

from TTS.tts.models import setup_model
from TTS.config import load_config
from TTS.tts.models.vits import *

from TTS.tts.utils.speakers import SpeakerManager

### Paths definition

OUT_PATH = '/home/ubuntu/tts/data/wavs_synthesized'

# create output path
os.makedirs(OUT_PATH, exist_ok=True)

# model vars
MODEL_PATH = "/home/ubuntu/tts/data/YourTTS_All_Preprocessed-November-24-2023_05+38AM-0000000/best_model.pth"
CONFIG_PATH = "/home/ubuntu/tts/data/YourTTS_All_Preprocessed-November-24-2023_05+38AM-0000000/config.json"
TTS_LANGUAGES = "/home/ubuntu/tts/data/YourTTS_All_Preprocessed-November-24-2023_05+38AM-0000000/language_ids.json"
TTS_SPEAKERS = "/home/ubuntu/tts/speakers.json"
USE_CUDA = torch.cuda.is_available()

# encoder vars
CONFIG_SE_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json"
CHECKPOINT_SE_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar"

### Restore model

# load the config
C = load_config(CONFIG_PATH)


# load the audio processor
ap = AudioProcessor(**C.audio)

speaker_embedding = None

C.model_args['d_vector_file'] = TTS_SPEAKERS
C.model_args['use_speaker_encoder_as_loss'] = False

model = setup_model(C)
model.language_manager.load_ids_from_file(TTS_LANGUAGES)

cp = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

# remove speaker encoder
model_weights = cp['model'].copy()
for key in list(model_weights.keys()):
  if "speaker_encoder" in key:
    del model_weights[key]

model.load_state_dict(model_weights)

model.eval()

if USE_CUDA:
    model = model.cuda()

# synthesize voice
use_griffin_lim = False

# load speaker manager
SE_speaker_manager = SpeakerManager.init_from_config(C)

SE_speaker_manager.init_encoder(
    model_path=CHECKPOINT_SE_PATH,
    config_path=CONFIG_SE_PATH,
    use_cuda=USE_CUDA
)

### Define inference variables

# Get embeddings
embeddings = [embedding['embedding'] for embedding in SE_speaker_manager.embeddings.values()]

# Define inference variables
model.length_scale = 1  # scaler for the duration predictor. The larger it is, the slower the speech.
model.inference_noise_scale = 0.3 # defines the noise variance applied to the random z vector at inference.
model.inference_noise_scale_dp = 0.3 # defines the noise variance applied to the duration predictor z vector at inference.

# Choose language id
language_id = 0

with open('/home/ubuntu/tts/data/scorer-cv.txt', 'r') as inference_texts_file:
  inference_texts = inference_texts_file.read().splitlines()

min_text_length = 33
max_text_length = 83

for text in inference_texts:
  text_length = len(text.replace(' ', ''))

  if text_length<min_text_length or text_length>max_text_length:
    inference_texts.remove(text)

### Sythesis

file_name = "your-tts-inference"
datalist = []

for idx, text in enumerate(inference_texts):
    reference_emb = embeddings[np.random.randint(0, len(embeddings))]
    print(" > text: {}".format(text))

    wav, alignment, _, _ = synthesis(
                        model,
                        text,
                        C,
                        "cuda" in str(next(model.parameters()).device),
                        speaker_id=None,
                        d_vector=reference_emb,
                        style_wav=None,
                        language_id=language_id,
                        use_griffin_lim=True,
                        do_trim_silence=False,
                    ).values()

    out_path = os.path.join(OUT_PATH, f"{file_name}-{str(idx)}.wav")
    print(" > Saving output to {}".format(out_path))
    ap.save_wav(wav, out_path)

    datalist.append([f"{file_name}-{str(idx)}.wav", text])

dataframe = pd.DataFrame(data=datalist, columns=['filename', 'transcript'])
dataframe.to_csv('/home/ubuntu/tts/data/synthesized.csv', sep='|', index=False)
