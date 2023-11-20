import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Add here the xtts_config path
CONFIG_PATH = "/home/ubuntu/tts/data/GPT_XTTS_v2.0_Common_Voice-November-18-2023_03+05PM-0000000/config.json"
# Add here the vocab file that you have used to train the model
TOKENIZER_PATH = "/home/ubuntu/tts/tokenizer.json"
# Add here the checkpoint that you want to do inference with
XTTS_CHECKPOINT = "/home/ubuntu/tts/data/GPT_XTTS_v2.0_Common_Voice-November-18-2023_03+05PM-0000000/best_model.pth"
# Add here the speaker reference
SPEAKER_REFERENCE = [
    "/home/ubuntu/tts/data/wavs/common_voice_lg_24013035.wav",
]
LANGUAGE = "lg"
INFERENCE_TEXT = "abantu bajja kufuna emirimu olwa zi pulojekiti ennyingi eziri mu kitundu"

# output wav path
OUTPUT_WAV_PATH = "xtts-inference.wav"

print("Loading model...")
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=XTTS_CHECKPOINT, vocab_path=TOKENIZER_PATH, use_deepspeed=False)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=SPEAKER_REFERENCE)

print("Inference...")
out = model.inference(
    INFERENCE_TEXT,
    LANGUAGE,
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.7, # Add custom parameters here
)
torchaudio.save(OUTPUT_WAV_PATH, torch.tensor(out["wav"]).unsqueeze(0), 24000)