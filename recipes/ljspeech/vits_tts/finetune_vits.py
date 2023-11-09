import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig, CharactersConfig, VitsArgs
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

output_path = "/home/alvin_n256/tts/data/"

dataset_config = [
    BaseDatasetConfig(formatter="custom_formatter", meta_file_train="top_spk_filtered_cv.csv", path=output_path, language="lg", phonemizer="lg_phonemizer")
]

audio_config = VitsAudioConfig(
    sample_rate=16000, 
    win_length=1024, 
    hop_length=256, 
    num_mels=80, 
    mel_fmin=95, 
    mel_fmax=6000,
)

characters_config = CharactersConfig(
    pad="<PAD>",
    eos="<EOS>",
    bos="<BOS>",
    blank="<BLNK>",
    characters_class="TTS.tts.models.vits.VitsCharacters",
    characters="abcdefghijklmnoprstuvwxyz\u0272\u025f\u014b",
    punctuations="' ",
)

vitsArgs = VitsArgs(
    spec_segment_size=32,
    num_layers_text_encoder=6,
    resblock_type_decoder="1",
    use_sdp=True,
    noise_scale_dp=0.6,
    inference_noise_scale_dp=0.3,
)

config = VitsConfig(
    audio=audio_config,
    batch_size=16,
    eval_batch_size=8,
    batch_group_size=2,
    num_loader_workers=os.cpu_count(),
    num_eval_loader_workers=os.cpu_count(),
    run_eval=True,
    eval_split_size=0.1,
    test_delay_epochs=-1,
    epochs=10000,
    text_cleaner="english_cleaners",
    use_phonemes=True,
    phoneme_language="lg",
    phonemizer="lg_phonemizer",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=False,
    print_step=25,
    print_eval=False,
    mixed_precision=False,
    output_path=output_path,
    datasets=dataset_config,
    characters=characters_config,
    cudnn_enable=True,
    cudnn_benchmark=False,
    lr_gen=0.000002,
    lr_disc=0.000002,
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init model
model = Vits(config, ap, tokenizer, speaker_manager=None)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
