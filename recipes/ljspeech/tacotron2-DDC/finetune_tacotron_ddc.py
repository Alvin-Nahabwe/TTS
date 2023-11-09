import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# from TTS.tts.datasets.tokenizer import Tokenizer

output_path = "/home/alvin_n256/tts/data/"

# init configs
dataset_config = BaseDatasetConfig(
    formatter="custom_formatter", meta_file_train="top_female_cv_metadata.csv", path=output_path
)

characters_config = CharactersConfig(
    pad="[PAD]", eos="[EOS]", bos="[BOS]", characters="abcdefghijklmnoprstuvwxyz\u0272\u025f\u014b", punctuations="' "
)

audio_config = BaseAudioConfig()

tune_audio_params={'sample_rate': 16000, 'preemphasis': 0.97, 'ref_level_db': 35, 'power': 1.7, 'mel_fmin': 95.0, 'mel_fmax': 6000.0, 'do_trim_silence': False}
reset_audio_params={'signal_norm': True, 'stats_path': None, 'symmetric_norm': False, 'max_norm': 1, 'clip_norm': True}

audio_config.update(reset_audio_params)
audio_config.update(tune_audio_params)

config = Tacotron2Config(  # This is the config that is saved for the future use
    audio=audio_config,
    batch_size=64,
    eval_batch_size=16,
    num_loader_workers=8,
    num_eval_loader_workers=8,
    run_eval=True,
    eval_split_size=0.1,
    test_delay_epochs=10,
    r=3,
    gradual_training=[[278000, 3, 64], [290000, 2, 48], [302000, 1, 32]],
    double_decoder_consistency=True,
    epochs=1000,
    lr=0.000001,
    text_cleaner="english_cleaners",
    use_phonemes=True,
    phoneme_language="lg",
    phonemizer="lg_phonemizer",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    min_text_len=3,
    max_text_len=160,
    precompute_num_workers=8,
    print_step=25,
    print_eval=False,
    mixed_precision=False,
    output_path=output_path,
    datasets=[dataset_config],
    characters=characters_config,
)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# If characters are not defined in the config, default characters are passed to the config
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

# INITIALIZE THE MODEL
# Models take a config object and a speaker manager as input
# Config defines the details of the model like the number of layers, the size of the embedding, etc.
# Speaker manager is used by multi-speaker models.
model = Tacotron2(config, ap, tokenizer, speaker_manager=None)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()
