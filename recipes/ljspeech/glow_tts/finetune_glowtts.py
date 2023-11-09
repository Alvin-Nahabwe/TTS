import os

# Trainer: Where the ✨️ happens.
# TrainingArgs: Defines the set of arguments of the Trainer.
from trainer import Trainer, TrainerArgs

# GlowTTSConfig: all model related values for training, validating and testing.
from TTS.tts.configs.glow_tts_config import GlowTTSConfig

# BaseDatasetConfig: defines name, formatter and path of the dataset.
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# we use the same path as this script as our training folder.
output_path = "/home/alvin_n256/tts/data/"

# DEFINE DATASET CONFIG
# Set LJSpeech as our target dataset and define its path.
# You can also use a simple Dict to define the dataset and pass it to your custom formatter.
dataset_config = BaseDatasetConfig(
    formatter="custom_formatter", meta_file_train="top_female_yogera_metadata.csv", path=output_path
)

characters_config = CharactersConfig(
    pad="[PAD]", eos="[EOS]", bos="[BOS]", characters="abcdefghijklmnoprstuvwxyz\u0272\u025f\u014b", punctuations="' "
)

audio_config = BaseAudioConfig()

tune_audio_params={'sample_rate': 16000, 'preemphasis': 0.97, 'ref_level_db': 35, 'power': 1.7, 'mel_fmin': 95.0, 'mel_fmax': 6000.0, 'do_trim_silence': False}
reset_audio_params={'signal_norm': True, 'stats_path': None, 'symmetric_norm': False, 'max_norm': 1, 'clip_norm': True}

audio_config.update(reset_audio_params)
audio_config.update(tune_audio_params)

# INITIALIZE THE TRAINING CONFIGURATION
# Configure the model. Every config class inherits the BaseTTSConfig.
config = GlowTTSConfig(
    audio=audio_config,
    batch_size=64,
    eval_batch_size=16,
    num_loader_workers=8,
    num_eval_loader_workers=8,
    run_eval=True,
    eval_split_size=0.1,
    test_delay_epochs=10,
    epochs=1000,
    lr=0.0001,
    text_cleaner="english_cleaners",
    use_phonemes=True,
    phoneme_language="lg",
    phonemizer="lg_phonemizer",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    min_text_len=3,
    max_text_len=160,
    print_step=25,
    print_eval=False,
    mixed_precision=False,
    output_path=output_path,
    datasets=[dataset_config],
    characters=characters_config,
)

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
model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

# INITIALIZE THE TRAINER
# Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
# distributed training, etc.
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

# AND... 3,2,1... 🚀
trainer.fit()
