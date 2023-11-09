import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs import MultibandMelganConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN

output_path = "/home/alvin_n256/tts/data/"
data_path = os.path.join(output_path, "wavs/")
eval_split_size = int(len(os.listdir(data_path)) * 0.1)

audio_config = BaseAudioConfig()

tune_audio_params={'sample_rate': 16000, 'preemphasis': 0.97, 'ref_level_db': 35, 'power': 1.7, 'mel_fmin': 95.0, 'mel_fmax': 6000.0, 'do_trim_silence': False}
reset_audio_params={'signal_norm': True, 'stats_path': None, 'symmetric_norm': False, 'max_norm': 1, 'clip_norm': True}

audio_config.update(reset_audio_params)
audio_config.update(tune_audio_params)

config = MultibandMelganConfig(
    audio=audio_config,
    batch_size=64,
    eval_batch_size=16,
    num_loader_workers=8,
    num_eval_loader_workers=8,
    run_eval=True,
    test_delay_epochs=10,
    epochs=10000,
    seq_len=16384,
    pad_short=2000,
    use_noise_augment=True,
    eval_split_size=eval_split_size,
    print_step=250,
    print_eval=False,
    mixed_precision=False,
    lr_gen=1e-5,
    lr_disc=1e-5,
    data_path=data_path,
    output_path=output_path,
)

l1_spec_loss_params={'sample_rate': 16000, 'mel_fmin': 95.0, 'mel_fmax': 6000.0}
config.l1_spec_loss_params.update(l1_spec_loss_params)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)

# init model
model = GAN(config, ap)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()
