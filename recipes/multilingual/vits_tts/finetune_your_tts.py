import os
from glob import glob

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

output_path = "/home/ubuntu/tts/data"

dataset_config = [
    BaseDatasetConfig(
        formatter="custom_formatter", 
        meta_file_train="top_cv.csv", 
        path=output_path, 
        language="lg", 
    ),
    BaseDatasetConfig(
        formatter="custom_formatter", 
        meta_file_train="top_yogera.csv", 
        path=output_path, 
        language="lg", 
    )
]

audio_config = VitsAudioConfig(
    sample_rate=16000,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None,
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
    num_chars=29,
    use_d_vector_file=True,
    d_vector_file=["/home/ubuntu/tts/speakers-cv.json", "/home/ubuntu/tts/speakers-yogera.json"],
    use_sdp=True,
    spec_segment_size=32,
    num_layers_text_encoder=6,
    resblock_type_decoder="2",
    noise_scale=1.0,
    inference_noise_scale=0.3,
    length_scale=1.5,
    noise_scale_dp=0.6,
    inference_noise_scale_dp=0.3,
    d_vector_dim=512,
    use_speaker_encoder_as_loss=True,
    speaker_encoder_config_path="https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json",
    speaker_encoder_model_path="https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar",
)

config = VitsConfig(
    model_args=vitsArgs,
    audio=audio_config,
    batch_size=32,
    eval_batch_size=16,
    batch_group_size=8,
    eval_split_size=0.1,
    precompute_num_workers=os.cpu_count()-2,
    num_loader_workers=os.cpu_count()-2,
    num_eval_loader_workers=os.cpu_count()-2,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=250,
    text_cleaner="english_cleaners",
    use_phonemes=True,
    phoneme_language='lg',
    phonemizer="lg_phonemizer",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=False,
    print_step=250,
    print_eval=False,
    mixed_precision=False,
    output_path=output_path,
    datasets=dataset_config,
    characters=characters_config,
    use_speaker_weighted_sampler=True,
    speaker_weighted_sampler_alpha=3.0,
    use_length_weighted_sampler=True,
    length_weighted_sampler_alpha=1.5,
    speaker_encoder_loss_alpha=3.0,
    lr_gen=2e-5,
    lr_disc=2e-5,
    cudnn_enable=True,
    use_noise_augment=False,
    test_sentences=[
        [
            "abantu bajja kufuna emirimu olwa zi pulojekiti ennyingi eziri mu kitundu",
            "9e4ae0f82e3d4d2747be843f029ba9e948a992a7035dcd342169d06059541ce4dc0f1bd15e371b08a06add7300d8ab93749b0ede144ec8b7c7273f961996940d\n",
            None,
            "lg",
        ],
        [
            "ngenda kukuleetera ekikopo ekirala",
            "f37d5566649a7c20152daf1a4a859316027bb4b2bb46e51c6a98de274d7c1172e74b9d23bf44570d2e902c4532a38594ab7bd9d85c4aa960ae2aa676d52748b7\n",
            None,
            "lg",
        ],
        [
            "abasomesa baasaba abazadde ssente z'okusomesa abayizi mu luwummula",
            "b0d13e8d3c164921430520a2db0ead9f2b077e673c47f247327bc233e56ece66030be2f19c84baee52ef28af1ae3fe1fdd62317d3de360e86c24f063ba92a598\n",
            None,
            "lg",
        ],
        [
            "kubanga mulyogerera mu bbanga",
            "7b01ca05762b1c62944a249461f1dc5ade5bc855aebd8e6439ac4ddac52428089907a6f7e85fbbb21c4251f444428092e19a5b617d917d0f6dcd6a21a83e3653\n",
            None,
            "lg",
        ],
        [
            "bwe muba mulima mukozesa nkumbi ki",
            "325\n",
            None,
            "lg",
        ],
        [
            "nasaba dda kitange ampe we nnimira",
            "90\n",
            None,
            "lg",
        ],
        [
            "nnasooka kunoonya ddagala kufuuyira nnyaanya nga ssinnajja",
            "203\n",
            None,
            "lg",
        ],
        [
            "kasujja emmwanyi ze zonna yazitunze mbisi ng'anoonya ffiizi",
            "202\n",
            None,
            "lg",
        ],
        [
            "amagye gasiima omulimu gwa bannayuganda eri eggwanga lyabwe",
            "8b0c90712c9c8ba1310f4449bfd477092ef3b6e6301bbda9936bd9102dbc0d7f627ccdbe114973e84f5cc2777b6963da52ce84c081071560ed6d828e57517e17\n",
            None,
            "lg",
        ],
        [
            "ebirango by'okuziika ennaku zino tebikyasomebwa nnyo",
            "d4aaacd3b77be95d0aa8210da1189a910f8ef5258907619ac7258d5618e1a651c3ee4ca1859715aafff4f12ebeef08fe3522ef4521e1c7fa2b2074c4ea5d141f\n",
            None,
            "lg",
        ],
        [
            "minisita w'ebyenjigiriza janet museveni annyonnyodde ku bino",
            "d3e3c9accb377bcba8df06efed58758932d8e64d7d7b2a9c29568009166a79273abecd6fd12389f5110c50e918c595d00171a7a1695d5f410e9d1d58b097dcf1\n",
            None,
            "lg",
        ],
        [
            "biringanya tezisobola kukula bulungi nga omuddo mungi",
            "121\n",
            None,
            "lg",
        ],
        [
            "ekiyumba ky'enkoko kibaamu kalimbwe mungi n'obukuta bw'emmwanyi",
            "192\n",
            None,
            "lg",
        ],
        [
            "buuza abalimisa okupima asidi w'ettaka",
            "199\n",
            None,
            "lg",
        ],
    ],
)

# force the convertion of the custom characters to a config attribute
config.from_dict(config.to_dict())

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager.init_from_config(config)
speaker_manager.init_encoder(
    model_path=config.model_args.speaker_encoder_model_path, 
    config_path=config.model_args.speaker_encoder_config_path, 
    use_cuda=True
)
config.model_args.num_speakers = speaker_manager.num_speakers

language_manager = LanguageManager(config=config)
config.model_args.num_languages = language_manager.num_languages

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# init model
model = Vits(config, ap, tokenizer, speaker_manager, language_manager)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()
