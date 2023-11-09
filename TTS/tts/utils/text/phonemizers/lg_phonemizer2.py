from typing import Dict

from TTS.tts.utils.text.luganda.phonemizer import luganda_text_to_phonemes2
from TTS.tts.utils.text.phonemizers.base import BasePhonemizer

_DEF_LG_PUNCS = "' "


class LG_Phonemizer2(BasePhonemizer):
    """🐸TTS Lg phonemizer using functions in `TTS.tts.utils.text.luganda.phonemizer`

    Args:
        punctuations (str):
            Set of characters to be treated as punctuation. Defaults to `_DEF_LG_PUNCS`.

        keep_puncs (bool):
            If True, keep the punctuations after phonemization. Defaults to False.

    Example ::

        "emisingi egyakozesebwa okunnyonnyola" -> 'e|m|i|s|i|ŋ|ŋ|i| e|ɟ|i|a|k|o|z|e|s|e|b|w|a| o|k|u|n|ɲ|o|n|ɲ|o|l|a'

    TODO: someone with Luganda knowledge should check this implementation
    """

    language = "lg"

    def __init__(self, punctuations=_DEF_LG_PUNCS, keep_puncs=False, **kwargs):  # pylint: disable=unused-argument
        super().__init__(self.language, punctuations=punctuations, keep_puncs=keep_puncs)

    @staticmethod
    def name():
        return "lg_phonemizer"

    @staticmethod
    def phonemize_lg(text: str, separator: str = "|") -> str:
        ph = luganda_text_to_phonemes2(text, separator)
        return ph

    def _phonemize(self, text, separator):
        return self.phonemize_lg(text, separator)

    @staticmethod
    def supported_languages() -> Dict:
        return {"lg": "Luganda (Uganda)"}

    def version(self) -> str:
        return "0.0.1"

    def is_available(self) -> bool:
        return True
