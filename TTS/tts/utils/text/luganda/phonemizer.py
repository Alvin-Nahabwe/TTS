from typing import List
from cvutils import Phonemiser

p = Phonemiser('lg')
p.lkp.update({'ng': ['ŋŋ']})

_CONVRULES = ['/ '.join(item) for item in {key:''.join(value) for key, value in p.lkp.items()}.items()]


def _makerulemap():
    l = [tuple(x.split("/")) for x in _CONVRULES]
    return tuple({k: v for k, v in l if len(k) == i} for i in (1, 2))


_RULEMAP1, _RULEMAP2 = _makerulemap()


def lg2phoneme(text: str) -> str:
    """Convert luganda text to phonemes."""
    text = text.strip()
    res = ""
    while text:
        if len(text) >= 2:
            x = _RULEMAP2.get(text[:2])
            if x is not None:
                text = text[2:]
                res += x
                continue
        x = _RULEMAP1.get(text[0])
        if x is not None:
            text = text[1:]
            res += x
            continue
        res += " " + text[0]
        text = text[1:]
    return res[1:]


def luganda_text_to_phonemes2(text: str, seperator: str = "|") -> str:
    """Convert Luganda text to phonemes."""
    res = lg2phoneme(res)
    res = res.replace("   ", "-")
    return res.replace(" ", seperator).replace("-", seperator+" ").replace(f"{seperator}'{seperator}", "'")


def luganda_text_to_phonemes(text: str, seperator: str = "|") -> str:
    return seperator+" ".join(map(lambda txt: seperator.join(list(p.phonemise(txt))), text.split()))
