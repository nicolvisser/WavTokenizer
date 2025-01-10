dependencies = ["torch", "torchaudio"]

URLS = {
    "small_600_24k_4096": "https://github.com/nicolvisser/WavTokenizer/releases/download/v0.1/WavTokenizer_small_600_24k_4096_d44c40fb.ckpt",
    "small_320_24k_4096": "https://github.com/nicolvisser/WavTokenizer/releases/download/v0.1/WavTokenizer_small_320_24k_4096_721a204f.ckpt",
}


import torch

from configs import ARGS_SMALL_320_24K_4096, ARGS_SMALL_600_24K_4096
from decoder.pretrained import WavTokenizer


def small_600_24k_4096(
    pretrained: bool = True,
    progress: bool = True,
) -> WavTokenizer:
    """WavTokenizer small. 24kHz, 600x downsample (40 Hz), 4096 codebook entries."""

    model = WavTokenizer(args=ARGS_SMALL_600_24K_4096)
    if pretrained:
        state_dict_raw = torch.hub.load_state_dict_from_url(
            URLS["small_600_24k_4096"], progress=progress
        )
        state_dict = dict()
        for k, v in state_dict_raw.items():
            if (
                k.startswith("backbone.")
                or k.startswith("head.")
                or k.startswith("feature_extractor.")
            ):
                state_dict[k] = v
        model.load_state_dict(state_dict)
        model.eval()
    return model


def small_320_24k_4096(
    pretrained: bool = True,
    progress: bool = True,
) -> WavTokenizer:
    """WavTokenizer small. 24kHz, 320x downsample (75 Hz), 4096 codebook entries."""

    model = WavTokenizer(args=ARGS_SMALL_320_24K_4096)
    if pretrained:
        state_dict_raw = torch.hub.load_state_dict_from_url(
            URLS["small_320_24k_4096"], progress=progress
        )
        state_dict = dict()
        for k, v in state_dict_raw.items():
            if (
                k.startswith("backbone.")
                or k.startswith("head.")
                or k.startswith("feature_extractor.")
            ):
                state_dict[k] = v
        model.load_state_dict(state_dict)
        model.eval()
    return model


if __name__ == "__main__":
    model = small_600_24k_4096(pretrained=True, progress=True)
    print(model)
