dependencies = ["torch", "torchaudio"]

URLS = {
    "small_600_24k_4096": "https://github.com/nicolvisser/WavTokenizer/releases/download/v0.1/WavTokenizer_small_600_24k_4096_d44c40fb.ckpt",
    "small_320_24k_4096": "https://github.com/nicolvisser/WavTokenizer/releases/download/v0.1/WavTokenizer_small_320_24k_4096_721a204f.ckpt",
}


import torch

from configs import ARGS_SMALL_320_24K_4096, ARGS_SMALL_600_24K_4096
from decoder.pretrained import WavTokenizer, WavTokenizerArgs


def _load(
    args: WavTokenizerArgs,
    url: str,
    pretrained: bool = True,
    map_location="cpu",
    progress: bool = True,
) -> WavTokenizer:
    """WavTokenizer small. 24kHz, 600x downsample (40 Hz), 4096 codebook entries."""

    model = WavTokenizer(args=args)
    if pretrained:
        state_dict_raw = torch.hub.load_state_dict_from_url(
            url, map_location=map_location, progress=progress, weights_only=True
        )["state_dict"]
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


def small_600_24k_4096(
    pretrained: bool = True,
    map_location="cpu",
    progress: bool = True,
) -> WavTokenizer:
    return _load(
        args=ARGS_SMALL_600_24K_4096,
        url=URLS["small_600_24k_4096"],
        pretrained=pretrained,
        map_location=map_location,
        progress=progress,
    )


def small_320_24k_4096(
    pretrained: bool = True,
    map_location="cpu",
    progress: bool = True,
) -> WavTokenizer:
    """WavTokenizer small. 24kHz, 320x downsample (75 Hz), 4096 codebook entries."""
    return _load(
        args=ARGS_SMALL_320_24K_4096,
        url=URLS["small_320_24k_4096"],
        pretrained=pretrained,
        map_location=map_location,
        progress=progress,
    )


if __name__ == "__main__":
    state_dict = torch.hub.load_state_dict_from_url(
        URLS["small_600_24k_4096"],
        map_location="cpu",
        progress=True,
        weights_only=True,
    )
    print(state_dict.keys())
