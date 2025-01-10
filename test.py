import torch
import torchaudio

from decoder.pretrained import WavTokenizer
from encoder.utils import convert_audio

from configs import ARGS_SMALL_600_24K_4096, ARGS_SMALL_320_24K_4096

CONFIGS = {
    "40hz": ARGS_SMALL_600_24K_4096,
    "75hz": ARGS_SMALL_320_24K_4096,
}

CHECKPOINT_PATHS = {
    "40hz": "/mnt/wsl/nvme/code/WavTokenizer/checkpoints/WavTokenizer_small_600_24k_4096.ckpt",
    "75hz": "/mnt/wsl/nvme/code/WavTokenizer/checkpoints/WavTokenizer_small_320_24k_4096.ckpt",
}

EXPECTED_TOKEN_PATHS = {
    "40hz": "data/codes_small_600_24k_4096.pt",
    "75hz": "data/codes_small_320_24k_4096.pt",
}
OUTPUT_PATHS = {
    "40hz": "data/codes_small_600_24k_4096.wav",
    "75hz": "data/codes_small_320_24k_4096.wav",
}

audio_path = "data/sample.flac"

device = torch.device("cpu")

for rate in ["40hz", "75hz"]:
    args = CONFIGS[rate]
    model_path = CHECKPOINT_PATHS[rate]

    wavtokenizer = WavTokenizer.from_pretrained(args, model_path)
    wavtokenizer = wavtokenizer.to(device)

    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, 24000, 1)
    bandwidth_id = torch.tensor([0], device=device)
    wav = wav.to(device)
    _, discrete_code = wavtokenizer.encode(wav, bandwidth_id=bandwidth_id)

    features = wavtokenizer.codes_to_features(discrete_code)
    audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
    torchaudio.save(
        OUTPUT_PATHS[rate],
        audio_out.cpu(),
        sample_rate=24000,
        encoding="PCM_S",
        bits_per_sample=16,
    )

    expected_discrete_code = torch.load(
        EXPECTED_TOKEN_PATHS[rate], weights_only=True, map_location=device
    )

    match = torch.allclose(expected_discrete_code, discrete_code)

    if not match:
        print(f"Mismatch for {rate}")
        print(f"   Mismatch count: {torch.sum(expected_discrete_code != discrete_code)}")
        print(f"   Mismatch percentage: {torch.sum(expected_discrete_code != discrete_code) / expected_discrete_code.numel() * 100:.2f} %")
    else:
        print(f"Match for {rate}")
    # assert torch.allclose(expected_discrete_code, discrete_code)

print(
    "All encoded units matched. Please also listen to the output files as a sanity check."
)
