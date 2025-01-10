import torch
import torchaudio

from decoder.pretrained import WavTokenizer
from encoder.utils import convert_audio

device = torch.device("cpu")

config_path = "/mnt/wsl/nvme/code/WavTokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
model_path = (
    "/mnt/wsl/nvme/code/WavTokenizer/checkpoints/WavTokenizer_small_320_24k_4096.ckpt"
)

audio_path = "data/sample.flac"

wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device)

wav, sr = torchaudio.load(audio_path)
wav = convert_audio(wav, sr, 24000, 1)
bandwidth_id = torch.tensor([0])
wav = wav.to(device)
_, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)

features = wavtokenizer.codes_to_features(discrete_code)
audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
torchaudio.save(
    "data/codes_small_320_24k_4096.wav",
    audio_out,
    sample_rate=24000,
    encoding="PCM_S",
    bits_per_sample=16,
)

expected_discrete_code = torch.load(
    "data/codes_small_320_24k_4096.pt", weights_only=True
)

assert torch.allclose(expected_discrete_code, discrete_code)

print(
    "All encoded units matched. Please also listen to the output file as a sanity check."
)
