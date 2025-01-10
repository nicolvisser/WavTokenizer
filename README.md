# WavTokenizer for inference

Forked and stripped-down version of [WavTokenizer](https://github.com/jishengpeng/WavTokenizer).

Used for **inference only** within workspaces with more recent python (3.12) and torch (2.5.1) versions. By stripping unused code and dependencies, this repo avoids [dependency hell](https://en.wikipedia.org/wiki/Dependency_hell). This repo depends only on `torch` and `torchaudio`.

Refer to the [original repo](https://github.com/jishengpeng/WavTokenizer) if you need to modify, train or **cite** the model.

## Usage

```python
import torch
import torchaudio

device = "cuda" if torch.cuda.is_available() else "cpu"

# load model
model = torch.hub.load("nicolvisser/WavTokenizer", "small_600_24k_4096")  # 40 Hz
# or
# model = torch.hub.load("nicolvisser/WavTokenizer", "small_320_24k_4096")  # 75 Hz
model.to(device)
model.eval()

# load audio
wav, sr = torchaudio.load("data/sample.flac")
wav = torchaudio.functional.resample(wav, sr, 24000)
wav = wav.to(device)
print(wav.shape)

# encoding
bandwidth_id = torch.tensor([0], device=device)
_, codes = model.encode(wav, bandwidth_id=bandwidth_id)
print(codes.shape)

# # decoding
features = model.codes_to_features(codes)
bandwidth_id = torch.tensor([0], device=device)
audio_out = model.decode(features, bandwidth_id=bandwidth_id)
print(audio_out.shape)

```
