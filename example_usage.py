import torch
import torchaudio

device = "cuda" if torch.cuda.is_available() else "cpu"

# load model
model = torch.hub.load("nicolvisser/WavTokenizer", "small_600_24k_4096")  # 40 Hz
# or
# model = torch.hub.load("nicolvisser/WavTokenizer", "small_320_24k_4096")  # 75 Hz
model.to(device)
model.eval()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

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

# save audio
torchaudio.save("data/sample_out.wav", audio_out.cpu(), 24000)
