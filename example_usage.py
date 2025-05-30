import torch
import torchaudio

device = "cuda" if torch.cuda.is_available() else "cpu"

# load model
# 40 Hz:
model, encode, decode = torch.hub.load(
    "nicolvisser/WavTokenizer", "small_600_24k_4096", trust_repo=True
)
# or 75 Hz:
# model, encode, decode = torch.hub.load("nicolvisser/WavTokenizer", "small_320_24k_4096", trust_repo=True)
model.to(device)
model.eval()

# load audio
wav, sr = torchaudio.load("data/sample.flac")

# encode and decode with provided functions
codes = encode(model, wav, sr)
wav_, sr_ = decode(model, codes)

# save audio
torchaudio.save("data/sample_out.wav", wav_.cpu(), sr_)
