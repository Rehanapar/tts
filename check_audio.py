import json
import os
import soundfile as sf
import numpy as np
from scipy.signal import stft

with open("config.json", "r") as f:
    hps = json.load(f)["data"]

root = "dataset/audio"

bad_files = []

for fname in os.listdir(root):
    if fname.endswith(".wav"):
        path = os.path.join(root, fname)

        try:
            audio, sr = sf.read(path)
            if len(audio) == 0:
                bad_files.append(path)
                continue

            f, t, Zxx = stft(audio, sr, nperseg=1024)
            if Zxx.shape[1] == 0:  
                bad_files.append(path)

        except Exception as e:
            bad_files.append(path)

print("\nFILES WITH EMPTY MEL:", len(bad_files))
for f in bad_files:
    print(f)



