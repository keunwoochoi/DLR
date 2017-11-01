from transform import get_DLR_1D, get_DLR_2D, get_tempogram, get_mel_stft
from config import SR, DUR, ASSET_DIR

import matplotlib.pyplot as plt
import librosa

file_path = 'test.mp3'
y, _ = librosa.load(file_path, sr=SR, duration=DUR)
# mel 
mel = get_mel_stft(y)
plt.imshow(mel.transpose([1,0]))
plt.axis('off')
plt.title("mel spectogram")
plt.savefig(ASSET_DIR+'mel.png')
plt.clf()
# tempo
tempo = get_tempogram(y)
plt.imshow(tempo.transpose([1,0]))
plt.axis('off')
plt.title("tempogram")
plt.savefig(ASSET_DIR+'tempo.png')
plt.clf()
# dlr
dlr = get_DLR_1D(y)
plt.imshow(dlr.transpose([1,0]))
plt.axis('off')
plt.title("DLR")
plt.savefig(ASSET_DIR+'DLR.png')
plt.clf()
