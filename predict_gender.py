import librosa
import os
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, help='Input filename in dir wav_data')
args = parser.parse_args()


filename = os.path.dirname(__file__)+'\\wav_data\\'+args.filename

x = np.zeros(22)
audio, sr = librosa.load(filename, mono=True) #загрузка файла
        
audio, _ = librosa.effects.trim(audio, top_db=20, frame_length=2048, hop_length=64) #триммирование тишины
spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)  #спектральный центр
spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)  #ширина полосы
mfcc = librosa.feature.mfcc(y=audio, sr=sr)  #Мел-частотные кепстральные коэффициенты 
x[0] = np.mean(spec_cent)
x[1] = np.mean(spec_bw)
for i in range(20):
     x[i+2] = np.mean(mfcc[i])
        
        
with open(os.path.dirname(__file__)+'\\model.pkl', 'rb') as f:
    model = pickle.load(f)

print('female') if model.predict(x.reshape(1,-1))[0] else print('male')