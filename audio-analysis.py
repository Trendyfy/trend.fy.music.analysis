import audioread
import librosa
import numpy as np

# Função para carregar um arquivo de áudio
def load_audio(file_path):
    try:
        with audioread.audio_open(file_path) as f:
            print(f"Tentando carregar o arquivo: {file_path}")
            y, sr = librosa.load(file_path, sr=None)
            print(f"Arquivo carregado com sucesso: {file_path}")
            return y, sr
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return None, None

# Função para extrair características do áudio
def extract_audio_features(audio, sr):
    if audio is None or sr is None:
        print("Áudio não carregado corretamente.")
        return None
    
    features = {}
    
    try:
        features['tempo'], _ = librosa.beat.beat_track(y=audio, sr=sr)
        features['chroma_stft'] = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
        features['rmse'] = np.mean(librosa.feature.rms(y=audio).T, axis=0)
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr).T, axis=0)
        features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr).T, axis=0)
        features['rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr).T, axis=0)
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
        features['mfcc'] = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)
    except Exception as e:
        print(f"Erro ao extrair características: {e}")
        return None
    
    return features

# Caminho do arquivo de áudio
audio_path = r'C:\music\evidencias.wav'

# Carregar o arquivo de áudio
audio, sr = load_audio(audio_path)

# Extrair características do áudio
audio_features = extract_audio_features(audio, sr)

# Exibir as características extraídas
if audio_features:
    print(audio_features)
else:
    print("Não foram extraídas características do áudio.")
