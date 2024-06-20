import audioread
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

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

# Função para aplicar um filtro passa-alta
def high_pass_filter(y, sr, cutoff=100):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, y)
    return y

# Função para normalização
def normalize_audio(y):
    y = librosa.util.normalize(y)
    return y

# Função para redução de ruído
def reduce_noise(y, sr):
    y_denoised = librosa.effects.remix(y, intervals=librosa.effects.split(y, top_db=20))
    return y_denoised

# Função para melhorar o áudio
def improve_audio(y, sr):
    y = high_pass_filter(y, sr)    
    y = reduce_noise(y, sr)
    y = normalize_audio(y)
    return y

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

# Função para separação harmônica e percussiva
def harmonic_percussive_separation(audio):
    harmonic, percussive = librosa.effects.hpss(audio)
    return harmonic, percussive

# Função para extrair a melodia
def extract_melody(audio, sr):
    pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sr)
    melody = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            melody.append(pitch)
        else:
            melody.append(0)
    return np.array(melody)

# Função para salvar o áudio
def save_audio(y, sr, output_path):
    try:
        sf.write(output_path, y, sr)
        print(f"Áudio salvo com sucesso em: {output_path}")
    except Exception as e:
        print(f"Erro ao salvar o arquivo de áudio: {e}")

# Função para extrair características e salvar em CSV
def extract_features_and_save(song_paths, label, output_csv):
    all_features = []
    for song in song_paths:
        audio, sr = load_audio(song['path'])
        improved_audio = improve_audio(audio, sr)
        features = extract_audio_features(improved_audio, sr)
        if features:
            features.update({
                'genre': song['genre'],
                'emotion': song['emotion'],
                'style': song['style']
            })
            all_features.append(features)
    df = pd.DataFrame(all_features)
    df.to_csv(output_csv, index=False)

# Lista de caminhos para arquivos de músicas de sucesso
success_songs = [
    {'path': 'data/raw/briga.wav', 'genre': 'sertanejo', 'emotion': 'animada', 'style': 'modão'},
    {'path': 'data/raw/evidencias.wav', 'genre': 'sertanejo', 'emotion': 'triste', 'style': 'caipira'},
    # Adicione mais músicas e seus atributos
]

# Extrair características e salvar em CSV
extract_features_and_save(success_songs, 'success', 'data/processed/success_songs_features.csv')

# Carregar os dados
df = pd.read_csv('data/processed/success_songs_features.csv')
X = df.drop(columns=['genre', 'emotion', 'style'])
y = df[['genre', 'emotion', 'style']]

# Caminho do arquivo de áudio
audio_path = 'data/raw/briga.wav'
output_path = 'data/processed/briga_melhorada.wav'

# Carregar o arquivo de áudio
audio, sr = load_audio(audio_path)

# Melhorar o áudio
audio = improve_audio(audio, sr)

# Salvar o áudio melhorado
save_audio(audio, sr, output_path)

# Extrair características do áudio
audio_features = extract_audio_features(audio, sr)

# Exibir as características extraídas
if audio_features:
    print(audio_features)
else:
    print("Não foram extraídas características do áudio.")

# Separar componentes harmônicos e percussivos
harmonic, percussive = harmonic_percussive_separation(audio)

# Extrair a melodia
melody = extract_melody(harmonic, sr)
print("Melody:", melody)
