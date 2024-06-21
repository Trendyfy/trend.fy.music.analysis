import grpc
from concurrent import futures
import joblib
import pandas as pd
import librosa
import numpy as np
import os

# Importações de music_service_pb2 e music_service_pb2_grpc
from music_service import music_service_pb2 as music__service__pb2
from music_service import music_service_pb2_grpc as music__service__pb2_grpc

def convert_to_list(string):
    try:
        return [float(i) for i in string.strip('[]').split()]
    except ValueError:
        return string

def preprocess_data(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr.T, axis=0)
    
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms.T, axis=0)
    
    features = np.hstack([mfccs_mean, chroma_mean, zcr_mean, rms_mean])
    
    columns = [f'mfcc_{i}' for i in range(13)] + [f'chroma_{i}' for i in range(12)] + ['zcr', 'rms']
    df = pd.DataFrame([features], columns=columns)
    
    return df

# Usar caminhos absolutos
current_dir = os.path.dirname(os.path.abspath(__file__))
model_genre_path = os.path.join(current_dir, '../models/model_genre.joblib')
model_emotion_path = os.path.join(current_dir, '../models/model_emotion.joblib')
model_style_path = os.path.join(current_dir, '../models/model_style.joblib')

model_genre = joblib.load(model_genre_path)
model_emotion = joblib.load(model_emotion_path)
model_style = joblib.load(model_style_path)

class MusicService(music__service__pb2_grpc.MusicServiceServicer):
    def PredictGenre(self, request, context):
        data = preprocess_data(request.audio_path)
        genre_prediction = model_genre.predict(data)[0]
        emotion_prediction = model_emotion.predict(data)[0]
        style_prediction = model_style.predict(data)[0]
        
        return music__service__pb2.MusicResponse(
            genre=genre_prediction,
            emotion=emotion_prediction,
            style=style_prediction
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    music__service__pb2_grpc.add_MusicServiceServicer_to_server(MusicService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
