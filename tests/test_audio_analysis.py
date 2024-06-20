import unittest
import os
import numpy as np
import soundfile as sf
import sys

# Adicionar o caminho do diretório scripts ao sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from audio_analysis import load_audio, extract_audio_features

class TestAudioAnalysis(unittest.TestCase):

    def setUp(self):
        self.test_audio_path = 'data/raw/briga.wav'
        
        # Criar um arquivo de áudio de teste simples
        sr = 22050
        t = 5.0  # 5 segundos
        freq = 440.0  # A4
        x = (np.sin(2 * np.pi * np.arange(sr * t) * freq / sr)).astype(np.float32)
        sf.write(self.test_audio_path, x, sr)

    def tearDown(self):
        # Remover arquivos de teste criados
        if os.path.exists(self.test_audio_path):
            os.remove(self.test_audio_path)

    def test_load_audio(self):
        audio, sr = load_audio(self.test_audio_path)
        self.assertIsNotNone(audio)
        self.assertIsNotNone(sr)
        self.assertEqual(sr, 22050)

    def test_extract_audio_features(self):
        audio, sr = load_audio(self.test_audio_path)
        features = extract_audio_features(audio, sr)
        self.assertIsNotNone(features)
        self.assertIn('tempo', features)
        self.assertIn('chroma_stft', features)
        self.assertIn('rmse', features)
        self.assertIn('spectral_centroid', features)
        self.assertIn('spectral_bandwidth', features)
        self.assertIn('rolloff', features)
        self.assertIn('zero_crossing_rate', features)
        self.assertIn('mfcc', features)

if __name__ == '__main__':
    unittest.main()
