import unittest
import sys

# Adicionar o caminho do diretório scripts ao sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from lyrics_analysis import generate_lyrics, preprocess_text_spacy, extract_text_features, extract_topics

class TestLyricsAnalysis(unittest.TestCase):

    def setUp(self):
        self.lyrics = generate_lyrics()

    def test_generate_lyrics(self):
        lyrics = generate_lyrics()
        self.assertIsInstance(lyrics, str)
        self.assertGreater(len(lyrics), 0)

    def test_preprocess_text_spacy(self):
        processed_text = preprocess_text_spacy(self.lyrics)
        self.assertIsInstance(processed_text, str)
        self.assertGreater(len(processed_text), 0)

    def test_extract_text_features(self):
        features = extract_text_features(self.lyrics)
        self.assertIsInstance(features, dict)
        self.assertIn('compound', features)
        self.assertIn('positive', features)
        self.assertIn('neutral', features)
        self.assertIn('negative', features)
        self.assertIn('word_count', features)
        self.assertIn('sentence_count', features)
        self.assertIn('avg_word_length', features)
        self.assertIn('stanza_count', features)
        self.assertIn('rhyme_count', features)

    def test_extract_topics(self):
        topics = extract_topics(self.lyrics)
        self.assertIsInstance(topics, dict)
        self.assertIn('topic_0', topics)
        self.assertIn('topic_1', topics)

if __name__ == '__main__':
    unittest.main()
