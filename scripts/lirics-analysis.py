import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
from textblob import TextBlob
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Baixar recursos necessários do nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Carregar modelo do spaCy
nlp = spacy.load('en_core_web_sm')

# Função para pré-processar o texto usando spaCy
def preprocess_text_spacy(text):
    doc = nlp(text.lower())
    filtered_words = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(filtered_words)

# Função para extrair rimas
def extract_rhymes(lyrics):
    lines = lyrics.split('\n')
    rhymes = [line.split()[-1] for line in lines if line]
    rhyme_pairs = Counter([rhyme[-2:] for rhyme in rhymes if len(rhyme) > 1])
    return len(rhyme_pairs)

# Função para gerar versos aleatórios
def generate_verse():
    verses = [
        "Eu acordei com o som do sertão",
        "Vaqueiro e viola, vida no estradão",
        "O coração bate forte por você",
        "Mesmo distante, não consigo te esquecer",
        "No bar da esquina, lembrei de você",
        "Um brinde à saudade, tentando esquecer",
        "O coração não quer mais sofrer",
        "Mas a lembrança insiste em doer",
        "Caminho sozinho, estrada sem fim",
        "Pensando em você, chorando por mim",
        "Cada música no rádio me faz lembrar",
        "Dos nossos momentos, do nosso lugar"
    ]
    return random.choice(verses)

# Função para gerar refrões aleatórios
def generate_chorus():
    choruses = [
        "Amor, volta pra mim\nSem você, meu mundo é tão ruim\nSaudade machuca, coração vazio\nSem você, minha vida é um desafio",
        "Volta, meu amor, não aguento mais\nEssa distância só me traz solidão\nSeu abraço é o que me satisfaz\nVem curar de vez meu coração",
        "Vem me amar, meu bem\nSem você, eu não sou ninguém\nA saudade aperta e não tem fim\nVolta logo, fica perto de mim",
        "Te quero de volta, meu amor\nNão consigo viver sem teu calor\nSaudade dói, é um grande tormento\nVolta logo, é o meu sentimento"
    ]
    return random.choice(choruses)

# Função para gerar uma letra completa
def generate_lyrics():
    lyrics = f"""
    {generate_verse()}
    {generate_verse()}
    {generate_verse()}
    {generate_verse()}

    Refrão:
    {generate_chorus()}

    {generate_verse()}
    {generate_verse()}
    {generate_verse()}
    {generate_verse()}

    Refrão:
    {generate_chorus()}

    Ponte:
    {generate_verse()}
    {generate_verse()}
    {generate_verse()}

    Refrão Final:
    {generate_chorus()}
    """
    return lyrics.strip()

# Gerar lista de 50 letras
lyrics_list = [generate_lyrics() for _ in range(50)]

# Pré-processar todas as letras
processed_lyrics = [preprocess_text_spacy(lyrics) for lyrics in lyrics_list]

# Função para extrair características das letras
def extract_text_features(lyrics):
    blob = TextBlob(lyrics)
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(lyrics)
    
    word_count = len(blob.words)
    sentence_count = len(blob.sentences)
    avg_word_length = np.mean([len(word) for word in blob.words])
    
    stanzas = lyrics.split('\n\n')
    stanza_count = len(stanzas)
    
    rhyme_count = extract_rhymes(lyrics)
    
    features = {
        'compound': sentiment['compound'],
        'positive': sentiment['pos'],
        'neutral': sentiment['neu'],
        'negative': sentiment['neg'],
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': avg_word_length,
        'stanza_count': stanza_count,
        'rhyme_count': rhyme_count
    }
    return features

# Análise de Tópicos com LDA
stop_words = list(set(stopwords.words('portuguese')).union(ENGLISH_STOP_WORDS))
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=stop_words)
dtm = vectorizer.fit_transform(processed_lyrics)
lda = LatentDirichletAllocation(n_components=2, random_state=0)
lda.fit(dtm)

# Função para extrair tópicos
def extract_topics(lyrics):
    processed_lyrics = preprocess_text_spacy(lyrics)
    dtm = vectorizer.transform([processed_lyrics])
    topic_distribution = lda.transform(dtm)[0]
    topics = {f'topic_{i}': prob for i, prob in enumerate(topic_distribution)}
    return topics

# Extração de características textuais para todas as letras, incluindo análise de tópicos
text_features_with_topics = []
for lyrics in lyrics_list:
    features = extract_text_features(lyrics)
    topics = extract_topics(lyrics)
    features.update(topics)
    text_features_with_topics.append(features)

# Converter para DataFrame
text_features_with_topics_df = pd.DataFrame(text_features_with_topics)
print(text_features_with_topics_df)
