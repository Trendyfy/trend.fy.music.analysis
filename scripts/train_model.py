import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import ast
import os

# Função para converter strings de listas em listas reais
def convert_to_list(string):
    try:
        # Remove colchetes e converte para lista de floats
        return [float(i) for i in string.strip('[]').split()]
    except ValueError:
        return string

# Carregar os dados
data = pd.read_csv('data/processed/success_songs_features.csv')

# Pré-processamento
# Converter colunas que são listas de strings para listas reais
list_columns = ['chroma_stft', 'mfcc']

for col in data.columns:
    if col not in ['genre', 'emotion', 'style']:
        data[col] = data[col].apply(lambda x: convert_to_list(x) if isinstance(x, str) else x)

# Flatten colunas que são listas de floats
flatten_columns = {}

for col in data.columns:
    if isinstance(data[col].iloc[0], list):
        flatten_columns[col] = pd.DataFrame(data[col].tolist(), index=data.index).add_prefix(f'{col}_')

# Concatenar as novas colunas no dataframe original e remover as antigas
for col, new_df in flatten_columns.items():
    data = pd.concat([data, new_df], axis=1)
    data.drop(columns=[col], inplace=True)

# Separar as features e os labels
X = data.drop(columns=['genre', 'emotion', 'style'])
y_genre = data['genre']
y_emotion = data['emotion']
y_style = data['style']

# Dividir os dados em treino e teste
X_train, X_test, y_genre_train, y_genre_test = train_test_split(X, y_genre, test_size=0.2, random_state=42)
_, _, y_emotion_train, y_emotion_test = train_test_split(X, y_emotion, test_size=0.2, random_state=42)
_, _, y_style_train, y_style_test = train_test_split(X, y_style, test_size=0.2, random_state=42)

# Treinar um modelo simples
model_genre = RandomForestClassifier()
model_genre.fit(X_train, y_genre_train)

# Criar pasta 'models' se não existir
if not os.path.exists('models'):
    os.makedirs('models')

# Salvar o modelo
joblib.dump(model_genre, 'models/model_genre.joblib')

# Repita o processo para emoção e estilo
model_emotion = RandomForestClassifier()
model_emotion.fit(X_train, y_emotion_train)
joblib.dump(model_emotion, 'models/model_emotion.joblib')

model_style = RandomForestClassifier()
model_style.fit(X_train, y_style_train)
joblib.dump(model_style, 'models/model_style.joblib')

print('Modelos treinados e salvos com sucesso.')
