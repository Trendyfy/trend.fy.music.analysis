import sys
import pandas as pd
import joblib
import ast

def convert_to_list(string):
    try:
        # Remove colchetes e converte para lista de floats
        return [float(i) for i in string.strip('[]').split()]
    except ValueError:
        return string

def preprocess_data(features_csv_path):
    # Carregar os dados de características
    data = pd.read_csv(features_csv_path)

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

    # Remover colunas desnecessárias (genre, emotion, style)
    data = data.drop(columns=['genre', 'emotion', 'style'], errors='ignore')

    return data

def main():
    # Caminhos dos modelos salvos
    model_genre_path = 'models/model_genre.joblib'
    model_emotion_path = 'models/model_emotion.joblib'
    model_style_path = 'models/model_style.joblib'

    # Caminho do arquivo CSV com as características da música
    features_csv_path = sys.argv[1]

    # Carregar os modelos
    model_genre = joblib.load(model_genre_path)
    model_emotion = joblib.load(model_emotion_path)
    model_style = joblib.load(model_style_path)

    # Pré-processar os dados
    data = preprocess_data(features_csv_path)

    # Fazer previsões
    genre_prediction = model_genre.predict(data)
    emotion_prediction = model_emotion.predict(data)
    style_prediction = model_style.predict(data)

    # Imprimir as previsões
    print(f'Gênero previsto: {genre_prediction[0]}')
    print(f'Emoção prevista: {emotion_prediction[0]}')
    print(f'Estilo previsto: {style_prediction[0]}')

if __name__ == "__main__":
    main()
