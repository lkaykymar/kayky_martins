import pandas as pd
import re
import unicodedata
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
import nltk
from nltk.corpus import stopwords

# Baixando a lista de stopwords (só na primeira vez)
nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))  # Stopwords em português

def preprocess_lyrics(lyrics):
    if pd.isna(lyrics):
        return ""
    lyrics = lyrics.lower()  # Tudo minúsculo
    lyrics = re.sub(r'[^\w\s]', '', lyrics)  # Remove pontuações
    lyrics = ''.join(char for char in unicodedata.normalize('NFD', lyrics) if unicodedata.category(char) != 'Mn')  # Remove acentos
    lyrics = ' '.join(word for word in lyrics.split() if word not in stop_words)  # Remove stopwords
    return lyrics

def recommend_songs(user_input, songs_df):
    user_input = preprocess_lyrics(user_input)  # Pré-processa a entrada

    all_lyrics = songs_df['cleaned_song_lyrics'].tolist() + [user_input]  # Letras + entrada

    vectorizer = CountVectorizer(binary=True)  # Bag of words binário
    bow_matrix = vectorizer.fit_transform(all_lyrics)

    similarities = cosine_similarity(bow_matrix[-1], bow_matrix[:-1])[0]  # Similaridade com todas as músicas
    angles = [round(math.degrees(math.acos(sim)), 1) if -1 <= sim <= 1 else None for sim in similarities]

    songs_df['similarity'] = similarities
    songs_df['angle_degrees'] = angles

    recommended_songs = songs_df.sort_values(by='similarity', ascending=False).head(10)  # Top 10
    return recommended_songs[['song_name', 'artist', 'song_lyrics']]

# Ajuste o caminho para o arquivo Excel (ou deixe na mesma pasta do script)
data_path = 'bossa_nova_songs_portugues.xlsx'
songs_df = pd.read_excel(data_path, engine="openpyxl")

# Limpa as letras
songs_df['cleaned_song_lyrics'] = songs_df['song_lyrics'].apply(preprocess_lyrics)

# Entrada do usuário
user_input = input("Digite a letra ou parte da letra da música para recomendar: ")
recommendations = recommend_songs(user_input, songs_df)

# Limite de caracteres para mostrar do trecho da letra
MAX_LETRA = 150

print("\nMúsicas recomendadas:")
for index, row in recommendations.iterrows():
    letra = row['song_lyrics']
    letra_exibida = (letra[:MAX_LETRA] + "...") if len(letra) > MAX_LETRA else letra

    print(f"Título: {row['song_name']}")
    print(f"Artista: {row['artist']}")
    print(f"Letra (trecho):\n{letra_exibida}\n")
