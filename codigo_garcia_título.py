import pandas as pd
import re
import unicodedata
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math
import nltk
from nltk.corpus import stopwords

# Baixar stopwords (só se ainda não tiver baixado)
nltk.download('stopwords')

# Lista de stopwords em português
stop_words = set(stopwords.words('portuguese'))

# Função para limpar os títulos das músicas
def preprocess_song_name(song_name):
    if pd.isna(song_name):
        return ""

    title = re.sub(r"[\[\]']", "", song_name)
    title = re.sub(r"[^\w\s]", "", title)
    title = title.lower()
    title = ''.join(
        char for char in unicodedata.normalize('NFD', title)
        if unicodedata.category(char) != 'Mn'
    )

    # Remove stop words
    words = title.split()
    filtered_words = [word for word in words if word not in stop_words]

    return " ".join(filtered_words)

# Função de recomendação
def recommend_songs(user_input, songs_df):
    user_input = preprocess_song_name(user_input)
    user_vector = np.zeros(len(unique_words))

    for word in user_input.split():
        if word in word_index:
            user_vector[word_index[word]] = 1

    similarities = cosine_similarity([user_vector], word_bow.values)[0]
    angles = [round(math.degrees(math.acos(sim)), 1) if -1 <= sim <= 1 else None for sim in similarities]

    songs_df['similarity'] = similarities
    songs_df['angle_degrees'] = angles

    recommended_songs = songs_df.sort_values(by='similarity', ascending=False).head(10)
    return recommended_songs[['song_name', 'artist']]

# Carregando o dataset
data_path = 'bossa_nova_songs_portugues.xlsx'  # Arquivo na mesma pasta do script
songs_df = pd.read_excel(data_path)

# Limpando os nomes das músicas
songs_df['cleaned_song_name'] = songs_df['song_name'].apply(preprocess_song_name)

# Criando o índice de palavras únicas
unique_words = sorted(set(" ".join(songs_df['cleaned_song_name']).split()))
word_index = {word: idx for idx, word in enumerate(unique_words)}

# Criando a matriz Bag of Words
word_bow = pd.DataFrame(0, index=songs_df.index, columns=unique_words)
for idx, song in enumerate(songs_df['cleaned_song_name']):
    for word in song.split():
        if word in word_bow.columns:
            word_bow.loc[idx, word] = 1

# Entrada do usuário e recomendação
user_input = input("Digite o nome da música para recomendar: ")
recommendations = recommend_songs(user_input, songs_df)

# Exibindo somente nomes e artistas recomendados
print("\nMúsicas recomendadas:")
for idx, row in recommendations.iterrows():
    print(f"{row['song_name']} - {row['artist']}")
