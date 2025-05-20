import pandas as pd
import re
import unicodedata
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
import nltk
from nltk.corpus import stopwords
 
nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))  

def preprocess_lyrics(lyrics):
    if pd.isna(lyrics):
        return ""
    lyrics = lyrics.lower()  
    lyrics = re.sub(r'[^\w\s]', '', lyrics)  
    lyrics = ''.join(char for char in unicodedata.normalize('NFD', lyrics) if unicodedata.category(char) != 'Mn')  
    lyrics = ' '.join(word for word in lyrics.split() if word not in stop_words)  
    return lyrics

def recommend_songs(user_input, songs_df):
    user_input = preprocess_lyrics(user_input)  

    all_lyrics = songs_df['cleaned_song_lyrics'].tolist() + [user_input]  

    vectorizer = CountVectorizer(binary=True)  
    bow_matrix = vectorizer.fit_transform(all_lyrics)

    similarities = cosine_similarity(bow_matrix[-1], bow_matrix[:-1])[0]  
    angles = [round(math.degrees(math.acos(sim)), 1) if -1 <= sim <= 1 else None for sim in similarities]

    songs_df['similarity'] = similarities
    songs_df['angle_degrees'] = angles

    recommended_songs = songs_df.sort_values(by='similarity', ascending=False).head(10)  
    return recommended_songs[['song_name', 'artist', 'song_lyrics']]

 
data_path = 'bossa_nova_songs_portugues.xlsx'
songs_df = pd.read_excel(data_path, engine="openpyxl")


songs_df['cleaned_song_lyrics'] = songs_df['song_lyrics'].apply(preprocess_lyrics)


user_input = input("Digite a letra ou parte da letra da música para recomendar: ")
recommendations = recommend_songs(user_input, songs_df)


MAX_LETRA = 150

print("\nMúsicas recomendadas:")
for index, row in recommendations.iterrows():
    letra = row['song_lyrics']
    letra_exibida = (letra[:MAX_LETRA] + "...") if len(letra) > MAX_LETRA else letra

    print(f"Título: {row['song_name']}")
    print(f"Artista: {row['artist']}")
    print(f"Letra (trecho):\n{letra_exibida}\n")
