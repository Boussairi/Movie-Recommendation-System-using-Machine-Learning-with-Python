# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np 
import pandas as pd
import pickle
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


input_data = "Spiderman"





loaded_model = pickle.load(open("C:/Users/Xps/IA compet/Movie recommendation/trained_model.sav",'rb'))

df=pd.read_csv('C:/Users/Xps/Desktop/S4/IA compet/movies/movies.csv')

selected_features = ['genres','keywords','overview','production_companies','production_countries','spoken_languages','tagline','cast','director']

for feature in selected_features:
    df[feature] = df[feature].fillna('')
    
vectorizer = TfidfVectorizer()
df['descreption']=df['genres']+', '+df['keywords']+', '+df['overview']+', '+df['production_companies']+', '+df['production_countries']+', '+df['spoken_languages']+', '+df['tagline']+', '+df['cast']+', '+df['director']
feature_vectors = vectorizer.fit_transform(df['descreption'])
similarity = cosine_similarity(feature_vectors)





list_of_all_titles = df['original_title'].tolist()

find_close_match = difflib.get_close_matches(input_data, list_of_all_titles)

close_match = find_close_match[0]

index_of_the_movie = df[df.original_title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
	index = movie[0]
	title_from_index = df[df.index==index]['original_title'].values[0]
	if (i<30):
		print(i, '.',title_from_index)
		i+=1 