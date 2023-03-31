# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 18:48:39 2023

@author: Xps
"""

import streamlit as st
import numpy as np 
import pandas as pd
import pickle
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


loaded_model = pickle.load(open("C:/Users/Xps/IA compet/Movie recommendation/trained_model.sav",'rb'))

def Movies_prediction(input_movie):
    df=pd.read_csv('C:/Users/Xps/Desktop/S4/IA compet/movies/movies.csv')

    selected_features = ['genres','keywords','overview','production_companies','production_countries','spoken_languages','tagline','cast','director']

    for feature in selected_features:
        df[feature] = df[feature].fillna('')
        
    vectorizer = TfidfVectorizer()
    df['descreption']=df['genres']+', '+df['keywords']+', '+df['overview']+', '+df['production_companies']+', '+df['production_countries']+', '+df['spoken_languages']+', '+df['tagline']+', '+df['cast']+', '+df['director']
    feature_vectors = vectorizer.fit_transform(df['descreption'])
    similarity = cosine_similarity(feature_vectors)





    list_of_all_titles = df['original_title'].tolist()

    find_close_match = difflib.get_close_matches(input_movie, list_of_all_titles)

    close_match = find_close_match[0]

    index_of_the_movie = df[df.original_title == close_match]['index'].values[0]

    similarity_score = list(enumerate(similarity[index_of_the_movie]))

    sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

    
    #print('Movies suggested for you : \n')

    i = 1
    liste= ' '
    for movie in sorted_similar_movies:
    	index = movie[0]
    	title_from_index = df[df.index==index]['original_title'].values[0]
    	if (i<21):
    		liste= '\n' + liste + str(i)+'.'+ str(title_from_index) + '\n\n'
    		i+=1
    return liste   
        

#a=Movies_prediction("inception")
#print(a)


def main():
    st.title("MOVIES RECOMMENDATION SYSTEM")
    
    movie_input= st.text_input("Enter Your Favorite movie ")
    
    output = ''
    
    if st.button("Recommendation Result "):
        output=Movies_prediction(movie_input)
    
    
    st.success(output)
        
    
if __name__=='__main__':
    main()
    
   
    
    
    
    
    