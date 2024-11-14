import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#************Helper functions************************
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    try:
        return df[df.title == title]["index"].values[0]
    except:
        print("Sorry, unable to fetch recommendations :( ")
#****************************************************


#1. Read CSV File
df = pd.read_csv("movie_dataset.csv")
#print(df.columns)

#2. Select features
features = ['keywords', 'cast', 'genres', 'director']

#3. Create a column in Df which combines all selected features
for feature in features:
    df[feature] = df[feature].fillna('') #To replace NaN with empty string

def combine_features(row):
    try:
        return row['keywords'] +" "+row['cast']+" "+row['genres']+" "+row['director'] 
    except: 
        print("Error:", row)

df["combined_features"] = df.apply(combine_features, axis=1) #axis=1 to combine vertically

#print("Combined Features:", df['combined_features'].head())

#4. Create count matrix from this new combined column
count_vectorizer= CountVectorizer()

count_matrix = count_vectorizer.fit_transform(df['combined_features'])
#print(count_matrix)

#5. Compute the Cosine Similarity
cosine_sim = cosine_similarity(count_matrix)
#print(cosine_sim)
movie_user_likes = str(input("Enter a Movie Name: "))

#6. Get the index of a movie from its title.
movie_index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(cosine_sim[movie_index])) #To get a list of tuples.

#7. Get a list of similar movies in descending order of similarity score
sorted_similar_movies = sorted(similar_movies, key = lambda x:x[1], reverse = True) #Sort by x[1](similarity scores) and put it in descending order(reverse=True).

#8. Print titles of first 10 movies
i=0
for movie in sorted_similar_movies:
    print(get_title_from_index(movie[0]))
    i=i+1
    if i>10:
        break

