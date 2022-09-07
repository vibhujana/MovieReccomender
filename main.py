
from collections import defaultdict
import difflib
import gzip
import re
import shutil
from math import nan
import json

import numpy as np
import pandas as pd
from IPython.core.display import display
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
LIKED_FILE_NAME = "likemovies.txt"
REVIEW_WEIGHT = 1
COUNT_WEIGHT = 3

    



def combine_title_data():
    files = ["./titlecrew.tsv", "./titleratings.tsv"]
    tsv1 = pd.read_csv("./titlebasics.tsv", sep='\t')
    tsv2 = pd.read_csv("./titlecrew.tsv", sep='\t')
    output = pd.merge(tsv1, tsv2, on='tconst', how='inner')
    for f in files:
        tsv = pd.read_csv(f, sep='\t')
        output = pd.merge(output, tsv, on='tconst', how='inner')
    output.to_csv("./movieData.tsv", sep='\t', header=True, index=False)


def get_imdb_data():
    links = ["https://datasets.imdbws.com/name.basics.tsv.gz", "https://datasets.imdbws.com/title.akas.tsv.gz",
             "https://datasets.imdbws.com/title.basics.tsv.gz", "https://datasets.imdbws.com/title.crew.tsv.gz",
             "https://datasets.imdbws.com/title.episode.tsv.gz", "https://datasets.imdbws.com/title.principals.tsv.gz",
             "https://datasets.imdbws.com/title.ratings.tsv.gz"]
    for link in links:
        url1 = link
        file_name1 = re.split(pattern='/', string=url1)[-1]
        # r1 = request.urlretrieve(url=url1, filename=file_name1)
        data = re.split(pattern=r'\.', string=file_name1)[0] + re.split(pattern=r'\.', string=file_name1)[1] + ".tsv"
        with gzip.open(file_name1, 'rb') as f_in:
            with open(data, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        combine_title_data()

# while True:
#     schedule.run_pending()
#     break

def clean_data():
    movies = pd.read_csv("movieData.tsv", sep="\t")
    movies = movies.set_index('tconst')
    og_movies = movies.copy()
    og_movies.drop(['originalTitle', 'isAdult', 'endYear', 'directors_x', 'writers_x', 'directors_y', 'writers_y'], axis=1,
            inplace=True)
    typeMapping = ['movie', 'tvSeries', 'tvShort', 'tvMovie', 'tvMiniSeries', 'tvSpecial']
    og_movies = og_movies.drop(og_movies[~og_movies["titleType"].isin(typeMapping)].index)
    movies.drop(['originalTitle', 'isAdult', 'endYear', 'directors_x', 'writers_x', 'directors_y', 'writers_y','primaryTitle'], axis=1,
                inplace=True)
    typeMapping = {'movie': 0, 'tvSeries': 1, 'tvShort': 2, 'tvMovie': 3, 'tvMiniSeries': 4, 'tvSpecial': 5}
    badGenres = ["\\N", nan]
    genreMapping = {'Documentary': 1, 'Short': 2, 'Animation': 3, 'Comedy': 4, 'Romance': 5, 'Sport': 6, 'News': 7,
                    'Drama': 8, 'Fantasy': 9, 'Horror': 10, 'Biography': 11, 'Music': 12, 'War': 13, 'Crime': 14,
                    'Western': 15, 'Family': 16, 'Adventure': 17, 'Action': 18, 'History': 19, 'Mystery': 20,
                    'Sci-Fi': 21, 'Musical': 22, 'Thriller': 23, 'Film-Noir': 24, 'Game-Show': 25, 'Talk-Show': 26,
                    'Reality-TV': 27, 'Adult': 28, nan: 0}
    movies = movies.drop(movies[~movies["titleType"].isin(typeMapping.keys())].index)
    movies = movies.drop(movies[movies["genres"].isin(badGenres)].index)
    movies[["genre1", 'genre2', 'genre3']] = movies.genres.str.split(",", expand=True)
    movies = movies.drop(movies[movies["startYear"] == "\\N"].index)
    movies = movies.drop(movies[movies["runtimeMinutes"] == "\\N"].index)
    movies = movies.replace(
        {'titleType': typeMapping, 'genre1': genreMapping, 'genre2': genreMapping, 'genre3': genreMapping})
    movies['titleType'] = MinMaxScaler().fit_transform(np.array(movies['titleType']).reshape(-1, 1))
    movies['startYear'] = MinMaxScaler().fit_transform(np.array(movies['startYear']).reshape(-1, 1))
    movies['runtimeMinutes'] = MinMaxScaler().fit_transform(np.array(movies['runtimeMinutes']).reshape(-1, 1))
    movies['averageRating'] = MinMaxScaler().fit_transform(np.array(movies['averageRating']).reshape(-1, 1))
    movies['numVotes'] = MinMaxScaler().fit_transform(np.array(movies['numVotes']).reshape(-1, 1))
    movies['genre1'] = MinMaxScaler().fit_transform(np.array(movies['genre1']).reshape(-1, 1))
    movies['genre2'] = MinMaxScaler().fit_transform(np.array(movies['genre2']).reshape(-1, 1))
    movies['genre3'] = MinMaxScaler().fit_transform(np.array(movies['genre3']).reshape(-1, 1))
    movies.drop(['genres'],axis = 1, inplace= True)
    og_movies.to_csv("./cleanedMovieData.tsv", sep='\t', header=True)
    movies.to_csv("./normalizedMovieData.tsv",sep='\t',header=True)
def start_up(exe):
    if exe:
        get_imdb_data()
        clean_data()


def add_title(data):
    user_input = input(
        "Type 'like' to add movie you like.  Type 'done' to get reccomendations: ")
    while (user_input != 'done'):
        if (user_input == 'like'):
            with open(LIKED_FILE_NAME, "a") as f:
                user_input = input("Type the movie you like: ")
                close_to_input = set(
                    difflib.get_close_matches(user_input, data['primaryTitle'].tolist(), n=10, cutoff=0.5))
                possible_movies = pd.DataFrame()
                for movie in close_to_input:
                    frames = [possible_movies, data.loc[data['primaryTitle'] == movie]]
                    possible_movies = pd.concat(frames)
                display(possible_movies.to_string())
                user_input = input("type the tconst of the thing you want: ")
                f.write(user_input + "\n")
        user_input = input(
            "Type 'like' to add movie you like. Type 'done' to get reccomendations: ")

def calc_avg_like(movies):
    likes = open(LIKED_FILE_NAME, "r")
    lines = likes.readlines()
    avg_vector = [0, 0, 0, 0, 0, 0, 0, 0]
    count = 0
    most_common_genres = defaultdict(int)
    for line in lines:
        if 'tt' in line:
            data = movies.loc[movies["tconst"] == line.strip()].values.tolist()[0]
            # print(individual.values.tolist()) #important way to convert line to a list
            avg_vector[0] += float(data[1]) #type of medium
            avg_vector[1] += float(data[2]) #year
            avg_vector[2] += float(data[3]) #runtime
            most_common_genres[data[6]] += 1
            most_common_genres[data[7]] += 1 
            most_common_genres[data[8]] += 1   
            avg_vector[3] += (float)(data[4]) #review
            avg_vector[4] += (float)(data[5]) #numVotes
            count += 1
    for i in range(len(avg_vector)):
        avg_vector[i] = avg_vector[i] / count
    most_common_genres = sorted(most_common_genres.items(),key = lambda x : x[1], reverse = True)
    avg_vector[5] = most_common_genres[0][0]
    avg_vector[6] = most_common_genres[1][0]
    avg_vector[7] = most_common_genres[2][0]
    return avg_vector

def similarity_to_avg(avg_vector,movies):
    movies = movies.set_index("tconst")
    likes = open(LIKED_FILE_NAME, "r")
    lines = likes.readlines()
    for line in lines:
        if 'tt' in line:
            movies = movies.drop(line.strip())
    movies['similarity'] = cosine_similarity(movies,np.array(avg_vector).reshape(1,-1))
    return movies.astype('float').nlargest(10,'similarity')
def get_data_of_rec_titles(similarity,cleanedMovieData):
    return cleanedMovieData[cleanedMovieData['tconst'].isin(similarity.index)]

