import csv
import re
import sys
from math import sqrt, nan
from urllib import request
import gzip
import shutil

import numpy as np
import schedule
import pickle
import difflib
import pandas as pd
import os
from IPython.core.display import display
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler


# https://medium.com/analytics-vidhya/content-based-recommender-systems-in-python-2b330e01eb80

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
        r1 = request.urlretrieve(url=url1, filename=file_name1)
        data = re.split(pattern=r'\.', string=file_name1)[0] + re.split(pattern=r'\.', string=file_name1)[1] + ".tsv"
        with gzip.open(file_name1, 'rb') as f_in:
            with open(data, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        combine_title_data()


# get_imdb_data()
while True:
    schedule.run_pending()
    break


def add_title():
    user_input = input(
        "Type 'like' to add movie you like.  Type 'done' to get reccomendations")
    data = pd.read_csv("movieData.tsv", sep="\t")
    data.drop(['originalTitle', 'isAdult', 'endYear', 'directors_x', 'writers_x', 'directors_y', 'writers_y'], axis=1,
              inplace=True)
    while (user_input != 'done'):
        if (user_input == 'like'):
            with open("likemovies.txt", "a") as f:
                user_input = input("Type the movie you like: ")
                close_to_input = set(
                    difflib.get_close_matches(user_input, data['primaryTitle'].tolist(), n=10, cutoff=0.5))
                possible_movies = pd.DataFrame()
                for movie in close_to_input:
                    frames = [possible_movies, data.loc[data['primaryTitle'] == movie]]
                    possible_movies = pd.concat(frames)
                display(possible_movies.to_string())
                user_input = input("type the tconst of the thing you want")
                f.write(user_input + "\n")
        user_input = input(
            "Type 'like' to add movie you like. Type 'done' to get reccomendations")


add_title()


def distance_formula(a, b):
    a = np.array(a)
    b = np.array(b)
    return cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))


def rec_title():
    movies = pd.read_csv("movieData.tsv", sep="\t")
    og_movies = movies.copy()
    movies.drop(['originalTitle', 'isAdult', 'endYear', 'directors_x', 'writers_x', 'directors_y', 'writers_y'], axis=1,
                inplace=True)
    typeMapping = {'movie': 0, 'tvSeries': 1, 'tvShort': 2, 'tvMovie': 3, 'tvMiniSeries': 4, 'tvSpecial': 5,
                   'videoGame': 6}
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
    likes = open("likemovies.txt", "r")
    lines = likes.readlines()
    avg_vector = [0, 0, 0, 0, 0, 0, 0, 0]
    count = 0
    for line in lines:
        if 'tt' in line:
            data = movies.loc[movies["tconst"] == line.strip()].values.tolist()[0]
            # print(individual.values.tolist()) #important way to convert line to a list
            avg_vector[0] += float(data[1]) #type of medium
            avg_vector[1] += float(data[3]) #year
            avg_vector[2] += float(data[4]) #runtime
            avg_vector[3] += float(data[8]) #genre1
            avg_vector[4] += float(data[9]) #genre2
            avg_vector[5] += float(data[10]) #genre3
            avg_vector[6] += (float)(data[6]*3) #review
            avg_vector[7] += (float)(data[7]*7) #numVotes
            count += 1
    for elem in avg_vector:
        elem / count
    m = ""
    cos_sim = 0
    temp = [0, 0, 0, 0, 0, 0, 0, 0]
    data = movies.values.tolist()
    best_data = 0
    lines = [x[:-1] for x in lines]
    for movie in data:
        if movie[0] not in lines:
            temp[0] = float(movie[1])
            temp[1] = float(movie[3])
            temp[2] = float(movie[4])
            temp[3] = float(movie[8])
            temp[4] = float(movie[9])
            temp[5] = float(movie[10])
            temp[6] = (float)(movie[6]*5)
            temp[7] = (float)(movie[7]*7)
            perfect = distance_formula(avg_vector,avg_vector)
            if distance_formula(temp, avg_vector) > cos_sim and distance_formula(temp, avg_vector) < perfect:
                cos_sim = distance_formula(temp, avg_vector)
                print(cos_sim)
                m = movie[2]
                print(m)
                best_data = og_movies[og_movies["tconst"] == movie[0]]
    return m, best_data.to_string()


print(rec_title())
