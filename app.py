import streamlit as st
import pickle
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

add_bg_from_local('image.jpg')


def moviePrediction(movie_name):
    warnings.filterwarnings('ignore')

    column_names = ["user_id", "item_id", "rating", "timestamp"]

    df = pd.read_csv("ml-100k/u.data", sep='\t', names=column_names)

    movies_title = pd.read_csv("ml-100k/u.item", encoding="ISO-8859-1", sep="\|", header=None)

    movies_title = movies_title[[0, 1]]

    movies_title.columns = ['item_id', 'title']

    df = pd.merge(df, movies_title, on="item_id")

    sns.set_style('white')

    df.groupby('title').mean()['rating'].sort_values(ascending=False).head()

    df.groupby('title').count()['rating'].sort_values(ascending=False)

    rating = pd.DataFrame(df.groupby('title').mean()['rating'])

    rating['number of ratings'] = pd.DataFrame(df.groupby('title').count()['rating'])

    rating.sort_values(by='rating', ascending=False)

    sns.jointplot(x='rating', y='number of ratings', data=rating, alpha=0.5)

    moviematrix = df.pivot_table(index="user_id", columns="title", values="rating")

    rating.sort_values("number of ratings", ascending=False).head()

    starwars_user_ratings = moviematrix['Star Wars (1977)']
    starwars_user_ratings.head()

    similar_to_starwars = moviematrix.corrwith(starwars_user_ratings)

    corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])

    corr_starwars.dropna(inplace=True)

    corr_starwars.sort_values('Correlation', ascending=False).head(10)

    corr_starwars = corr_starwars.join(rating['number of ratings'])

    corr_starwars[corr_starwars['number of ratings'] > 100].sort_values('Correlation', ascending=False)

    def predict_movies(movie_name):
        movie_user_ratings = moviematrix[movie_name]
        similar_to_movie = moviematrix.corrwith(movie_user_ratings)

        corr_movie = pd.DataFrame(similar_to_movie, columns=['Correlation'])
        corr_movie.dropna(inplace=True)

        corr_movie = corr_movie.join(rating['number of ratings'])
        corr_movie['Movie'] = corr_movie.index
        predictions = corr_movie[corr_movie['number of ratings'] > 100].sort_values('Correlation', ascending=False)

        predictions = predictions.head()
        predictions = np.array(predictions)
        list = predictions[:, 2]
        return list

    return predict_movies(movie_name)


movies_dict = pickle.load(open('movie_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)

st.title('Movie Recommendation System')

selectedMovieName = st.selectbox(
    'Select the name of the movie',
    movies['title'].values)

if st.button('Recommend'):
    recommendation = moviePrediction(selectedMovieName)
    for i in recommendation:
        st.write(i)
