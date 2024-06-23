import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

movie = pd.read_csv('datasets/ml-100k/u.item', sep='|', encoding='latin-1', header=None, usecols=[0, 1, 2], names=['movieId', 'title', 'genres'])
movieRatings = pd.read_csv('datasets/ml-100k/u.data', sep='\t', header=None, names=['userId', 'movieId', 'rating', 'timestamp'])

movie['genres'] = movie['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])

data = pd.merge(movieRatings, movie, on='movieId')

genreSet = set(g for genres in movie['genres'] for g in genres)
for genre in genreSet:
    data[genre] = data['genres'].apply(lambda x: 1 if genre in x else 0)

features = list(genreSet) + ['userId', 'movieId']
X = data[features]
y = data['rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

def recommend_movies(user_id, model, data, num_recommendations=5):
    user_data = data[data['userId'] == user_id]
    user_ratings = user_data[['movieId', 'rating']]
    movie_ids = user_ratings['movieId'].tolist()
    
    recommendations = []
    for movie_id in movie_ids:
        movie_features = data[data['movieId'] == movie_id].iloc[0][features].values.reshape(1, -1)
        predicted_rating = model.predict(movie_features)
        recommendations.append((movie_id, predicted_rating))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    recommended_movie_ids = [movie[0] for movie in recommendations[:num_recommendations]]
    
    recommended_movies = movie[movie['movieId'].isin(recommended_movie_ids)]
    return recommended_movies

user_id = 2
recommended_movies = recommend_movies(user_id, model, data)
print(recommended_movies)
