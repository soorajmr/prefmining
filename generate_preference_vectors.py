
# Movie Preference Vectors Using Matrix Factorization
# 
# 
# Use graphlab package to do matrix factorization on Movielens-20m movie rating data. The purpose is to get preference vectors for movies.

import pandas as pd
import matplotlib.pyplot as plt
import graphlab as gl
import itertools
import pickle
import numpy np

ratings = pd.read_csv("data/ml-20m/ratings.csv")


# Exclude rarely rated movies and highly active users
# 
# To reduce noise and also to bring down the size of the data set, we exclude moveis rated only by a handful of people and users who have rated too many movies.

fig, axes = plt.subplots(2, figsize=(20, 7))

movie_counts = ratings.groupby("movieId").count().userId
user_counts = ratings.groupby("userId").count().movieId
movie_counts.plot(kind='hist', bins=range(0, 150, 3), ax=axes[0]).set_title("Number of movies Vs Number of ratings received")
user_counts.plot(kind='hist', bins=range(0, 500, 10), ax=axes[1]).set_title("Number of users Vs Number of movies rated")
user_counts = dict(user_counts)
movie_counts = dict(movie_counts)


ratings['mcounts'] = ratings.apply(lambda x: movie_counts[x.movieId], axis=1)
ratings['ucounts'] = ratings.apply(lambda x: user_counts[x.userId], axis=1)

# Ideally the comparisons should be done sequentially as one affects the other. Doing this as an approximation.
ratings_small = ratings[(ratings.mcounts > 100) # Movies that at least 100 people rated
                         & (ratings.ucounts < 150)] # Users who rated less than 150 movies


ratings_sf = gl.SFrame(ratings_small)
(train_set, test_set) = ratings_sf.random_split(0.7)

regularization_vals = [0.00001, 0.000001]
vector_lengths = [50, 80]

models = [gl.factorization_recommender.create(train_set, 'userId', 'movieId', 'rating',
                                              max_iterations=100, num_factors=factors, 
                                              regularization=reg, solver='sgd')
          for reg, factors in itertools.product(regularization_vals, vector_lengths)]

# Find the best model
best_model = None
smallest_error = 10.0
for m in models:
    rmse = gl.evaluation.rmse(test_set['rating'], m.predict(test_set))
    if(rmse < smallest_error):
        smallest_error = rmse
        best_model = m

print("The best model: Training RMSE %f, test RMSE %f" % (best_model['training_rmse'], smallest_error))

#print(best_model)

vecotrs = np.array(best_model.coefficients['movieId']['factors'])
movieIds = map(int, list(best_model.coefficients['movieId']['movieId']))
with open('data/output/pref_vectors.pickle', 'wb') as f:
    pickle.dump(vecotrs, f)

with open('data/output/pref_movieids.pickle', 'wb') as f:
    pickle.dump(movieIds, f)
