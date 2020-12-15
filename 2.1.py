#!/usr/bin/env python
# coding: utf-8

import json

import numpy as np
import pandas as pd

def sim(u, v):
    """
    Takes two vectors of the same size, 
    where two elements with same indices relate to the same movie
    """
    if len(u) != len(v):
        raise ValueError("Vectors must have same size")

    sumprod = 0.0 # Sum of products (u_i * v_i), numerator
    sumsquare_u = 0.0 # Sum of squares of u_i
    sumsquare_v = 0.0 # Sum of squares of v_i

    for i in range(len(u)):
        if u[i] == -1 or v[i] == -1:
            continue
        sumprod += u[i] * v[i]
        sumsquare_u += u[i] * u[i]
        sumsquare_v += v[i] * v[i]
    return sumprod / (np.sqrt(sumsquare_u) * np.sqrt(sumsquare_v))

def assume_rating(data, u_i, i, k=4, rated_filter=None):
    """
    Calculates assumed rating of movie (i) 
    for user ratings data line (u_i)
    from (data)
    """
    # Get user vector
    u = data[u_i]
    
    # Calculate averages for u
    avg_u = np.mean(u[np.where(u != -1)])
    
    # Find k nearest neighbors and their similarity metric
    knn = user_knn(data, u_i, i, k=k, rated_filter=rated_filter)
    
    sim_product_sum = 0.0  # Numerator of the fraction
    sim_sum = 0.0  # Denominator of the fraction
    
    # For all users in the kNN result, calculate 
    # both numerator and denominator values
    for j in range(len(knn)):
        v_i = int(knn[j][0])  # Index of neighbor
        sim = knn[j][1]  # Calculated metric
        
        # Calculate averages for v
        v = data[v_i]
        avg_v = np.mean(v[np.where(v != -1)])
        
        # Get user v rating of i
        rating = v[i]
               
        # Calculate and update numerator
        sim_product_sum += sim * (rating - avg_v)
        
        # Update denominator
        sim_sum += np.abs(sim)
    
    return avg_u + (sim_product_sum / sim_sum)


def user_knn(data, u_i, i, k=4, metric=sim, rated_filter=None):
    """
    Finds first k similar users that rated the movie.
    
    Returns array of user indices alongside calculated metric value
    """
    # We're filtering, therefore we need to know indices
    indices = np.indices((len(data), 1))[0,:]
    rated = np.append(indices, data, axis=1)
    
    # Get rid of user that we're calculating data for
    rated = rated[rated[:, 0] != u_i]
        
    # Apply external filter if provided
    if rated_filter is not None:
        rated = rated_filter(rated, u_i, i)
        
    # Return None if no neighbors were found
    if len(rated) == 0:
        return None
    
    # Extract user ratings to calculate similarities for
    user_ratings = data[u_i]
    
    # Calculate similarity metric for all users who rated the movie
    result = np.zeros((len(rated), 2))
    for j in range(len(rated)):
        result[j][0] = rated[j][0]
        result[j][1] = metric(user_ratings, rated[j, 1:])
        
    # Descending sort result by similarity
    result = result[result[:,1].argsort()[::-1]]
    
    # Get first k neighbors only
    result = result[:k]
    
    return result

def assume_ratings_for(u_i, k=4):
    """
    Assumes ratings of movies using kNN. 
    
    Returns a dict containing movie names as keys 
    and ratings assumptions as values.
    """
    # Who needs readability anyway? Load csv and immediately
    # convert it to numpy array
    data = pd.read_csv('data.csv')
    np_data = data.to_numpy()
    raw_data = np_data[:,1:]
    
    # Offset u_i
    u_i -= 1

    # Calculate rating assumptions for all non-rated movies of user u_i
    result = {
        data.columns[1:][i].strip(): 
        np.round(assume_rating(raw_data, u_i, i, k=k), decimals=3)
        for i in range(len(raw_data[0]))
        if raw_data[u_i][i] == -1
    }
    return result

def find_weekend_home_movie(u_i):
    """
    The same process really, with a slight modification.
    All kNN neighbors are filtered to match both criteria:
        1. They saw the movie on a weekend
        2. They did it at home
    """
    data = pd.read_csv('data.csv')
    day_data = pd.read_csv('context_day.csv')
    place_data = pd.read_csv('context_place.csv')
    
    np_data = data.to_numpy()
    raw_data = np_data[:,1:]
    
    u_i -= 1
    
    # Helper method that has loaded context data in its scope
    def filter_by_wd_and_place(rated, filter_u_i, i):
        """
        Filters all users who didn't watch the movie at home on weekend
        """

        # I'm ashamed of this code, but so should be pandas developers :\
        fday = day_data[[day_data.columns[i + 1]]].copy()
        fday = fday.rename({fday.columns[0]: "dow"}, axis=1)
        fday['dow'] = fday['dow'].apply(lambda x: x.strip())
        fday['l'] = place_data[[place_data.columns[i + 1]]].copy()
        fday['l'] = fday['l'].apply(lambda x: x.strip())
        fday = fday[fday.dow.isin(["Sat", "Sun"]) & fday.l.eq("h")]

        neighbors = list(fday.index)
        new_rated = rated[np.where(np.isin(rated[:, 0], neighbors))]
        
        return new_rated
    
    user_vector = raw_data[u_i]
    movies_cnt = len(raw_data[0])
    result = np.zeros((movies_cnt, 2))
    for i in range(movies_cnt):
        if user_vector[i] != -1:
            continue
        result[i][0] = i
        result[i][1] = assume_rating(raw_data, u_i, i, rated_filter=filter_by_wd_and_place)
                
    # Descending sort result by rating
    result = result[result[:,1].argsort()[::-1]]
    # Get first result
    result = result[:1]
    if result[0][1] == 0:
        return {
            "Not found": 0.0
        }
    
    return {
        data.columns[1:][int(result[0][0])].strip(): 
        np.round(result[0][1], decimals=3)
    }

def recommend(u_i):
    return {
        "user": u_i,
        "1": assume_ratings_for(u_i),
        "2": find_weekend_home_movie(u_i)
    }

if __name__ == "__main__":
    with open('result.json', 'w') as f:
        json.dump(recommend(2), f)

