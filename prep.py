from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import timeit

# Add goal ratio and end game status
def goal_handler(matches):
    ratio_vector = []
    status_vector = []
    i = 0
    for index, row in matches.iterrows():
        if row['away_team_goal'] != 0:
            ratio_vector.append(row['home_team_goal']/row['away_team_goal'])
        else:
            ratio_vector.append(row['home_team_goal'])
        if ratio_vector[i] > 1:
            status_vector.append(2)  # home team win
        elif ratio_vector[i] < 1:
            status_vector.append(1)  # away team win
        else:
            status_vector.append(0)  # draw
        i += 1
    matches['goal_ratio'] = ratio_vector
    matches['status'] = status_vector
    matches.drop(['home_team_goal', 'away_team_goal'], axis=1)
    return matches

def fill_team_na(row, mean_values):
    little = mean_values[0]
    lots = mean_values[1]
    normal = mean_values[2]
    if row['buildUpPlayDribblingClass'] == 'Little':
        return little
    if row['buildUpPlayDribblingClass'] == 'Lots':
        return lots
    return normal