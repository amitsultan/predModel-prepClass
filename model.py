import sqlite3
import pandas as pd
import dataPrep as prep
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
from sklearn.model_selection import train_test_split


conn = sqlite3.connect('input/database.sqlite')

player_attr = pd.read_sql("SELECT * FROM Player_Attributes;", conn)
team_attr = pd.read_sql("SELECT * FROM Team_Attributes;", conn)
match_data = pd.read_sql("SELECT * FROM Match;", conn)


rows = ["country_id", "league_id", "season", "stage", "date", "match_api_id", "home_team_api_id",
        "away_team_api_id", "home_team_goal", "away_team_goal", "home_player_1", "home_player_2",
        "home_player_3", "home_player_4", "home_player_5", "home_player_6", "home_player_7",
        "home_player_8", "home_player_9", "home_player_10", "home_player_11", "away_player_1",
        "away_player_2", "away_player_3", "away_player_4", "away_player_5", "away_player_6",
        "away_player_7", "away_player_8", "away_player_9", "away_player_10", "away_player_11"]
match_data.dropna(subset=rows, inplace=True)
match_data.drop(match_data.columns.difference(rows), 1, inplace=True)

# add features of goal ratio and win aspect
match_data = prep.goal_handler(match_data)

# Normalizing player attributes and
relevant_player_attrs = ["player_api_id", "date", "overall_rating", "potential", "crossing"]
# Weights for player and teams attributes
player_weights = [0.4, 0.4, 0.2]

prep_matches = prep.get_matches(player_attr, relevant_player_attrs, player_weights, match_data, team_attr)

train, test = train_test_split(prep_matches, test_size=0.2)

print(prep_matches.shape)
print(prep_matches.columns)
print(prep_matches)
