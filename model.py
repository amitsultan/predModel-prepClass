import sqlite3
import pandas as pd
import dataPrep as prep
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn


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
teams_weights = [1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9]
# match_data = prep.prepare_players_data_in_match(player_attr, relevant_player_attrs, player_weights, match_data)

# print(match_data)
# clean team data and add missing values based on mean
prep.prepare_teams_data_in_match(match_data, team_attr, teams_weights)
