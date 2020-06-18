import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import sqlite3


def prepare_and_save_data(path):
    preparing_data(path, "SELECT * FROM Match WHERE season LIKE '2015/2016';", "prep_predict.csv")
    preparing_data(path, "SELECT * FROM Match WHERE season NOT LIKE '2015/2016';", "prep_matches.csv")


def preparing_data(path, query, name):
    conn = sqlite3.connect(path)

    player_attr = pd.read_sql("SELECT * FROM Player_Attributes;", conn)
    team_attr = pd.read_sql("SELECT * FROM Team_Attributes;", conn)
    match_data = pd.read_sql(query, conn)

    rows = ["country_id", "league_id", "season", "stage", "date", "match_api_id", "home_team_api_id",
            "away_team_api_id", "home_team_goal", "away_team_goal", "home_player_1", "home_player_2",
            "home_player_3", "home_player_4", "home_player_5", "home_player_6", "home_player_7",
            "home_player_8", "home_player_9", "home_player_10", "home_player_11", "away_player_1",
            "away_player_2", "away_player_3", "away_player_4", "away_player_5", "away_player_6",
            "away_player_7", "away_player_8", "away_player_9", "away_player_10", "away_player_11"]
    match_data.dropna(subset=rows, inplace=True)
    match_data.drop(match_data.columns.difference(rows), 1, inplace=True)

    # add features of goal ratio and win aspect
    match_data = goal_handler(match_data)

    # getting all data from teams and players together
    prep_matches = get_matches(player_attr, match_data, team_attr)
    del prep_matches['stage']
    del prep_matches['goal_ratio']

    # save data to disk
    prep_matches.to_csv(name, encoding='utf-8', index=False)


# Add goal ratio and end game status
def goal_handler(matches):
    ratio_vector = []
    status_vector = []
    i = 0
    for index, row in matches.iterrows():
        if row['away_team_goal'] != 0:
            ratio_vector.append(row['home_team_goal'] / row['away_team_goal'])
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
    matches.drop(['home_team_goal', 'away_team_goal'], axis=1, inplace=True)
    return matches


# preparing data of players by choosing 3 most important features with PCA and adding it to matches
def prepare_players_data_in_match(player_data, matches):
    player_data.dropna(inplace=True)
    player_data = player_data.sort_values('date').groupby('player_api_id').tail(1).drop('date', axis=1)
    for column in player_data:
        if column != 'date' and not np.issubdtype(player_data[column].dtype, np.number):
            player_data.drop(column, axis=1, inplace=True)
    ids = player_data['player_api_id'].values
    player_data.drop(player_data.columns[0:3], axis=1, inplace=True)
    pca = PCA(n_components=2, svd_solver='full')
    principalComponents = pca.fit_transform(player_data)
    principalDf = pd.DataFrame(data=principalComponents)
    principalDf['player_api_id'] = ids
    for i in range(1, 12):
        principalDf = principalDf.rename(
            columns={0: "home_player_" + str(i) + "_f1", 1: "home_player_" + str(i) + "_f2"})
        matches = matches.set_index('home_player_' + str(i)).join(principalDf.set_index('player_api_id'))
        principalDf = principalDf.rename(
            columns={"home_player_" + str(i) + "_f1": 0, "home_player_" + str(i) + "_f2": 1})
    for i in range(1, 12):
        principalDf = principalDf.rename(
            columns={0: "away_player_" + str(i) + "_f1", 1: "away_player_" + str(i) + "_f2"})
        matches = matches.set_index('away_player_' + str(i)).join(principalDf.set_index('player_api_id'))
        principalDf = principalDf.rename(
            columns={"away_player_" + str(i) + "_f1": 0, "away_player_" + str(i) + "_f2": 1})
    matches.dropna(inplace=True)
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


def prepare_teams_data_in_match(matches, teams):
    teams.drop('team_fifa_api_id', axis=1, inplace=True)
    teams.drop('id', axis=1, inplace=True)
    mean_values = teams.groupby('buildUpPlayDribblingClass').mean().loc[:, 'buildUpPlayDribbling'].values
    teams['buildUpPlayDribbling'] = teams.apply(
        lambda row: fill_team_na(row, mean_values) if np.isnan(row['buildUpPlayDribbling']) else row[
            'buildUpPlayDribbling'], axis=1)
    for column in teams:
        if column != 'date' and not np.issubdtype(teams[column].dtype, np.number):
            teams.drop(column, axis=1, inplace=True)
    teams = teams.sort_values('date').groupby('team_api_id').tail(1).drop('date', axis=1)
    ids = teams['team_api_id'].values
    teams.drop('team_api_id', axis=1, inplace=True)
    pca = PCA(n_components=3, svd_solver='full')
    principalComponents = pca.fit_transform(teams)
    principalDf = pd.DataFrame(data=principalComponents)
    principalDf['team_api_id'] = ids
    principalDf = principalDf.rename(
        columns={0: "home_f_1", 1: "home_f_2", 2: "home_f_3"})
    matches = matches.set_index('home_team_api_id').join(principalDf.set_index('team_api_id'))
    principalDf = principalDf.rename(
        columns={"home_f_1": "away_f_1", "home_f_2": "away_f_2", "home_f_3": "away_f_3"})
    matches = matches.set_index('away_team_api_id').join(principalDf.set_index('team_api_id'))
    matches.dropna(inplace=True)
    return matches


def get_matches(player_attr, match_data, teams):
    match_data = prepare_players_data_in_match(player_attr, match_data)
    match_data = prepare_teams_data_in_match(match_data, teams)
    match_data.drop(['country_id', 'league_id', 'season', 'date', 'match_api_id'], axis=1, inplace=True)
    return match_data
