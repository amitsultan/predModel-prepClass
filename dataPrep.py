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


def normalize_player_atters(player_data, relevant_player_attrs, player_weights):
    sc = StandardScaler()
    overall = []
    relevant_player_attrs.remove('player_api_id')
    relevant_player_attrs.remove('date')
    player_data = player_data.sort_values('date').groupby('player_api_id').tail(1).drop('date', axis=1)
    scaled = sc.fit_transform(player_data.iloc[:, 1:])
    for row in scaled:
        sum = 0
        for i in range (0, len(row)):
            sum += player_weights[i]*row[i]
        overall.append(sum / len(row))
    player_data.drop(relevant_player_attrs, 1, inplace=True)
    player_data['score'] = overall
    return player_data


# creating a normalized score for each player to analyze his affect on the winning /loosing of the team
def players_processing(matches, player_data):
    # normalize score for home
    start = timeit.default_timer()
    for index, match in matches.iterrows():
        # home
        for player_index in range(1, 12):
            i = 'home_player_' + str(player_index)
            match[i] = player_data.loc[player_data.player_api_id == match[i]].loc[:, 'score'].values[0]
            j = 'away_player_' + str(player_index)
            match[j] = player_data.loc[player_data.player_api_id == match[j]].loc[:, 'score'].values[0]
    stop = timeit.default_timer()
    print('Time: ', stop - start)
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


def prepare_players_data_in_match(player_attr, relevant_player_attrs, player_weights, match_data):
    player_attr.dropna(subset=relevant_player_attrs, inplace=True)
    player_attr.drop(player_attr.columns.difference(relevant_player_attrs), 1, inplace=True)
    player_attr = normalize_player_atters(player_attr, relevant_player_attrs, player_weights)
    match_data = players_processing(match_data, player_attr)
    return match_data


def clean_team_data(teams, team_weights):
    sc = StandardScaler()
    teams.drop('team_fifa_api_id', axis=1, inplace=True)
    teams.drop('id', axis=1, inplace=True)
    mean_values = teams.groupby('buildUpPlayDribblingClass').mean().loc[:, 'buildUpPlayDribbling'].values
    teams['buildUpPlayDribbling'] = teams.apply(
        lambda row: fill_team_na(row, mean_values) if np.isnan(row['buildUpPlayDribbling']) else row['buildUpPlayDribbling'], axis=1)
    for column in teams:
        if column != 'date' and not np.issubdtype(teams[column].dtype, np.number):
            teams.drop(column, axis=1, inplace=True)
    teams = teams.sort_values('date').groupby('team_api_id').tail(1).drop('date', axis=1)
    scaled = sc.fit_transform(teams.iloc[:, 1:])
    team_scores = team_processing(scaled, team_weights)
    # scaled_features_df = pd.DataFrame(np.column_stack([teams['team_api_id'], team_scores]), columns={"team_api_id", "scores"})
    scaled_features_df = pd.DataFrame({"team_api_id": teams['team_api_id'] , "scores" :team_scores})
    # scaled_features_df = scaled_features_df.astype({'team_api_id': 'int64'})
    return scaled_features_df


def team_processing(scaled_teams, team_weights):
    overall = []
    for row in scaled_teams:
        sum = 0
        for i in range (0, len(row)):
            sum += team_weights[i]*row[i]
        overall.append(sum / len(row))
    return overall

def assign(row, overall):
    row['home_team_score'] = overall[row['home_team_score']]
    row['away_team_score'] = overall[row['away_team_score']]

def insert_team_scores(matches, overall):
    #renaming columns of teams
    matches = matches.rename(columns={"home_team_api_id": "home_team_score", "away_team_api_id": "away_team_score"})
    # normalize score for home
    start = timeit.default_timer()
    x = pd.Series(overall.scores.values, index=overall.team_api_id).to_dict()
    print(len(x))
    print(matches.shape)
    matches = matches[matches.home_team_score.isin(x)]
    matches = matches[matches.away_team_score.isin(x)]
    matches.apply(lambda row: assign(row, x), axis=1)
    print(matches.shape)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    return matches


def prepare_teams_data_in_match(matches, teams, team_weights):
    teams_scores = clean_team_data(teams, team_weights)
    matches = insert_team_scores(matches, teams_scores)

