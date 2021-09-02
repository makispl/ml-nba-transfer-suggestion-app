# import the libraries
from nba_api.stats.endpoints import boxscoreadvancedv2
import sqlite3 as sql
import pandas as pd
import numpy as np
from multiprocessing import Pool
import requests
from functools import partial
from io import BytesIO
import os
from datetime import datetime
import config


# get proxies
def get_proxies(nba_api=True):
    res = requests.get(
        'https://api.proxyscrape.com/v2/?request=getproxies&protocol=http&timeout=3000&country=all&ssl=yes&anonymity=all&simplified=true')
    content = res.content
    res.close()
    proxy_list = [item[0]
                  for item in pd.read_csv(BytesIO(content)).values.tolist()]
    proxy_list = proxy_list + \
        requests.get(
            'http://pubproxy.com/api/proxy?limit=25&format=txt&http=true&country=US&type=http').content.decode("utf-8").split('\n')
    proxy_list = proxy_list + ['165.225.77.42:8800', '165.225.77.44:80', '3.10.5.102:8080', '165.225.77.44:9400', '165.225.77.42:9443', '165.225.77.42:443',
                               '165.225.77.42:80', '165.225.77.47:443', '165.225.77.47:8800', '165.225.77.47:9443', '165.225.77.47:80', '165.225.77.47:9400',
                               '165.225.77.47:9401']

    proxies = proxy_list
    if not nba_api:
        proxy_list = ["http://" + proxy for proxy in proxies]
        proxies = {"http": proxy_list, "https": proxy_list}
    return proxies


# get nba api endoints
def get_nba_api_endpoint(input_val, input_val_label, endpoint, process_response_dfs, proxies, game_ids):
    # define helpful variables
    if np.where(game_ids == input_val)[0].size > 0:
        print("Making request #{} for game id: {} | {}".format(
            np.where(game_ids == input_val)[0], input_val, datetime.now()))
    no_res = True
    proxy_collection_counter = 0
    proxy_index = 0
    arg = {input_val_label: input_val}
    # while no response
    while no_res:
        # try getting a response without a proxy
        try:
            dfs = endpoint(**arg, proxy="http://" + proxies,
                           timeout=3).get_data_frames()
            no_res = False
            break
        except:
            # if that fails
            while no_res:
                # try getting with a certain proxy
                try:
                    dfs = endpoint(
                        **arg, proxy="http://" + proxies[proxy_index], timeout=3).get_data_frames()
                    no_res = False
                    break
                except:
                    # if that fails, move on to next proxy unless out of proxies
                    if (proxy_index + 1) >= len(proxies):
                        # unless tried proxies 5 times
                        if proxy_collection_counter < 5:
                            # if out of proxies: get more proxies, fix counters, and try without a proxy again
                            proxy_index = 0
                            proxy_collection_counter = proxy_collection_counter + 1
                            break
                        else:
                            return None
                    else:
                        proxy_index = proxy_index + 1
    res = process_response_dfs(dfs)
    return res


def process_response_dfs_box_advan(res):
    """

    """
    return {'players': res[0], 'teams': res[1]}


# get players & teams dataframes
def get_df(game_ids):
    """
    Takes in a dataframe of the 'GAME_ID',
    returns 2 dataframes of player & team data
    Parameters
    ---------
    game_ids : a dataframe object
             Contains the ID of each game 
    Returns
    -------
    player_df : a dataframe object
             Contains the player attributes per game
    team_df : a dataframe object
             Contains the team attributes per game
    """
    proxies = get_proxies()

    dfs = map(partial(get_nba_api_endpoint, input_val_label='game_id', endpoint=boxscoreadvancedv2.BoxScoreAdvancedV2,
                      process_response_dfs=process_response_dfs_box_advan, proxies=proxies, game_ids=game_ids), game_ids)
    print("successfully extracted all game ids ... concatenating and returning | {}".format(
        datetime.now()))

    player_df = pd.DataFrame()
    teams_df = pd.DataFrame()
    for df in dfs:
        player_df = pd.concat([player_df, df['players']], axis=0)
        teams_df = pd.concat([teams_df, df['teams']], axis=0)
    return player_df, teams_df


# get seasons' game winners
def get_game_winners(db, seasons):
    """
    Takes in the db and the seasons of interest,
    returns a dataframe game winner teams
    Parameters
    ---------
    db : a str object
             Contains the db's path
    season: a list object
             Contains the seasons years (2019-->22019, 2020-->22020, etc)
    Returns
    -------
    game_winners_df : a dataframe object
             Contains the per game winners 'TEAM_ID'
    """
    q = str()
    for i in range(len(seasons)):
        if i < (len(seasons) - 1):
            season = "'" + str(seasons[i]) + "'"
            q = q + f"(SEASON_ID={season}) OR "
        else:
            season = "'" + str(seasons[-1]) + "'"
            q = q + f"(SEASON_ID={season})"

    # fetch the game data
    with sql.connect(db) as conn:
        game_info = pd.read_sql(
            f"SELECT * FROM Game WHERE {q}",
            conn
        )
        game_info = game_info.loc[:, [
            'GAME_ID', 'GAME_DATE', 'TEAM_ID_HOME', 'TEAM_ID_AWAY', 'WL_HOME', 'SEASON_ID']]

    # find the game winner teams
    game_info['WINNER_ID'] = game_info['TEAM_ID_HOME']
    game_info.loc[(game_info.loc[:, 'WL_HOME'] == 'L'),
                  'WINNER_ID'] = game_info.loc[:, 'TEAM_ID_AWAY']
    game_winners_df = game_info.loc[:, [
        'GAME_ID', 'WINNER_ID', 'GAME_DATE']].copy()

    # drop duplicate values
    game_winners_df.drop_duplicates(inplace=True)

    return game_winners_df


# get players dataframes
def get_plays(player_df, game_winners_df, seasons):
    """
    Takes in the player_df and game_winners_df dataframes,
    returns the merged dataframe
    Parameters
    ---------
    player_df : a dataframe object
             Contains the player attributes per game
    game_winners_df: a dataframe object
             contains the per game winners 'TEAM_ID'
    season: a list object
             Contains the seasons years (2019-->22019, 2020-->22020, etc)
    Returns
    -------
    plays_df : a dataframe object
             Contains the in-use dataframe of per game player attributes
    """
    plays_df = pd.merge(
        left=player_df,
        right=game_winners_df,
        how='left',
        on='GAME_ID'
    )

    cols = [
        'GAME_ID', 'GAME_DATE', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_CITY',
        'PLAYER_ID', 'PLAYER_NAME', 'NICKNAME', 'START_POSITION', 'COMMENT', 'MIN',
        'E_OFF_RATING', 'OFF_RATING', 'E_DEF_RATING', 'DEF_RATING',
        'E_NET_RATING', 'NET_RATING', 'AST_PCT', 'AST_TOV', 'AST_RATIO',
        'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT',
        'USG_PCT', 'E_USG_PCT', 'E_PACE', 'PACE', 'PACE_PER40', 'POSS', 'PIE'
    ]

    plays_df = plays_df[cols].copy()

    plays_df.to_csv(data_dir + 'plays_total.csv', index=False)

    return plays_df


if __name__ == '__main__':

    print(
        "Please enter the desired season years (2019-->22019, 2020-->22020, etc) separated by comma, like:\n22019,22020"
    )

    # set seasons
    seasons = input().split(',')

    # set the db
    db = config.EXTERNAL_DATA_FILE

    # set the data directory
    data_dir = config.RAW_DATA_OUTPUT

    # extract the game winner teams
    game_winners_df = get_game_winners(db, seasons)

    # create the player and team datasets
    player_df, teams_df = get_df(game_winners_df.GAME_ID)

    # create and export the final dataset
    get_plays(player_df, game_winners_df, seasons)
