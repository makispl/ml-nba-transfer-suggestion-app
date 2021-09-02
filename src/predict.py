# import libraries
import argparse
import os
import joblib
import pandas as pd
import config


def make_predictions():
    """
    Returns a dataframe with the predicted clusters.
    Parameters
    ---------
    -
    Returns
    -------
    preds_df : a dataframe object
             Contains the per game 'pred_cluster'
    """

    # read the testing dataset
    df = pd.read_csv(config.TESTING_FILE, converters={
        'GAME_ID': lambda x: str(x)})

    # define pca features
    pca_feats = ["pca_"+f"{i}" for i in range(1, 10)]

    # define original features
    feats = [
        col
        for col in df.columns
        if col
        not in (
            "GAME_ID",
            "TEAM_ID",
            "TEAM_ABBREVIATION",
            "TEAM_CITY",
            "PLAYER_ID",
            "PLAYER_NAME",
            "NICKNAME",
            "START_POSITION",
            "MIN",
            "gm_cluster",
            "NET_SCORE",
            "kfold",
            "GAME_DATE"
        )
    ]

    # define normalized features
    norm_feats = [
        feat+'_n' for feat in feats
    ]

    # define the selected features
    # opt for the pca feats
    features = pca_feats

    # switch to the prediction data from 2020-01-01 to 2020-08-31
    preds_df = df.loc[(df.GAME_DATE < '2020-11-01'), :].copy()

    X_test = preds_df.loc[:, pca_feats].values

    # load the model
    clf = joblib.load(config.MODEL_IN_USE)

    # Make prediction via the logres model, using the 9 pca_feats
    y_pred = clf.predict(X_test)

    # complete the testing dataset
    preds_df.loc[:, 'pred_cluster'] = y_pred

    # save the new csv with kfold column
    preds_df.to_csv(
        '../data/processed/test_proc_labeled.csv', index=False)

    return preds_df


def select_player(preds_df):
    """
    Takes in the preds_df, prompts the user to enter
    the desired players full names,
    returns a dictionary with each players % of cluster_3 plays
    Parameters
    ---------
    preds_df : a dataframe object
             Contains the per game 'pred_cluster'
    Returns
    -------
    ranking_sorted : a dictionary object
             Contains the cluster_3 % of plays per player
    """

    # set candidates
    candidates = [item for item in input(
        "enter the candidate players' full names separated by comma, like:\nGerasimos Plegas, GitHub Reader : ").split(',')]

    # define the dataset's players
    names = pd.Series(preds_df.PLAYER_NAME.unique()).tolist()

    # assert the final candidates names
    final_candidates = []
    for candit in candidates:
        try:
            assert candit in names
            final_candidates.append(candit)
        except:
            print(f"The name: {candit} is not registered.")

    ranking = {}

    # check for their mebmership in cluster_3 and the ratio
    for candit in final_candidates:
        candit_df = preds_df.loc[preds_df.PLAYER_NAME == candit, :].copy()
        vals = candit_df.loc[:, 'pred_cluster'].value_counts()
        rank = vals.loc[2] / vals.sum()
        ranking[candit] = round(rank, 2)

    # sort the players by their ranking
    ranking_sorted = {k: v for k, v in sorted(
        ranking.items(), reverse=True, key=lambda item: item[1])}

    print(ranking_sorted)


if __name__ == "__main__":

    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add the different arguments you need and their type
    # currently, we only need fold
    parser.add_argument(
        "--rank",
        type=bool
    )

    # read the arguments from the command line
    args = parser.parse_args()

    preds_df = make_predictions()

    if args.rank == True:
        # run player's ranking
        select_player(preds_df)
