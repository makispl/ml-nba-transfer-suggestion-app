import argparse
import os
import joblib
import pandas as pd
from sklearn.metrics import f1_score
import model_dispatcher
import config


def run(df, fold, model):
    """
    Takes in the training dataset, the # of folds and
    the model,
    prints the per fold score and saves the model
    Parameters
    ---------
    df : a dataframe object
             The training dataset
    fold: int
             The # fold
    model: str
             The desired model
    Returns
    -------
    -
    """

    # training data is where kfold is not equal to provided fold
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # convert data to a numpy array by using .values.
    x_train = df_train[features].values
    y_train = df_train.gm_cluster.values

    # similarly, for validation, we have
    x_valid = df_valid[features].values
    y_valid = df_valid.gm_cluster.values

    # fetch the model from model_dispatcher
    clf = model_dispatcher.models[model]

    # fit the model on training data
    clf.fit(x_train, y_train)

    # create predictions for validation samples
    valid_preds = clf.predict(x_valid)

    # calculate & print f1 score
    f1 = f1_score(
        df_valid.gm_cluster.values,
        valid_preds,
        average='weighted'
    )

    print(f"Fold={fold}, F1={f1}")

    # save the model
    joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
    )


def total_run(df, model):
    """
    Takes in the training dataset and the model,
    saves the model
    Parameters
    ---------
    df : a dataframe object
             The training dataset
    model: str
             The desired model
    Returns
    -------
    -
    """

    # convert data to a numpy array by using .values.
    x_train = df[features].values
    y_train = df.gm_cluster.values

    # fetch the model from model_dispatcher
    clf = model_dispatcher.models[model]

    # fit the model on training data
    clf.fit(x_train, y_train)

    # save the model
    joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTPUT,
                     f"{args.model}.bin")
    )


if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add the different arguments you need and their type
    # currently, we only need fold
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )
    # read the arguments from the command line
    args = parser.parse_args()

    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE, converters={
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

    if args.fold == None:
        # run all the folds - export trained model
        total_run(
            df,
            model=args.model
        )
    else:
        # run the fold specified by command line arguments
        run(
            df,
            fold=args.fold,
            model=args.model
        )
