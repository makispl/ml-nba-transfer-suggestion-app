# import the libraries
import argparse
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import config

# read the dataset


def read_raw_data(input_file):
    """
    Takes in the raw data file and
    returns a pandas dataframe of the original
    Parameters
    ---------
    input_file : a str object
             Contains the raw file's path
    Returns
    -------
    df : a dataframe object
             Contains the dataframe of the input file
    """
    df = pd.read_csv(input_file, converters={'GAME_ID': lambda x: str(x)})
    return df


# translate the time str to integer seconds
def get_sec(time_str):
    """
    Reads a string of time
    returns the seconds of it in numerical form
    Parameters
    ---------
    time_str : a str object
             Contains the time in MIN
    Returns
    -------
        : int
             Contains the time in seconds
    """
    m, s = time_str.split(":")
    return int(m) * 60 + int(s)


def calculate_net_score(df, features):
    """
    Reads in the training dataset and features,
    calculates net score according to the equation:
    `NET_SCORE` = `0.5` * `group_1` + `0.3` * `group_2` + `0.2` * `group_3` + `0.0` * `group_4` + `-0.3` * `group_5`,
    returns the samme dataset with NET_SCORE column
    Parameters
    ---------
    df : a dataframe object
             Contains the training dataset
    features : a list object
             Contains the features
    Returns
    -------
    df : a dataframe object
             Contains the training dataset with 'NET_SCORE'
    """

    # classify features by domain importance
    group_1 = ['OFF_RATING', 'AST_PCT', 'AST_TOV',
               'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT', 'POSS']
    group_2 = ['MIN', 'AST_RATIO', 'DREB_PCT']
    group_3 = ['OREB_PCT', 'REB_PCT', 'USG_PCT', 'PACE', 'PACE_PER40', 'PIE']
    group_4 = ['START_POSITION']
    group_5 = ['DEF_RATING']

    # allocate weights
    wts = []
    for col in features:
        if col in group_1:
            wts.append(0.5)
        elif col in group_2:
            wts.append(0.3)
        elif col in group_3:
            wts.append(0.2)
        elif col in group_4:
            wts.append(0)
        else:
            wts.append(-0.3)

    # make a dictionary of key:value feature: weight
    weights_dict = dict(zip(features, wts))

    # segment the eligible for score original dataset
    scores_df = df.loc[:, features].copy()

    # the calculations to be executed
    df['NET_SCORE'] = scores_df.dot(pd.Series(weights_dict))

    return df


def preproc_basic(df):
    """
    Reads in the original dataset, executes basic pre-processing,
    splits it to train and test
    procedures and return 
    calculates net score according to the equation:
    `NET_SCORE` = `0.5` * `group_1` + `0.3` * `group_2` + `0.2` * `group_3` + `0.0` * `group_4` + `-0.3` * `group_5`,
    returns the samme dataset with NET_SCORE column
    Parameters
    ---------
    df : a dataframe object
             Contains the training dataset
    features : a list object
             Contains the features
    Returns
    -------
    df : a dataframe object
             Contains the training dataset with 'NET_SCORE'
    """

    print('Preprocessing raw data')
    # drop redundant columns
    df.drop(
        columns=[
            "COMMENT",
            "E_OFF_RATING",
            "E_DEF_RATING",
            "E_NET_RATING",
            "E_USG_PCT",
            "NET_RATING",
            "E_PACE"
        ],
        inplace=True,
    )

    # filter-out no-plays (plays w/ 'MIN' = 0)
    df.dropna(subset=["MIN"], inplace=True)

    # translate 'MIN' col toy minutes
    df.loc[:, "MIN"] = df.loc[:, "MIN"].map(lambda x: get_sec(x))

    # encode 'START_POSITION' col to numerical
    df.loc[df.START_POSITION.isna(), "START_POSITION"] = "NaN"
    mapping = {
        "NaN": 0,
        "G": 1,
        "F": 2,
        "C": 3
    }
    df.loc[:, "START_POSITION"] = df.START_POSITION.map(mapping)
    df.loc[:, "START_POSITION"] = df.loc[:, "START_POSITION"].astype(int)

    # check for nulls
    try:
        assert df.loc[:, df.isnull().sum() > 0].shape[1] == 0
    except:
        print("There are Nulls in the dataset")

    # check for duplicate rows
    try:
        assert df.duplicated().sum() == 0
    except:
        print("There are duplicate raws in the dataset")
        # drop duplicates
        df.drop_duplicates(inplace=True)
        print("Duplicates eliminated from the dataset")

    # define features
    features = [
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
            "GAME_DATE"
        )
    ]

    # calculate 'NET_SCORE' - not a feat!
    df = calculate_net_score(df, features)

    # split train and test datasets
    # train dataset --> 2017-10-17 until 2019-12-31
    # test dataset --> 2020-01-01 until 2020-10-11 (end of play-offs)

    train_df = df.loc[
        (df.GAME_DATE <= '2019-12-31'), :].copy()

    test_df = df.loc[(
        df.GAME_DATE >= '2020-01-01')
        &
        (df.GAME_DATE < '2020-10-11'), :].copy()

    # export only the test dataset - no processing for clustering purposes
    print('Exporting the test dataset')

    test_df.to_csv(
        os.path.join(config.INTERIM_DATA_OUTPUT, f"test.csv"),
        index=False
    )

    return train_df


def preproc_clustering(df):

    print('Preprocessing train data for clustering')

    # define features
    features = [
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
            "GAME_DATE"
        )
    ]

    # define data to be processed
    data = df[features].copy()

    # normalization
    scaler = MinMaxScaler()
    data_stnd = scaler.fit_transform(data)

    # PCA
    pca = PCA(n_components=4)
    pca.fit(data_stnd)
    pca_scores = pca.transform(data_stnd)

    df_pca = pd.concat([df.reset_index(drop=True), pd.DataFrame(
        data=pca_scores, columns=['pca_1', 'pca_2', 'pca_3', 'pca_4'])], axis=1)

    # export the datasets
    print('Exporting the Preprocessed training dataset (ready for clustering)')
    df_pca.to_csv(
        os.path.join(config.INTERIM_DATA_OUTPUT, f"train_preproc.csv"),
        index=False
    )

    return df_pca


def preproc_classifier():

    # read the training data
    train_df = pd.read_csv('../data/interim/train_preproc_labeled.csv',
                           converters={'GAME_ID': lambda x: str(x)}
                           )

    # read the testing data
    test_df = pd.read_csv('../data/interim/test.csv',
                          converters={'GAME_ID': lambda x: str(x)}
                          )

    #### TRAINING DAATASET ####

    print('Preprocessing train data for classification')

    # drop redundant columns - be careful pca components belongs to
    # the previous clustering process
    train_df.drop(columns=['pca_1', 'pca_2', 'pca_3', 'pca_4'], inplace=True)

    # define original features
    train_feats = [
        col
        for col in train_df.columns
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

    # define training data to be normalized
    train_data = train_df[train_feats].values

    # initialize scaler
    scaler = MinMaxScaler()

    # fit scaler
    scaler.fit(train_data)

    # transform training data
    train_data_stnd = scaler.transform(train_data)

    # define normalized training features
    train_norm_feats = [
        feat+'_n' for feat in train_feats
    ]

    # complete training dataset
    train_norm_df = pd.concat([train_df.reset_index(drop=True), pd.DataFrame(
        data=train_data_stnd, columns=train_norm_feats)], axis=1)

    # define training normalized data for PCA
    train_data_pca = train_norm_df.loc[:, train_norm_feats].values

    # instantiate PCA - choose 9 commponents to retain almost all the variance level
    pca = PCA(n_components=9)
    pca.fit(train_data_pca)

    # transform training data
    train_pca_scores = pca.transform(train_data_pca)

    # complete training dataset
    train_norm_pca_df = pd.concat([train_norm_df.reset_index(drop=True), pd.DataFrame(
        data=train_pca_scores, columns=['pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pca_6', 'pca_7', 'pca_8', 'pca_9'])], axis=1)

    # define pca features
    train_pca_feats = ["pca_"+f"{i}" for i in range(1, 10)]

    # export the dataset
    print('Exporting the Processed training dataset (ready for classification)')
    train_norm_pca_df.to_csv(
        os.path.join(config.PROCESSED_DATA_OUTPUT, f"train_proc_labeled.csv"),
        index=False
    )

    #### TESTING DATASET ####

    print('Preprocessing testing data for classification')

    # drop redundant columns
    test_df.drop(
        columns=[
            "START_POSITION",
            "MIN"
        ],
        inplace=True,
    )

    # define original features
    test_feats = [
        col
        for col in test_df.columns
        if col
        not in (
            "GAME_ID",
            "TEAM_ID",
            "TEAM_ABBREVIATION",
            "TEAM_CITY",
            "PLAYER_ID",
            "PLAYER_NAME",
            "NICKNAME",
            "NET_SCORE",
            "GAME_DATE"
        )
    ]

    # define testing data to be normalized
    test_data = test_df[test_feats].values

    # transform testing data
    test_data_stnd = scaler.transform(test_data)

    # define normalized testing features
    test_norm_feats = [
        feat+'_n' for feat in test_feats
    ]

    # complete testing dataset
    test_norm_df = pd.concat([test_df.reset_index(drop=True), pd.DataFrame(
        data=test_data_stnd, columns=test_norm_feats)], axis=1)

    # define testing normalized data for PCA
    test_data_pca = test_norm_df.loc[:, test_norm_feats].values

    # transform test data
    test_pca_scores = pca.transform(test_data_pca)

    test_norm_pca_df = pd.concat([test_norm_df.reset_index(drop=True), pd.DataFrame(
        data=test_pca_scores, columns=['pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pca_6', 'pca_7', 'pca_8', 'pca_9'])], axis=1)

    # define testing pca features
    test_pca_feats = ["pca_"+f"{i}" for i in range(1, 10)]

    # export the dataset
    print('Exporting the Processed testing dataset')
    test_norm_pca_df.to_csv(
        os.path.join(config.PROCESSED_DATA_OUTPUT, f"test_proc.csv"),
        index=False
    )

    return train_norm_pca_df, test_norm_pca_df


def main(procedure):

    if procedure == 'clustering':
        train_df = preproc_basic(df)
        preproc_clustering(train_df)

    elif procedure == 'classification':
        preproc_classifier()

    else:
        print("Wrong argument - choose either 'clustering' or 'classification'.")

    return None


if __name__ == '__main__':
    input_file = config.RAW_DATA_FILE
    df = read_raw_data(input_file)

    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add the argument we need and its type
    # we only need procedure
    parser.add_argument(
        "--proc",
        type=str
    )

    # read the arguments from the command line
    args = parser.parse_args()

    # run the main specified by command line arguments
    main(
        procedure=args.proc,
    )
