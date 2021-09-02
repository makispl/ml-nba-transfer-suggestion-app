# import libraries
import pandas as pd
import numpy as np
from sklearn import model_selection

if __name__ == '__main__':

    # read training data
    df = pd.read_csv('../data/processed/train_proc_labeled.csv',
                     converters={'GAME_ID': lambda x: str(x)})

    # we create a new column called kfold and fill it with -1
    df['kfold'] = -1

    # randomize the rows of data
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch labels
    y = df.gm_cluster.values

    # initiate the kfold class from model_Selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    # save the new csv with kfold column
    df.to_csv(
        '../data/processed/train_proc_labeled_folds.csv', index=False)
