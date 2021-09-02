# Import the libraries
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
import config
import joblib
import os

# read in the preprocessed data
df = pd.read_csv('../data/interim/train_preproc.csv',
                 converters={'GAME_ID': lambda x: str(x)})

# define pca components
pca_feats = ["pca_"+f"{i}" for i in range(1, 5)]

# define pca scores
pca_scores = df.loc[:, pca_feats].copy()

# Instantiate a GM model with 4 clusters, fit and predict cluster indices
gm = GaussianMixture(n_components=4, init_params='kmeans', tol=1e-4,
                     covariance_type='full', n_init=10, random_state=1)
df['gm_cluster'] = gm.fit_predict(pca_scores)

# export the clustered dataset
print('Exporting the clustered training dataset')
df.to_csv(
    os.path.join(config.INTERIM_DATA_OUTPUT, f"train_preproc_labeled.csv"),
    index=False
)
