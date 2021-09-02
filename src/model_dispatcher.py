from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
import xgboost as xgb

models = {
    "log_res": LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        C=10,
        max_iter=1500,
        n_jobs=-1
    ),
    "xgb": xgb.XGBClassifier(
        colsample_bytree=1.0,
        reg_lambda=0.1,
        max_depth=3,
        n_estimators=200,
        n_jobs=-1
    ),
    "rf": ensemble.RandomForestClassifier(
        criterion='gini',
        max_depth=15,
        n_estimators=500,
        n_jobs=-1),
}
