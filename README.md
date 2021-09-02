
# Can a Data Scientist Replace a NBA Scout? ML App Development for Best Transfer Suggestions

[NBA API ~ K-Means, Gaussian Mixture Models Clustering, Logistic Regression, Random Forest, XGBoost Classifiers | Originally published on this Medium post via the Towards Data Science publication]

![alt text](https://github.com/makispl/ml-nba-transfer-suggestion-app/blob/main/reports/figures/jc-gellidon-XmYSlYrupL8-unsplash.jpg?raw=true)

## Introduction

The project’s domain relies on the most popular American sport; **NBA**. The transaction window is by far the most interesting, high-pressure and expensive period of each season. Vast datasets are analysed, millions $ are spent and strategic movements are deployed, for each Team tries to elevate its performance the best possible...

## Scope

The main scope, hereof, is to present an end-to-end ML app development procedure, which embodies quite a number of Supervised and Unsupervised ML algorithms, including **Gaussian Mixtures Models (GMM)**, **K-Means**, **Principal Component Analysis (PCA)**, **XGBoost**, **Random Forest** & **Multinomial Logistic Regression Classifiers**. The mission is clear; predict the next best transaction a team has to make, for a specific position, to boost its performance.

## Scenario

The Data Corp I work for, accepted a new project: assist Milwaukee Bucks to make the best next move, during the 2020 transaction window. That is, to pre-access the candidate players for the Shooting Guard (SG) position (**Jrue Holiday**, **Danny Green**, **Bogdan Bogdanovic**) and buy the one who best performs. Being oblivious of Basketball knowledge, leads me to a tricky alternative:

*How about requesting the NBA API, fetching player data from the past seasons’ games (e.g. assist to turnovers, assist % and so on), categorising them in a meaningful way for the General Manager (GM) and finally guide him on whom they should spent the transfer budget on?*

## Roadmap

1. Build the **dataset**; fetch the player-wise statistics per game (from now on ‘plays’).
2. Perform **EDA;** build intuition on the variables’ exploitation, come to earliest conclusions.
3. **Cluster** ‘plays’ via **K-Means** & **GMM**; reveal underlying patterns and identify the most suitable cluster for the case.
4. Using the now labeled dataset (clusters = labels), train a number of Multi-class Classifiers, incl. **Multinomial****Logistic Regression**, **Random Forest** & **XGBoost**.
5. Make **Predictions** on the candidate players’ latest ‘plays’ (2020 season) and benchmark them accordingly.
6. **Serve** the trained models to the end-user, by building & serving an API (*future imlementation*).

## Guide

![alt text](https://github.com/makispl/ml-nba-transfer-suggestion-app/blob/main/reports/figures/workflow@2x.png?raw=true)

1. Download the `basketball.sqlite` from [Kaggle](https://www.kaggle.com/wyattowalsh/basketball) and store it in the ../data/external directory. It is 773.77 MB and exceeds GitHub's file size limit of 100.00 MB.

2. Set up the `config.py` to declare the necessary data/models directories/files.

3. Run `dataset.py` to fetch the desired seasons' plays - you are prompted to enter the seasons. (*Be mind that it is an extremely time-consuming process*)

4. Run `preprocess.py` by passing the argument 'clustering' in the option '--proc', i.e.:

   `python preprocess.py --proc clustering`

   This will preprocess data, splits it to train and test and prepare the former for the clustering procedure.

5. Run `clustering.py` to cluster the training dataset's plays.

6. Run `preprocess.py` by passing the argument 'classification' in the option '--proc', i.e.:

   `python preprocess.py --proc classification`

   This will preprocess both training and test data and prepare them for the classification models.

7. Run `create_folds.py` to create a CV=5 Stratified K-fold cross-validation.

8. Configure `model_discpatcher.py` with the models you want to train.

9. Run `train.py` to either:

   a. to train the declared model in the selected fold

   `python train.py --fold 0 --model log_res`

   b. to train the declared model in the whole training dataset

   `python train.py --model log_res`

10. Run predict.py either to:

   a. predict the clusters for the testing dataset

   `python predict.py`

   b. predict the clusters and suggest the best player - you are prompted to enter the player full names.

   `python predict.py --rank True`

## Findings

#1: We have to deeply study the most significant features for the case of SG (`group_1`), in a way that will not only **guarantee significant levels** for the respective features, but also won't **compromise** (the greatest possible) the rest.

Sorting the dataset by a single feature (e.g. `AST_PCT`), taking the upper segment (95th Percentile) and evaluating the plays 'horizontally' (across all features), proved wrong. By comparing the population with the 95th percentile average features, we see that by maximising along AST_PCT many of the rest features get worse, violating the above Assumption.

#2: We have to build better intuition on the available data and use more advanced techniques, to effectively segment it and capture the **underlying patterns**, which may lead us to the best SG's profile.

By applying K-Means / Gaussian MIxture Models Clustering Algortithms, we revealed a clearer indication of what it really takes to be a top-class SG.

#3: `Cluster_3` encapsulates those 'plays' which derive from great SG performance, in a really balanced way  - ` group_1` features reach high levels, while most of the rest keep a decent average.

This analysis, takes into account more features than the initially attempted (ref. EDA) which leveraged a dominant one (AST_PCT). Which proves the point that…

#4: Clustering promotes a more comprehensive separation of data, deriving from signals of more components and along these lines we managed to reveal a clearer indication of what performance to anticipate from a top-class SG.

## Screenshots

![alt text](https://github.com/makispl/ml-nba-transfer-suggestion-app/blob/main/reports/figures/gm_cluster@2x.png?raw=true)


## Results

We predicted that most of the latest (2020 season) plays of Jrue Holiday belong to `cluster_3`, noting a ratio of 86%.

```python
# Results
{
 'Jrue Holiday': 0.86,
 'Bogdan Bogdanovic': 0.38,
 'Danny Green': 0.06
}
```

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Jrue is a Buck.<a href="https://twitter.com/hashtag/FearTheDeer?src=hash&amp;ref_src=twsrc%5Etfw">#FearTheDeer</a> | <a href="https://twitter.com/Jrue_Holiday11?ref_src=twsrc%5Etfw">@Jrue_Holiday11</a> <a href="https://t.co/LSdsrnzHlM">pic.twitter.com/LSdsrnzHlM</a></p>&mdash; Milwaukee Bucks (@Bucks) <a href="https://twitter.com/Bucks/status/1331278772969091077?ref_src=twsrc%5Etfw">November 24, 2020</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



![alt text](https://github.com/makispl/ml-nba-transfer-suggestion-app/blob/main/reports/figures/tweet_jrue_holiday.png?raw=true)

And guess what? On November 24th of 2020, Bucks officially announced Jrue Holiday’s transaction!

## Additional Analysis

There is quite a number of additional analyses to be performed, expanding this one. You are welcome to extend and shape yours in any direction you may prefer. For instance, you can develop richer datasets by requesting extra endopoint of the nba_api. Additionally, you can further optimise the currently used ML models, create new features or even try to fit different models.

## Authors and Acknowledgement
[*It stands as an independent analysis in an effort to enhance my ability to communicate results, reason about data statistically and stay motivated to continuously implement newly aquired skills & capabilities, so as to enrich my portfolio of data science-oriented projects*]
- [@makispl](https://github.com/makispl) for concept & implementation.
- [@MPlegas](https://twitter.com/MPlegas) Twitter
- [@gerasimos_plegas](https://medium.com/@gerasimos_plegas) Medium
