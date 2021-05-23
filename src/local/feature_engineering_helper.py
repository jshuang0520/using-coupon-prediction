# -*- coding: utf-8 -*-
from xgboost import plot_importance
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
# from pandas.api.types import is_integer_dtype
import seaborn as sns
import sys
from sklearn.feature_selection import mutual_info_regression
import warnings
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))
# from src.local.etl_helper import Extract as local_extractor
from src.utility.utils import Logger


# ignore warnings
warnings.filterwarnings('ignore')
# Set plotting defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)


class MutualInformation:
    """
    X = df.copy()
    y = X.pop('your_target_col')
    """
    def __init__(self):
        self.logger = Logger().get_logger('feature engineering')

    @staticmethod
    def make_mi_scores(df_x, y):
        df_x = df_x.copy()
        for col in df_x.select_dtypes(["object", "category"]):
            df_x[col], _ = df_x[col].factorize()
        # All discrete features should now have integer dtypes
        discrete_features = [pd.api.types.is_integer_dtype(t) for t in df_x.dtypes]
        mi_scores = mutual_info_regression(df_x, y, discrete_features=discrete_features, random_state=0)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=df_x.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores

    @staticmethod
    def plot_mi_scores(scores):
        scores = scores.sort_values(ascending=True)
        width = np.arange(len(scores))
        ticks = list(scores.index)
        color = np.array(["C0"] * scores.shape[0])
        # Color red for probes
        idx = [i for i, col in enumerate(scores.index)
               if col.startswith("PROBE")]
        color[idx] = "C3"
        # Create plot
        plt.barh(width, scores, color=color)
        plt.yticks(width, ticks)
        plt.title("Mutual Information Scores")


class CorrCoef:
    def __init__(self):
        self.logger = Logger().get_logger('feature engineering')

    @staticmethod
    def plot(df, selected_cols=None):
        if not selected_cols:
            # selected_cols = list(df.select_dtypes(include=["int"]).columns)
            selected_cols = list(df.columns)
        assert all(pd.api.types.is_integer_dtype(df[col]) for col in selected_cols), 'not all columns are int type'

        # fig, ax = plt.subplots(figsize=(20, 20))
        plt.subplots(figsize=(20, 20))

        cm = np.corrcoef(df[selected_cols].values.T)
        sns.set(font_scale=1)
        sns.heatmap(cm, cbar=True, annot=True, square=True, fmt=".2f", annot_kws={"size": 10},
                    linewidths=0.5,
                    vmin=-1.0, vmax=1.0,
                    cmap=sns.diverging_palette(220, 10, as_cmap=True),  # "YlGnBu",
                    yticklabels=selected_cols, xticklabels=selected_cols)


class XgbcFeatureImportance:

    @staticmethod
    def plot_feature_importance(xgbc_model, importance_type, plot_horizontally=True):
        """
        :param: importance_type Enum(['weight', 'gain', 'cover', 'total_gain', 'total_cover'])
        --

        reference:
        https://stackoverflow.com/questions/17109608/change-figure-size-and-figure-format-in-matplotlib
        https://stackoverflow.com/questions/37627923/how-to-get-feature-importance-in-xgboost
        """
        plt.rcParams["figure.figsize"] = [9, 6]

        feature_important = xgbc_model.get_booster().get_score(importance_type=importance_type)
        keys = list(feature_important.keys())
        values = list(feature_important.values())

        data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=True)

        if plot_horizontally:
            ax = data.plot(kind='barh')
            # show values
            for p in ax.patches:
                ax.annotate("{:.2f}".format(float(p.get_width())), (p.get_width() * 1.005, p.get_y() * 1.005))

        else:
            ax = data.plot(kind='bar')
            # show values
            for p in ax.patches:
                ax.annotate("{:.2f}".format(float(p.get_height())), (p.get_x() * 1.005, p.get_height() * 1.005))

    @staticmethod
    def plot_feature_importance_by_fscore(xgbc_model):
        plt.rcParams["figure.figsize"] = [9, 6]
        plot_importance(xgbc_model)
        plt.show()
