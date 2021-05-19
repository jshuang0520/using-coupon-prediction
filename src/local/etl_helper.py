# -*- coding: utf-8 -*-
#import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
from src.utility.utils import Logger


class Extract:
    """
    naming follows
    1. source - e.g. cloud(s3, google drive, ...), local file, db(mysql, neo4j, ...)
    2. file type - e.g. csv, json, parquet, yaml, pickle, ...
    3. to which data type - e.g. pd
    """

    def __init__(self):
        self.logger = Logger().get_logger('extract')

    def read(self, file_path, col_mapping=None):
        """
        read_local_csv_to_pd
        """
        df = pd.read_csv(file_path, na_values=["?", "NaN", "nan", None])  # we set some common na_values as default
        if col_mapping:
            df.rename(columns=col_mapping, inplace=True)

        self.logger.info("shape: {}".format(df.shape))
        self.logger.info("columns: {}".format(df.columns))
        self.logger.info(df.info())

        # missing values
        missing_values_count = df.isnull().sum()
        total_cells = np.product(df.shape)
        total_missing = missing_values_count.sum()
        percent_missing = (total_missing / total_cells) * 100
        self.logger.info("missing percentage: {:.{prec}f}%".format(percent_missing, prec=2))

        sns.heatmap(df.isnull(), cbar=False)
        #msno.matrix(df)

        # sns.set_style("darkgrid", {"font.sans-serif": ['simhei', 'Droid Sans Fallback']})
        # msno.heatmap(df)

        return df
