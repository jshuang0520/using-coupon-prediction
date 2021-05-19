# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CreateTargetLabel:
    """
    According to the definition,
    1) buy with coupon within (include) 15 days ==> 1
    2) buy with coupon but out of 15 days ==> 0
    3) buy without coupon ==> -1 (we don't care)
    """

    @staticmethod
    def label(row):
        # buy without coupon
        if np.isnan(row['Date_received']):
            return -1
        # buy with coupon within (include) 15 days ==> 1 ; buy with coupon but out of 15 days ==> 0
        if not np.isnan(row['Date']):
            td = pd.to_datetime(row['Date'], format='%Y%m%d') - pd.to_datetime(row['Date_received'], format='%Y%m%d')
            if td <= pd.Timedelta(15, 'D'):
                return 1
        return 0


class ColumnPreprocess:
    """
    preprocess and get basic info for each column
    """

    @staticmethod
    def get_weekday(in_date):
        """
        Generate features - weekday acquired coupon
        """
        if (np.isnan(in_date)) or (in_date == -1):
            return in_date
        else:
            return pd.to_datetime(in_date, format="%Y%m%d").dayofweek + 1  # add one to make it from 0~6 -> 1~7

    @staticmethod
    def is_weekend(weekday: str):
        """
        check ColumnPreprocess.is_weekend
        represents weekday from 1~7 (not 0~6)
        """
        if str(weekday) in [6, 7]:
            return 1
        else:
            return 0

    # Generate features - coupon discount and distance

    """
    Discount_rate
    ex. 20:1, 10:5, NaN
    """

    @staticmethod
    def get_discount_type(row):
        if row == 'null':
            return 'null'
        elif ':' in row:
            return 1
        else:
            return 0

    @staticmethod
    def convert_rate(row):
        """
        Convert discount to rate
        """
        if row == 'null':
            return 1.0
        elif ':' in row:
            rows = row.split(':')
            return 1.0 - float(rows[1]) / float(rows[0])
        else:
            return float(row)

    @staticmethod
    def get_discount_man(row):
        if ':' in row:
            rows = row.split(':')
            return int(rows[0])
        else:
            return 0

    @staticmethod
    def get_discount_jian(row):
        if ':' in row:
            rows = row.split(':')
            return int(rows[1])
        else:
            return 0

    def process_data(self, df):
        # process User_id
        df['User_id'] = df['User_id'].astype('str')

        # process Merchant_id
        df['Merchant_id'] = df['Merchant_id'].astype('str')

        # process Merchant_id
        df['Coupon_id'] = df['Coupon_id'].astype('str')

        # process Discount_rate
        df['discount_rate'] = df['Discount_rate'].astype('str').swifter.apply(self.convert_rate)
        df['discount_man'] = df['Discount_rate'].astype('str').swifter.apply(self.get_discount_man)
        df['discount_jian'] = df['Discount_rate'].astype('str').swifter.apply(self.get_discount_jian)
        df['discount_type'] = df['Discount_rate'].astype('str').swifter.apply(self.get_discount_type)

        # process Distance
        df.loc[df["Distance"].isna(), "Distance"] = 99

        # process Date_received
        df['weekday'] = df['Date_received'].swifter.apply(self.get_weekday)
        df['is_weekend'] = df['weekday'].astype('str').swifter.apply(self.is_weekend)
        # - weekday get_dummies
        weekday_cols = ['weekday_' + str(i) for i in range(1, 8)]
        df_weekday_one_hot = pd.get_dummies(df['weekday'].replace(-1, np.nan))
        df_weekday_one_hot.columns = weekday_cols
        df[weekday_cols] = df_weekday_one_hot

        return df


