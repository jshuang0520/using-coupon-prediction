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
