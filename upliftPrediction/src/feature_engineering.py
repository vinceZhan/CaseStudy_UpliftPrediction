import pandas as pd
import numpy as np
from typing import List

class FeatureGenerator:
    """Generate features for the model"""
    def __init__(self, dataframe, path_to_prod_table):
        self.dataframe = dataframe
        self.dataframe_prod = pd.read_csv(path_to_prod_table)
    
    def generate_features(self, prod_features:List):
        df_daily_full = self.dataframe.copy()
        df_daily_full['markdown'] = df_daily_full['week'].map(lambda x: 0 if x=='week0' else 1)
        sold_days = (
            df_daily_full
            .assign(sold_flag=(df_daily_full["purchases"] > 0).astype(int))
            .groupby(['week',"article"])["sold_flag"]
            .sum()
            .reset_index(name="sold_days")
        )

        baseline = (
            df_daily_full[df_daily_full["markdown"] == 0]
            .groupby(["article"], as_index=False)
            .agg(base_line_price=("price", "mean"),
                base_line_purchases=("purchases", "mean"))
        )

        df_features = df_daily_full.merge(sold_days, on=["week", "article"], how="left").merge(baseline, on=["article"], how="left")
        
        df_features["price"] = df_features["price"].fillna(
            df_features["base_line_price"]
        )

        df_features['price_ratio'] = df_features['price']/df_features['base_line_price']
        df_features["discount_rank"] = pd.qcut(
            df_features["discount"],
            q=4,
            labels=False
        ) + 1

        df_features["is_first_md_day"] = (
            (df_features["markdown"] == 1) &
            (df_features.groupby("article")["markdown"].shift(1).fillna(0) == 0)
        ).astype(int)

        df_features_ready = pd.merge(df_features, self.dataframe_prod[prod_features+['article']], on='article', how = 'left')
        df_feat = df_features_ready.sort_values(["article", "date"]).copy()
        df_feat["purchases_lag1"] = df_feat.groupby("article")["purchases"].shift(1)
        df_feat["purchases_lag1"] = df_feat["purchases_lag1"].fillna(0)
        return df_feat
