import pandas as pd
import numpy as np
from typing import List


class DataPrep:
    """This class is used to merge the dataframes"""
    def __init__(self, path_to_prod_data:str, path_to_sales_data:str):
        self.path_to_prod_data = path_to_prod_data
        self.path_to_sales_data = path_to_sales_data
        self.df_products = pd.read_csv(self.path_to_prod_data)
        self.df_sales = pd.read_csv(self.path_to_sales_data)

    def data_merger(self):
        df_products = self.df_products
        df_sales = self.df_sales
        df_sales = df_sales.dropna(subset=["net_amount", "gross_amount"])
        df_sales['variant_parent'] = df_sales['variant'].map(lambda x: x//1000)
        df_start = pd.merge(df_sales, df_products, left_on = 'variant_parent', right_on='article', how = 'left').sort_values('date', ascending = True)
        df_start["week"] = (
                    "week" +
                    (pd.to_datetime(df_start["date"])
                    .rank(method="dense")
                    .sub(1)
                    .floordiv(7)
                    .astype(int)
                    .astype(str))
                )
        return df_start 


class DailyTransfomer:
    """This class is used to transform the data into daily format"""
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def transform(self, group_columns:List):
        df_training = self.dataframe.copy()
        df_training["price"] = np.where(
                                    df_training["purchases"] > 0,
                                    df_training["gross_amount"] / df_training["purchases"],
                                    np.nan
                                ).round(2)
        df_training["discount"] = np.where(
                                    df_training["gross_amount"] > 0,
                                    1 - df_training["net_amount"] / df_training["gross_amount"],
                                    0.0
                                ).round(2)
        
        df_daily = (
                df_training
                .groupby(group_columns, as_index=False)
                .agg(
                    purchases=("purchases", "sum"),
                    price=("price", "mean"),
                    discount=("discount", "mean")
                )
            )

        df_daily[["price", "discount"]] = df_daily[["price", "discount"]].round(2)
        
        return df_daily

class FullDateTransformer:
    """Transform a column of dates into a column of full dates with correpsonding data"""

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def full_date_transformer(self):
        df_daily = self.dataframe.copy()

        dates = df_daily[["date", "week"]].drop_duplicates()
        articles = df_daily[["article", "week"]].drop_duplicates()
        full_index = articles.merge(dates, on="week", how="left")
        df_price_group = df_daily.groupby(["week", "article"], as_index=False).agg(
            price_group_mean=("price", "mean"), discount_group_mean=("discount", "mean")
        )
        df_daily_full = full_index.merge(
            df_daily, on=["date", "week", "article"], how="left"
        )
        df_daily_full["purchases"] = df_daily_full["purchases"].fillna(0)
        df_daily_full = df_daily_full.merge(
            df_price_group, on=["week", "article"], how="left"
        )
        df_daily_full["price"] = df_daily_full["price"].fillna(
            df_daily_full["price_group_mean"]
        )
        df_daily_full["discount"] = df_daily_full["discount"].fillna(
            df_daily_full["discount_group_mean"]
        )
        df_daily_full = df_daily_full.drop(
            columns=["price_group_mean", "discount_group_mean"]
        )

        return df_daily_full