import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from typing import List
import warnings
warnings.filterwarnings("ignore")

class HelperFunctions:
    def __init__(self):
        return

    @staticmethod
    def write_ydata_report_to_file(df: pd.DataFrame, filename: str) -> None:
        """
        Function to write ydata report to file.

        :param df: The raw DataFrame containing data.
        :param filename: The filename to save the report to.

        :return: None
        """

        ydata_report = ProfileReport(df)
        ydata_report.to_file(filename)

    @staticmethod
    def show_numeric_info( df: pd.DataFrame, column_name: str, n_bins: int) -> None:
        """
        Function to write ydata report to file.

        :param df: The raw DataFrame containing data.
        :param column_name: The column name to save the report to.
        :param n_bins: The number of bins to use for the histogram.

        :return: None
        """

        print(df[column_name].describe())

        print('Number of null values:', df[column_name].isnull().sum())

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        sns.boxplot(x=df[column_name], ax=axes[0], color='blue')
        axes[0].set_title(f'Box plot of {column_name}', fontsize=16, fontweight='bold')

        axes[1].hist(df[column_name], bins=n_bins, color='blue', edgecolor='black')
        axes[1].set_xlabel(column_name, fontsize=14)
        axes[1].set_ylabel('Frequency', fontsize=14)
        axes[1].set_title(f'Histogram of {column_name}', fontsize=16, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_categorical_info(df: pd.DataFrame, column_name: str) -> None:
        """
        Function to show categorical column information.

        :param df: The DataFrame containing data.
        :param column_name: The categorical column name to analyze.

        :return: None
        """

        print(df[column_name].describe())

        print('Number of null values:', df[column_name].isnull().sum())

        # Count occurrences of each category
        counts = df[column_name].value_counts()

        # Set figure size
        plt.figure(figsize=(12, 6), facecolor="black")

        # Create horizontal bar chart
        ax = sns.barplot(y=counts.index, x=counts.values, color='royalblue')
        ax.set_facecolor("black")

        # Adjust text labels inside bars
        for index, value in enumerate(counts.values):
            ax.text(min(value * 0.05, max(counts.values)), index, str(value),
                    color="white", fontsize=14, fontweight='bold', va='center')

        # Add a title to the plot
        plt.title(f"Distribution of {column_name} categories", fontsize=16, fontweight='bold', color='white')
        # Improve axis visibility
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks(fontsize=14, color='white')  # Make category labels bigger
        plt.grid(axis="x", linestyle="--", alpha=0.5)  # Add subtle grid lines

        # Remove borders
        sns.despine(left=True, bottom=True)

        # Show plot
        plt.show()

        print(df[column_name].dtype)

    @staticmethod
    def impute_categorical_unknown_values(df: pd.DataFrame, column_name: str, strategy: str) -> pd.DataFrame:
        """
        This function imputes categorical columns with unknown values.

        :param df: The raw DataFrame containing data.
        :param column_name: The column name to impute.

        :return: The imputed DataFrame.
        """
        df[column_name] = df[column_name].replace('unknown', np.nan)

        match strategy:
            case "mode":
                mode = df[column_name].mode()[0]
                df[column_name].fillna(mode, inplace=True)

            case _:
                print (f"[INFO] invalid imputation strategy '{strategy}'. No imputation will be performed. Supported strategies: ['mode']")

        return df

    @staticmethod
    def transform_string_into_category_type(df: pd.DataFrame, column_names: List[str]) -> pd.DataFrame:
        """
        This function transforms categorical column into a categorical column.
        :param df:
        :param column_name:

        :return: transformed DataFrame.
        """

        for column_name in column_names:
            if column_name in df.columns:
                df[column_name] = df[column_name].astype('category')

        return df


