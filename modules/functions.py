import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
import warnings
warnings.filterwarnings("ignore")


def write_ydata_report_to_file(df: pd.DataFrame, filename: str) -> None:
    """
    Function to write ydata report to file.

    :param df: The raw DataFrame containing data.
    :param filename: The filename to save the report to.
    :return: None
    """

    ydata_report = ProfileReport(df)
    ydata_report.to_file(filename)

def show_numeric_info(df: pd.DataFrame, column_name: str) -> None:
    """
    Function to write ydata report to file.

    :param df: The raw DataFrame containing data.
    :param column_name: The column name to save the report to.
    :return: None
    """

    print(df[column_name].describe())

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    sns.boxplot(x=df[column_name], ax=axes[0], color='blue')
    axes[0].set_title(f'Box plot of {column_name}', fontsize=16, fontweight='bold')

    axes[1].hist(df[column_name], bins=50, color='blue', edgecolor='black')
    axes[1].set_xlabel(column_name, fontsize=14)
    axes[1].set_ylabel('Frequency', fontsize=14)
    axes[1].set_title(f'Histogram of {column_name}', fontsize=16, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()