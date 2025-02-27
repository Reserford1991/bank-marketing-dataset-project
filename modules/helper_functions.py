import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
import warnings
warnings.filterwarnings("ignore")

class TransformFunctions:
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
