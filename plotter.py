import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Any

sns.set_style(style="darkgrid")


class Plotter:
    """
    A class for plotting kernel density estimates (KDEs) and histograms of data.

    Args:
    -----
    None

    Methods:
    --------
    read_csv_file(filename: str) -> pd.DataFrame:
        Reads in a CSV file and returns it as a Pandas DataFrame.

    plot_kde(df: pd.DataFrame, hist: Optional[bool] = False, rug: Optional[bool] = False, kde: Optional[bool] = False) -> Any:
        Plots the KDE and/or histogram of a given Pandas DataFrame.
    """

    def __init__(self) -> None:
        pass

    def read_csv_file(self, filename: str) -> pd.DataFrame:
        """
        Reads in a CSV file and returns it as a Pandas DataFrame.

        Args:
        -----
        filename (str): The name of the CSV file.

        Returns:
        --------
        df (pd.DataFrame): The Pandas DataFrame of the CSV file.
        """
        return pd.read_csv(filename, header=None, names=["x", "y", "p(x)"])

    def plot_kde(
        self,
        df: pd.DataFrame,
        title: str = None,
        hist: Optional[bool] = False,
        rug: Optional[bool] = False,
        kde: Optional[bool] = False,
    ) -> Any:
        """
        Plots the KDE and/or histogram of a given Pandas DataFrame.

        Args:
        -----
        df (pd.DataFrame): The Pandas DataFrame to plot.
        hist (bool, optional): Whether or not to plot a histogram. Default is False.
        rug (bool, optional): Whether or not to plot a rug plot. Default is False.
        kde (bool, optional): Whether or not to plot a KDE. Default is False.

        Returns:
        --------
        None
        """
        plt.figure(figsize=(12, 8))

        if hist:
            sns.histplot(
                data=df,
                x="y",
                kde=False,
                stat="density",
                bins=100,
                color="blue",
                label="Histogram",
            )

        if rug:
            sns.rugplot(
                df.x[::100], height=0.02, color="black", label="N = 8000 Samples"
            )
        if kde:
            sns.lineplot(data=df, x="x", y="p(x)", color="red", label="KDE")

        if title is not None:
            plt.title(title, fontsize=18)

        plt.ylabel("p(x)")
        plt.legend()
        plt.show()
