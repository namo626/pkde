import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Any
import os

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
        self.report_dir = "../reports/latex_docs/figures"

    def plot_single_kde(
        self,
        df: pd.DataFrame,
        title: str = None,
        hist: Optional[bool] = False,
        rug: Optional[bool] = False,
        kde: Optional[bool] = False,
        save: Optional[bool] = False,
        filename: str = "single_kde.png",
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
                df.x[::100], height=0.01, color="black", label="N = 8000 Samples"
            )
        if kde:
            sns.lineplot(data=df, x="x", y="p(x)", color="red", label="KDE")

        if title is not None:
            plt.title(title, fontsize=18)

        plt.xlabel("x", fontsize=14)
        plt.ylabel("p(x)", fontsize=14)
        plt.legend()

        if save:
            plt.savefig(os.path.join(self.report_dir, filename))

        plt.show()

    def plot_multiple_kdes(
        self,
        dfs: pd.DataFrame,
        titles: Optional[list[str]] = None,
        hist: Optional[bool] = False,
        kde: Optional[bool] = False,
        fig_size: tuple = (16, 10),
        save: Optional[bool] = False,
        filename: str = "four_kdes.png",
    ) -> Any:
        num_plots = len(dfs)
        rows = 2
        cols = 2

        fig, axs = plt.subplots(rows, cols, figsize=fig_size)
        axs = axs.ravel()  # Flatten the 2D array of subplots

        for i in range(num_plots):
            if hist:
                axs[i].hist(
                    dfs[i]["y"],
                    density=True,
                    bins=50,
                    color="blue",
                    label="Histogram",
                )

            if kde:
                sns.lineplot(
                    data=dfs[i],
                    x="x",
                    y="p(x)",
                    color="red",
                    label="KDE",
                    ax=axs[i],
                    linewidth=2,
                )

            axs[i].set_xlabel("x", fontsize=14)
            axs[i].set_ylabel("p(x)", fontsize=14)

            if titles is not None:
                axs[i].set_title(titles[i], fontsize=16)

        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.report_dir, filename))

    def plot_execution_time(
        self,
        execution_times_df: pd.DataFrame,
        x_grid: str,
        fig_size: tuple[int, int] = (12, 8),
        save: bool = False,
        filename: str = "execution.png",
    ) -> None:
        """
        Plots execution times from a Pandas DataFrame as line plots.

        Args:
        -----
        execution_times_df (pd.DataFrame): The DataFrame containing execution times.
        x_grid (str): The x-axis data column name.
        figsize (tuple[int, int], optional): The figure size. Default is (12, 8).
        save (bool, optional): Whether to save the plot to a file. Default is False.
        filename (str, optional): The filename for the saved plot. Default is "execution.png".

        Returns:
        --------
        None
        """
        # Set the figure size
        plt.figure(figsize=fig_size)

        # Define line plot parameters
        line_styles = ["--", "--", "--", "--"]
        marker_styles = ["o", "s", "^"]
        colors = ["blue", "green", "red"]
        labels = ["Serial", "SIMD", "MPI"]

        # Loop through each line plot
        for i in range(len(labels)):
            sns.lineplot(
                x=x_grid,
                y=labels[i],
                data=execution_times_df,
                label=labels[i],
                marker=marker_styles[i],
                color=colors[i],
                lw=1,
                ls=line_styles[i],
            )

        # Set labels and title
        if x_grid == "num_points":
            plt.xlabel("Number of Points", fontsize=14)
        else:
            plt.xlabel("Number of Cores", fontsize=14)
        plt.ylabel("Execution Time (seconds)", fontsize=14)
        plt.title("Kernel Density Estimation", fontsize=18)

        # Add legend
        plt.legend()

        if save:
            plt.savefig(os.path.join(self.report_dir, filename))

        # Show the plot
        plt.show()

    def plot_speedup_efficiency(
        self,
        df1: pd.DataFrame,
        fig_size: tuple[int, int] = (16, 10),
        save: bool = False,
        filename: str = "temp_name.png",
    ) -> None:
        """
        Plot two separate line plots side by side for speed up ratio and parallel efficiency.

        Args:
            df1 (pd.DataFrame): First data frame with columns 'num_cores', 'speed_up', and 'parallel_efficiency'.
            fig_size (Tuple[int, int], optional): Figure size. Defaults to (16, 10).
            save (bool, optional): Whether to save the plot to a file. Defaults to False.
            filename (str, optional): Filename to save the plot. Defaults to "temp_name.png".
        """

        # Set figure size
        plt.figure(figsize=fig_size)

        # Plot speed up ratio
        plt.subplot(1, 2, 1)
        sns.lineplot(
            x="num_cores",
            y="speed_up",
            data=df1,
            label="MPI",
            linewidth=2,
            linestyle="--",
            marker="o",
            color="red",
        )

        plt.xlabel("Number of Cores", fontsize=14)
        plt.ylabel("Speed up Ratio", fontsize=14)
        plt.title("Speed up Ratio vs. Number of Cores", fontsize=16)
        plt.legend()

        # Plot parallel efficiency
        plt.subplot(1, 2, 2)
        sns.lineplot(
            x="num_cores",
            y="parallel_efficiency",
            data=df1,
            label="MPI",
            linewidth=2,
            linestyle="--",
            marker="o",
            color="red",
        )

        plt.xlabel("Number of Cores", fontsize=14)
        plt.ylabel("Parallel Efficiency", fontsize=14)
        plt.title("Parallel Efficiency vs. Number of Cores", fontsize=16)
        plt.legend()

        if save:
            plt.savefig(os.path.join(self.report_dir, filename))

        # Show the plot
        plt.show()
