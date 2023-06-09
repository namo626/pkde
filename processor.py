import pandas as pd


class Processor:
    def __init__(self) -> None:
        self.raw_data_dir = "../data/raw"
        self.interim_data_dir = "../data/interim"
        self.processed_data_dir = "../data/processed"

    def read_csv_file(self, filename: str, column_names: list[str]) -> pd.DataFrame:
        """
        Reads in a CSV file and returns it as a Pandas DataFrame.

        Args:
        -----
        filename (str): The name of the CSV file.
        column_names (list[str]) List of column names for dataframe

        Returns:
        --------
        df (pd.DataFrame): The Pandas DataFrame of the CSV file.
        """
        if column_names == "kde":
            columns = ["x", "y", "p(x)"]

        elif column_names == "num_points":
            columns = ["Number_Points", "Execution_Time"]

        elif column_names == "num_cores":
            columns = ["Number_Cores", "Execution_Time"]

        elif column_names == "num_threads":
            columns = ["Number_Threads", "Execution_Time"]

        elif column_names == "num_tiles":
            columns = ["Number_Tiles", "Execution_Time"]
        else:
            columns = None

        return pd.read_csv(filename, header=None, names=columns)

    def read_pickle(self, filename: str) -> pd.DataFrame:
        """
        Reads in a pickle file and returns it as a Pandas DataFrame.

        Args:
        -----
        filename (str): The name of the pickle file.

        Returns:
        --------
        df (pd.DataFrame): The Pandas DataFrame of the pickle file.
        """
        return pd.read_pickle(filename)

    def save_dataframe_to_pickle(self, dataframe, file_path):
        """
        Saves a pandas DataFrame into a pickle file.

        Args:
            dataframe (pd.DataFrame): The DataFrame to be saved.
            file_path (str): The file path where the pickle file will be saved.

        Returns:
            None
        """
        dataframe.to_pickle(file_path)
        print(f"Dataframe saved to pickle file: {file_path}")

    def create_scaling_df(self, mpi_df: pd.DataFrame) -> pd.DataFrame:
        mpi_df.reset_index(inplace=True)
        # Use chaining to calculate speed up ratio and parallel efficiency
        mpi_df = mpi_df.assign(Speed_Up=mpi_df["Serial"] / mpi_df["MPI"]).assign(
            Parallel_Efficiency=lambda x: x["Speed_Up"] / x["Number_Cores"]
        )

        return mpi_df
