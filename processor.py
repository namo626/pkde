import pandas as pd


class Processor:
    def __init__(self) -> None:
        self.raw_data_dir = "../data/raw"
        self.interim_data_dir = "../data/interim"
        self.processed_data_dir = "../data/processed"

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

    def create_scaling_df(self, num_cores, serial_time, parallel_time):
        # Create a dictionary to hold the data
        data = {
            "num_cores": num_cores,
            "serial_time": serial_time,
            "parallel_time": parallel_time,
        }

        # Create a pandas DataFrame from the dictionary
        df = pd.DataFrame(data)

        # Use chaining to calculate speed up ratio and parallel efficiency
        df = df.assign(speed_up=df["serial_time"] / df["parallel_time"]).assign(
            parallel_efficiency=lambda x: x["speed_up"] / x["num_cores"]
        )

        return df
