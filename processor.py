import pandas as pd


class Processor:

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
    
    def create_scaling_df(self, num_cores, serial_time, parallel_time):
        
        # Create a dictionary to hold the data
        data = {
            "num_cores": num_cores,
            "serial_time": serial_time,
            "parallel_time": parallel_time
        }

        # Create a pandas DataFrame from the dictionary
        df = pd.DataFrame(data)

        # Use chaining to calculate speed up ratio and parallel efficiency
        df = df.assign(speed_up=df['serial_time'] / df['parallel_time']) \
            .assign(parallel_efficiency=lambda x: x['speed_up'] / x['num_cores'])
            
        return df