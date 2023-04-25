import sys

sys.path.append("..")

from pathlib import Path
from processor import Processor


def main():
    """
    Main function of the script.
    """
    processor = Processor()

    # Get a list of all files in the '../data/raw' directory with '.csv' extension
    raw_dir = Path("data/raw/time_vs_tile_size")
    csv_files = list(raw_dir.glob("*.csv"))

    if csv_files is None:
        print("No files found in the raw directory")

    # Process each CSV file
    for csv_file in csv_files:
        # Read the CSV file into a DataFrame
        temp_df = processor.read_csv_file(csv_file, column_names="num_tiles")

        # Construct the pickle file path in the '../data/interim' directory
        pickle_file = csv_file.with_suffix(".pkl")
        pickle_path = Path("data/processed") / pickle_file.name

        # Save the DataFrame as a pickle file
        processor.save_dataframe_to_pickle(temp_df, pickle_path)


if __name__ == "__main__":
    main()
