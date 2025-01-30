import os
import pandas as pd
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import get_typestore, Stores

# Path to the bag file
bag_directory = "/home/andrespulido/mml_ws/src/mml_guidance/hardware_data/RAL_csv/LCSS/"
bag_file_path = bag_directory + "transformer_09_10_2024_1"

is_write_csv = True  # True is to write csv from a bag file and False is to join the csvs to a single run

if is_write_csv:
    # Create a typestore and get the string class.
    typestore = get_typestore(Stores.LATEST)

    topic_to_include = {
        "/err_estimation": ["msg.point.x", "msg.point.y"],
        "/err_tracking": ["msg.point.x", "msg.point.y"],
        "/is_update": ["msg.data"],
        "/cov": ["msg.data"],
    }

    # Create reader instance and open for reading.
    with Reader(bag_file_path) as reader:
        # Initialize a dictionary to store the data for each column
        column_data = {
            f"{topic.split('/')[-1].replace('_', ' ')} {field.split('.')[-1]}": []
            for topic, fields in topic_to_include.items()
            for field in fields
        }

        # Iterate over messages and collect data
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic in topic_to_include:
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                for field in topic_to_include[connection.topic]:
                    value = eval(field)
                    col_name = f"{connection.topic.split('/')[-1].replace('_', ' ')} {field.split('.')[-1]}"
                    column_data[col_name].append(value)

        max_length = max(len(values) for values in column_data.values())

        # Pad shorter columns with None
        for col in column_data:
            while len(column_data[col]) < max_length:
                column_data[col].append(None)

        # Create a DataFrame from the collected data
        df = pd.DataFrame(column_data)

        # Calculate the norm for topics with "err" in their name
        for topic in topic_to_include:
            if "err" in topic:
                x_col = f"{topic.split('/')[-1].replace('_', ' ')} x"
                y_col = f"{topic.split('/')[-1].replace('_', ' ')} y"
                norm_col = f"{topic.split('/')[-1].replace('_', ' ')} norm"

                # Check if the columns exist before calculating the norm
                if x_col in df.columns and y_col in df.columns:
                    df[norm_col] = np.sqrt(df[x_col] ** 2 + df[y_col] ** 2)
                else:
                    print(f"Warning: Columns {x_col} or {y_col} not found in DataFrame")

                # print the mean of the norm column
                print(f"Mean of {norm_col}: {df[norm_col].mean()}")

        # Write the DataFrame to a CSV file
        csv_file_path = bag_file_path + ".csv"
        df.to_csv(csv_file_path, index=False)

    print(f"Data saved to {csv_file_path}")
else:
    include_cov_data = True

    # Get all the csv files in the directory
    csv_files = [f for f in os.listdir(bag_directory) if f.endswith(".csv")]
    print("CSV files found:", csv_files)

    # Initialize an empty DataFrame to store the joined data
    joined_df = pd.DataFrame()

    # Read each CSV file and join them into a single DataFrame
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(bag_directory, csv_file))
        joined_df = pd.concat([joined_df, df], axis=0)

    if include_cov_data:
        # Add a column to store the covariance determinant
        cov_dir = bag_directory + "../"
        csv_files = [f for f in os.listdir(cov_dir) if f.endswith(".csv")]
        cov_df = pd.DataFrame()
        for csv_file in csv_files:
            df = pd.read_csv(os.path.join(cov_dir, csv_file))
            cov_df = pd.concat([cov_df, df], axis=0)

        cov_df.reset_index(drop=True, inplace=True)
        joined_df["cov data"] = cov_df["cov data"]

    # convert the cov data column to string
    joined_df = joined_df.dropna(subset=["cov data"])
    joined_df["cov data"] = joined_df["cov data"].astype(str)

    float_list = (
        joined_df["cov data"]
        .dropna()
        .apply(
            lambda x: [
                float(num)
                for num in x.strip("[]").replace("\n", " ").replace("\r", "").split()
                if x.strip()
            ]
        )
        .tolist()
    )
    # cov det is the product of the first and fifth non-zero values
    joined_df["xyTh estimate cov det"] = [
        elem[0] * elem[4] if any(elem) else None for elem in float_list
    ]
    # remove rows with cov det higher than 5. This is done to omit unusually large covariances
    # that are higher than what a uniform distribution would give in the current workspace
    joined_df = joined_df[joined_df["xyTh estimate cov det"] < 5]

    # Write the joined DataFrame to a CSV file
    joined_csv_file_path = bag_directory + "../../all_runs/Information_all_runs.csv"
    joined_df.to_csv(joined_csv_file_path, index=False)

    print(f"\nData saved to {joined_csv_file_path}")
