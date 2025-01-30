""" This script reads the CSV files in the specified folder generated with the BagGetFilter tool
    and merges them into a single CSV file with all the topics. Then joins the ones that are in
    same method (same first word in the file name)
    This has to be modified for simulation data since topic names and messages are different
"""
import os
import pandas as pd
import rospkg
import ast
import numpy as np

rospack = rospkg.RosPack()
package_dir = rospack.get_path("mml_guidance")
folder_path = package_dir + "/hardware_data/RAL_csv"

is_multiple_runs = True

# create an empty dictionary to store the dataframes for each group
dfs_by_group = {}


def process_pose_stamped(df):
    new_columns = ["rosbagTimestamp"]
    keep_columns = ["rosbagTimestamp"]
    for i, col in enumerate(df.columns):
        if col == "position":
            new_columns.extend(
                [f"position_{suffix[0]}" for suffix in df.columns[i + 1 : i + 4]]
            )
            keep_columns.extend(df.columns[i + 1 : i + 4])
        elif col == "orientation":
            new_columns.extend(
                [f"orientation_{suffix[0]}" for suffix in df.columns[i + 1 : i + 5]]
            )
            keep_columns.extend(df.columns[i + 1 : i + 5])
    new_df = df[keep_columns].copy()
    if len(new_columns) > 0:
        new_df.columns = new_columns
    return new_df


# loop through all the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        first_word = "_".join(filename.split("_")[0:2])

        df = pd.read_csv(os.path.join(folder_path, filename), low_memory=False)

        # extract the topic name from the file name (-4 removes the ".csv" extension)
        last_words = "_".join(filename.split("_")[4:])[0:-4]
        last_words = last_words.replace("slash_", "")
        if last_words == "xyTh_estimate":
            # Convert the cov (covariance) string series to a list of floats
            float_list = df["cov"].apply(lambda x: ast.literal_eval(x)).tolist()
            # Filter the list to keep only the first and fifth non-zero values
            df["cov"] = [[elem[0], elem[4]] if any(elem) else [] for elem in float_list]
            columns_to_drop = [
                "header",
                "seq",
                "stamp",
                "frame_id",
                "secs",
                "nsecs",
                "mean",
                "yaw",
                "weight",
            ]
            drop_col = False
            for i, col in enumerate(df.columns):
                # ignore columns that are followed by the "all_particle" column
                if i < len(df.columns) - 1 and df.columns[i] == "all_particle":
                    drop_col = True
                if drop_col:
                    columns_to_drop.append(col)
            # drop the columns that are followed by the "all_particle" column
            df.drop(columns_to_drop, axis=1, inplace=True)
            # add the determinant of the covariance matrix (for uncorrelated variables)
            df["cov det"] = df["cov"].apply(lambda x: np.prod(x))
            # remove rows with cov det higher than 5. This is done to omit unusually large covariances
            # that are higher than what a uniform distribution would give in the current workspace
            df = df[df["cov det"] < 5]
        elif (
            last_words == "rail_nwu_pose_stamped"
            or last_words == "robot0_odom"
            or last_words == "takahe_nwu_pose_stamped"
            or last_words == "truth_NWU"
        ):
            df = process_pose_stamped(df)
            if last_words == "robot0_odom":
                last_words = "rail_nwu_pose_stamped"
            elif last_words == "truth_NWU":
                last_words = "takahe_nwu_pose_stamped"
        elif last_words in {
            "xyTh_predictions",
            "noisy_measurement",
            "fov_coord",
            "des_fov_coord",
        }:  # odom
            print("ignoring file: ", filename)
            continue
        elif last_words == "desired_state":
            df = df[["rosbagTimestamp", "x", "y", "z", "yaw"]]
            # Transform from NED to NWU
            df["y"] = df["y"] * -1
        elif last_words == "err_estimation" or last_words == "err_tracking":
            df = df[["rosbagTimestamp", "x", "y"]]
            df["norm"] = (df["x"] ** 2 + df["y"] ** 2) ** 0.5
        elif last_words == "xyTh_estimate_cov_det":
            df = df[df["data"] < 5]

        df.rename(
            columns={
                col: f"{last_words}_{col}" if i >= 0 else col
                for i, col in enumerate(df.columns)
            },
            inplace=True,
        )
        df.columns = df.columns.str.replace("_", " ")
        df.columns = df.columns.str.replace("slash", "")
        # add the dataframe to the dictionary for the appropriate group
        if first_word in dfs_by_group:
            dfs_by_group[first_word].append(df)
        else:
            dfs_by_group[first_word] = [df]

for group_name, group_dfs in dfs_by_group.items():
    joined_df = pd.concat(group_dfs, axis=1)

    outdir = folder_path + "/joined"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    print("Creating file: " + f"{outdir}/{group_name}_joined.csv")
    joined_df.to_csv(f"{outdir}/{group_name}_joined.csv", index=False)

if is_multiple_runs:
    print("\nJoining all runs for each method")
    dfs_by_group = {}

    # Joins all csv into a big CSV with all the runs per method
    for filename in os.listdir(outdir):
        if filename.endswith(".csv"):
            first_word = filename.split("_")[0]

            df = pd.read_csv(os.path.join(outdir, filename), low_memory=False)

            if first_word in dfs_by_group:
                dfs_by_group[first_word].append(df)
            else:
                dfs_by_group[first_word] = [df]

    # loop through the dictionary and join the dataframes for each group
    for group_name, group_dfs in dfs_by_group.items():
        joined_df = pd.concat(group_dfs)

        outdir = os.path.join(folder_path, "all_runs")
        os.makedirs(outdir, exist_ok=True)
        output_filename = os.path.join(outdir, f"{group_name}_all_runs.csv")
        print("Creating file: " + output_filename)
        joined_df.to_csv(output_filename, index=False)
