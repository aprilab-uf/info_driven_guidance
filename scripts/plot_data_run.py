import os
import rospkg
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# Set the name of the input CSV file
rospack = rospkg.RosPack()
package_dir = rospack.get_path("mml_guidance")
folder_path = package_dir + "/hardware_data/RAL_csv/joined/"

is_plot = False
print_rms = False

include_data = {
    "err estimation norm": "$e$ [m]",
    "err tracking norm": "$e$ [m]",
    "entropy data": "Entropy",
    "n eff particles data": "Effective Number of Particles",
    "xyTh estimate cov det": "Measure of spread of particles $[m^2]$",
}
cropped_plot = {
    "desired state x",
    "rail nwu pose stamped position x",
    "takahe nwu pose stamped position x",
    "desired state y",
    "rail nwu pose stamped position y",
    "takahe nwu pose stamped position y",
}

# font sizes for the plot labels
font = {"family": "serif", "weight": "normal", "size": 20}
sns.set_theme()
sns.set_style("white")
sns.set_context("paper", font_scale=2)


def crop_col(df_col):
    """Crop the column of a dataframe between begin and end percentage of the time"""
    beg = time_bounds_dict[df_col.name][0]
    end = time_bounds_dict[df_col.name][1]
    # print("column: ", df_col.name, "beg: ", beg, "end: ", end)
    return df_col.dropna().iloc[beg:end]


def main():
    global time_bounds_dict
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            print("\nFilename: ", filename)
            csv_file = folder_path + filename
            df = pd.read_csv(csv_file, low_memory=False)

            # Begin and end percentage of the time to plot
            beg_end = np.array([0.0, 0.9])
            beg_end = (
                np.array([0.0, 0.75])
                if filename == "Information2_2024-04-19-16-11-40_joined.csv"
                else beg_end
            )
            # Get min and max time from topic with most data
            min_time = df["takahe nwu pose stamped rosbagTimestamp"].min() / 10e8
            max_time = df["desired state rosbagTimestamp"].max() / 10e8
            crop_time = (max_time - min_time) * beg_end
            print(
                "Time range before crop is: ", round(max_time - min_time, 2), "seconds"
            )
            time_bounds_dict = {}
            for col_name in df.columns:
                if col_name.endswith("rosbagTimestamp"):
                    x_label = col_name
                    df[col_name] = df[col_name] / 10e8 - min_time
                    print("df[x_label].max(): ", df[x_label].max(), x_label)
                    min_max_indx = [
                        df.index[df[x_label] >= crop_time[0]].tolist()[0],
                        df.index[df[x_label] >= crop_time[1]].tolist()[0],
                    ]
                time_bounds_dict[col_name] = min_max_indx
                df[col_name] = crop_col(df[col_name])

            min_time = df["takahe nwu pose stamped rosbagTimestamp"].min()
            max_time = df["desired state rosbagTimestamp"].max()
            print("Time range: ", round(max_time - min_time, 2), "seconds")
            # Crop the data between begin and end for zoomed in plot
            zoomed_beg_end = np.array([0.711, 0.74])
            zoomed_time = (max_time - min_time) * zoomed_beg_end

            # Occlusions
            occ_width = 0.75
            occ_center = np.array([-1.25, -1.05])
            # list of positions of the occlusions to plot in x and y
            occ_pos = [
                [
                    occ_center[0] - occ_width / 2,
                    occ_center[0] + occ_width / 2,
                    occ_center[0] + occ_width / 2,
                    occ_center[0] - occ_width / 2,
                    occ_center[0] - occ_width / 2,
                ],
                [
                    occ_center[1] - occ_width / 2,
                    occ_center[1] - occ_width / 2,
                    occ_center[1] + occ_width / 2,
                    occ_center[1] + occ_width / 2,
                    occ_center[1] - occ_width / 2,
                ],
            ]

            # Indices where 'is update data' is false
            indx_not_upd = np.where(
                df["is update data"].dropna().astype(bool).to_numpy() == False
            )[0]
            indx_occ = np.where(
                df["is occlusion data"].dropna().astype(bool).to_numpy() == True
            )[0]

            x_label = ""
            for col_name in df.columns:
                if col_name.endswith("rosbagTimestamp"):
                    x_label = col_name
                    df[x_label] = df[x_label] - min_time

                elif any(col_name == word for word in include_data.keys()):
                    # Set the y axis label to the full column name
                    col_name = col_name

                    # get first word from filename
                    first_word = filename.split("_")[0]
                    outdir = folder_path + "figures/" + first_word + "/"
                    if not os.path.exists(outdir):
                        os.mkdir(outdir)

                    # Create a new plot with the x and y data
                    fig_size = (8, 5.1)  # inches
                    dpi = 300
                    plt.figure(figsize=fig_size)

                    # Plot the data
                    plt.plot(df[x_label], df[col_name], linewidth=1)
                    plt.scatter(df[x_label], df[col_name], marker=".", s=20)

                    # Target not in FOV
                    for xi_u in df["is update rosbagTimestamp"][indx_not_upd]:
                        plt.axvline(
                            x=xi_u, alpha=0.4, color="k", linestyle="-", linewidth=0.8
                        )
                    plt.axvline(
                        x=df["is update rosbagTimestamp"][indx_not_upd[0]],
                        alpha=0.4,
                        color="k",
                        linestyle="-",
                        linewidth=0.8,
                        label=r"$\mathbf{x}_T \in \mathcal{S}$",
                    )
                    # vertical line when there is occulsion (every 6th occulsion to avoid clutter)
                    if indx_occ.size > 0:
                        for xi_o in df["rail nwu pose stamped rosbagTimestamp"][
                            indx_occ[::6]
                        ]:
                            plt.axvline(
                                x=xi_o,
                                alpha=0.1,
                                color="r",
                                linestyle="-",
                                linewidth=0.8,
                            )

                        plt.axvline(
                            x=df["rail nwu pose stamped rosbagTimestamp"][indx_occ[0]],
                            alpha=0.1,
                            color="r",
                            linestyle="-",
                            linewidth=0.8,
                            label=r"$\mathbf{x}_T \in \mathcal{O}$",
                        )

                    # Add the legend and axis labels
                    plt.xlabel("Time [s]", fontdict=font)
                    plt.ylabel(include_data[col_name], fontdict=font)
                    plt.ylim(0, 2.5) if col_name == "err estimation norm" else None
                    plt.legend(loc="upper right", fontsize=20)
                    plt.title(first_word, fontdict=font)
                    plt.tight_layout()
                    plt.savefig(
                        outdir + first_word + "_" + col_name.replace(" ", "_") + ".png",
                        dpi=dpi,
                    )
                    # plt.show() if is_plot else plt.close()
                    plt.close()

                elif any(col_name == word for word in cropped_plot):
                    # index of min and max time
                    min_max_indx = [
                        df.index[df[x_label] >= zoomed_time[0]].tolist()[0],
                        df.index[df[x_label] >= zoomed_time[1]].tolist()[0],
                    ]
                    # print("Max time: ", df[x_label][min_max_indx[0]], "for column: ", col_name)
                    time_bounds_dict[col_name] = min_max_indx

                if print_rms:
                    if any(col_name == word for word in include_data):
                        with open(outdir + "rms.csv", "a") as csvfile:
                            row_list = [col_name]
                            rms = round(np.sqrt(np.mean(df[col_name] ** 2)), 3)
                            print("RMS of " + col_name + ": " + str(rms))
                            row_list.append(rms)
                            writer = csv.writer(csvfile)
                            writer.writerow(row_list)
                    elif col_name == "is update data":
                        # percent of the total length that 'is update data' column is true
                        with open(outdir + "rms.csv", "a") as csvfile:
                            row_list = ["is update percent"]
                            perc = round(
                                100
                                * np.sum(df[col_name].dropna())
                                / len(df[col_name].dropna()),
                                3,
                            )
                            row_list.append(perc)
                            writer = csv.writer(csvfile)
                            writer.writerow(row_list)
                        print(
                            "Percent of time 'is update data' is true: "
                            + str(perc)
                            + "%"
                        )

            if filename == "Information_2023-07-06-16-53-57_joined.csv":
                # Zoomed in road network trajectories
                fig_size = (9, 6)  # inches
                dpi = 800
                plt.figure(figsize=fig_size)
                alphas_robot = np.linspace(
                    0.1, 0.8, len(crop_col(df["rail nwu pose stamped position x"]))
                )
                colors_robot = cm.Blues(alphas_robot)
                plt.scatter(
                    crop_col(df["rail nwu pose stamped position x"]),
                    crop_col(df["rail nwu pose stamped position y"]),
                    c=colors_robot,
                    marker=".",
                    label="Turtlebot",
                )
                plt.scatter(
                    df["rail nwu pose stamped position x"]
                    .dropna()
                    .iloc[14464 : 14464 + 690 : 40],
                    df["rail nwu pose stamped position y"]
                    .dropna()
                    .iloc[14464 : 14464 + 690 : 40],
                    s=35,
                    c="darkblue",
                    marker="+",
                    label="Turtlebot Future Positions",
                )
                alphas_quad = np.linspace(
                    0.1, 0.6, len(crop_col(df["takahe nwu pose stamped position x"]))
                )
                colors_quad = cm.Oranges(alphas_quad)
                plt.scatter(
                    crop_col(df["takahe nwu pose stamped position x"]),
                    crop_col(df["takahe nwu pose stamped position y"]),
                    c=colors_quad,
                    marker=".",
                    label="Quadcopter",
                )
                alphas_fov = np.linspace(
                    0.05, 0.55, len(crop_col(df["desired state x"]))
                )
                colors_fov = cm.Greens(alphas_fov)
                plt.scatter(
                    crop_col(df["desired state x"]),
                    crop_col(df["desired state y"]),
                    alpha=0.7,
                    c=colors_fov,
                    marker="s",
                    s=4000,
                )
                plt.plot(
                    crop_col(df["desired state x"]),
                    crop_col(df["desired state y"]),
                    alpha=0.2,
                    color="g",
                    label="Reference Position",
                )
                plt.grid(linestyle="--", linewidth=0.5)
                plt.xlabel("X position [m]", fontdict=font)
                plt.ylabel("Y position [m]", fontdict=font)
                plt.title(
                    "Field of View Road Network with beg and end: "
                    + str(zoomed_beg_end),
                    fontdict=font,
                )
                plt.legend()
                plt.savefig(outdir + "zoomed_road" + ".png", dpi=dpi)
                plt.show() if is_plot else plt.close()

            # Complete road network trajectories
            fig_size = (8, 6)  # inches
            dpi = 800
            plt.figure(figsize=fig_size)
            plt.scatter(
                df["rail nwu pose stamped position x"],
                df["rail nwu pose stamped position y"],
                marker=".",
                label="Turtlebot",
            )
            plt.scatter(
                df["takahe nwu pose stamped position x"],
                df["takahe nwu pose stamped position y"],
                marker=".",
                label="Quadcopter",
            )
            plt.scatter(
                df["desired state x"],
                df["desired state y"],
                alpha=0.05,
                marker="s",
                s=4000,
            )
            plt.plot(
                df["desired state x"],
                df["desired state y"],
                alpha=0.2,
                color="g",
                label="Reference Position",
            )
            plt.plot(occ_pos[0], occ_pos[1], "--r", linewidth=2, label="Occlusion")
            plt.xlabel("X position [m]", fontdict=font)
            plt.ylabel("Y position [m]", fontdict=font)
            plt.title("Field of View Road Network", fontdict=font)
            plt.legend()
            plt.savefig(outdir + "road" + ".png", dpi=dpi)
            plt.show() if is_plot else plt.close()


if __name__ == "__main__":
    main()
