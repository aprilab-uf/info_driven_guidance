#!/usr/bin/env python3
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import seaborn as sns
import os

from simple_dnn import SimpleDNN
from transformer_functions import TransAm
from scratch_transformer import ScratchTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, criterion, optimizer, X_train, y_train, epochs=100):
    losses = []
    for epoch in range(epochs):
        outputs = model(X_train)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {loss.item()}")
    return losses


def evaluate(model, criterion, X_test, y_test):
    """Evaluate the model on the test set. Return the loss and the outputs."""
    with torch.no_grad():
        outputs = model.forward(X_test)
        loss = criterion(outputs, y_test)
    return loss.item(), outputs


def parameter_search(
    ModelClass, param_grid, X_train, y_train, X_test, y_test, epochs, weights_filename
):
    grid = ParameterGrid(param_grid)

    best_loss = float("inf")
    best_params = None

    # list of keys that have more than one value in the grid
    changing_keys = [key for key in param_grid.keys() if len(param_grid[key]) > 1]

    for params in grid:
        print("Training with parameters: ", params)

        if "n_head" in list(params.keys()):
            # check that n_embed is larger than n_head and n_embed is divisible by n_head
            if params["n_embed"] < params["n_head"]:
                print("n_embed must be larger than n_head")
                continue

        # initialize the model with the given parameters, make it so that it can take any parameter for any model
        model = ModelClass(**params)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        losses = train(model, criterion, optimizer, X_train, y_train, epochs)
        test_loss, _ = evaluate(model, criterion, X_test, y_test)
        print("Test Loss: ", test_loss, "\n")
        if test_loss < best_loss:
            best_loss = test_loss
            best_params = params
            torch.save(model.state_dict(), weights_filename)

        params = {key: params[key] for key in changing_keys}
        plt.plot(
            losses,
            label=f"{params}",
        )

    print(f"\nBest parameters: {best_params}")
    print(f"Best test loss: {best_loss}")

    plt.legend()
    plt.show()

    return best_params


def plot_NN_output(X_test, y_pred, y_test, is_velocities, df=None):
    # font sizes for the plot labels
    font = {"family": "serif", "weight": "normal", "size": 40}
    sns.set_theme()
    sns.set_style("white")
    sns.set_context("paper", font_scale=3)

    # figure with legends
    fig, ax = plt.subplots()

    occlusions = np.array(
        [[-1.75, -0.75, -1.1, -0.1], [-0.15, 0.85, -0.3, 0.7]]
    )  # [x_min, x_max, y_min, y_max]
    # plot the squares representing occlusions
    for occlusion in occlusions:
        plt.plot(
            [occlusion[0], occlusion[1], occlusion[1], occlusion[0], occlusion[0]],
            [occlusion[2], occlusion[2], occlusion[3], occlusion[3], occlusion[2]],
            color="black",
        )
    # plot an arrow from the last point of X_test to the predicted point and the actual point

    if not is_velocities:
        # Calculate arrow directions for predictions and actual values
        pred_dx = y_pred[:, 0] - X_test[:, -2]
        pred_dy = y_pred[:, 1] - X_test[:, -1]
        actual_dx = y_test[:, 0] - X_test[:, -2]
        actual_dy = y_test[:, 1] - X_test[:, -1]

        # Plot path
        # ax.plot(X_test[:, 0], X_test[:, 1], color="black", linestyle="--", label="Path", alpha=0.5)

        # Plot predicted arrows
        arrow_length = 3  # higher value means shorter arrows
        plt.quiver(
            X_test[:, -2],
            X_test[:, -1],
            pred_dx,
            pred_dy,
            color="red",
            label="Predicted",
            alpha=0.7,
            scale=arrow_length,
        )

        # Plot actual arrows
        plt.quiver(
            X_test[:, -2],
            X_test[:, -1],
            actual_dx,
            actual_dy,
            color="blue",
            label="Actual",
            alpha=0.7,
            scale=arrow_length,
        )

    else:
        # plot the predicted and actual velocities as arrows from the positions defined in df
        _, Xpos_test, _, _ = train_test_split(
            df.iloc[:, :-2].values, df.iloc[:, -2:], test_size=0.2, random_state=42
        )
        arrow_length = 0.1
        for i in range(Xpos_test.shape[0] - 1):
            plt.arrow(
                Xpos_test[i, -2],
                Xpos_test[i, -1],
                y_pred[i, -2] * arrow_length,
                y_pred[i, -1] * arrow_length,
                color="red",
            )
            plt.arrow(
                Xpos_test[i, -2],
                Xpos_test[i, -1],
                y_test[i, 0] * arrow_length,
                y_test[i, 1] * arrow_length,
                color="blue",
            )

    # axis labels
    plt.xlabel("$x_g$ [m]")
    plt.ylabel("$y_g$ [m]")
    # min and max x and y values
    plt.xlim([-2, 1.0])
    plt.ylim([-1.5, 2])
    # equal aspect ratio
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.show()


def select_model(model_name, input_size):
    if model_name == "SimpleDNN":
        model = SimpleDNN(
            input_size=input_size,
            num_layers=2,
            nodes_per_layer=80,
            output_size=2,
            activation_fn="relu",
        )
    elif model_name == "TransAm":
        print("input size: ", input_size)
        model = TransAm(in_dim=2, n_embed=10, num_layers=1, n_head=1, dropout=0.01)
    elif model_name == "ScratchTransformer":
        model = ScratchTransformer(
            input_dim=2, block_size=10, n_embed=5, n_head=4, n_layer=2
        )
    else:
        raise ValueError("Invalid model name")

    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
    model = model.to(device)

    return model


def select_parameters(model_name, input_size):
    if model_name == "SimpleDNN":
        param_grid = {
            "input_size": [input_size],
            "output_size": [2],
            "activation_fn": ["relu"],
            "num_layers": [2, 4, 8],
            "nodes_per_layer": [20, 40, 80],
        }
    elif model_name == "ScratchTransformer":
        param_grid = {
            "n_embed": [2, 5, 10],
            "n_head": [2, 4],
            "n_layer": [2, 3],
        }
    elif model_name == "TransAm":
        param_grid = {
            "n_embed": [2, 5, 10],
            "num_layers": [1, 2],
            "n_head": [1, 2],
        }
    else:
        raise ValueError("Invalid model name")

    return param_grid


def main():
    is_velocities = False
    is_parameter_search = False
    model_name = (
        "ScratchTransformer"  # "TransAm"  # "ScratchTransformer"  # "SimpleDNN"
    )
    prefix_name = "noisy_"

    path = os.path.expanduser("~/mml_ws/src/mml_guidance/sim_data/training_data/")
    # print the files in the directory
    if is_velocities:
        df_vel = pd.read_csv(path + "converted_training_data.csv")
    df = pd.read_csv(path + "converted_" + prefix_name + "training_data.csv")
    X = df.iloc[:, :-2].values if not is_velocities else df_vel.iloc[:, :-2].values
    y = df.iloc[:, -2:].values if not is_velocities else df_vel.iloc[:, -2:].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = torch.from_numpy(X_train.astype(np.float32)).to(
        device
    )  # (N, T*C) = (N, 10*2)
    X_test = torch.from_numpy(X_test.astype(np.float32)).to(device)
    y_train = torch.from_numpy(y_train.astype(np.float32)).to(device)
    y_test = torch.from_numpy(y_test.astype(np.float32)).to(device)

    print("X_train shape: ", X_train.shape)

    criterion = nn.MSELoss()

    model = select_model(model_name=model_name, input_size=X_train.shape[1])

    if is_parameter_search:
        param_grid = select_parameters(
            model_name=model_name, input_size=X_train.shape[1]
        )

        ModelClass = model.__class__
        weights_filename = (
            path
            + "/../../scripts/mml_network/models/best_"
            + prefix_name
            + model_name
            + ".pth"
        )

        epochs = 200
        best_params = parameter_search(
            ModelClass,
            param_grid,
            X_train,
            y_train,
            X_test,
            y_test,
            epochs,
            weights_filename=weights_filename,
        )
        model = ModelClass(**best_params)
        model.load_state_dict(torch.load(weights_filename))
    else:
        optimizer = optim.Adam(model.parameters(), lr=4e-3)
        losslist = train(model, criterion, optimizer, X_train, y_train, epochs=200)

        # plot the loss
        plt.plot(losslist)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

        # torch.save(model.state_dict(), "transformer.pth")

    model.eval()
    test_loss, y_pred = evaluate(model, criterion, X_test, y_test)
    print("Test Loss: ", test_loss)

    X_test = X_test.to("cpu").detach().numpy()
    y_pred = y_pred.to("cpu").detach().numpy()
    y_test = y_test.to("cpu").detach().numpy()
    plot_NN_output(X_test, y_pred, y_test, is_velocities, df)

    return 0


if __name__ == "__main__":
    main()
