#!/usr/bin/env python3
""" Generate a complete coverage path for a side-scan sonar vessel on an arbitrary convex region.
"""
import math
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.prepared import prep


class PathPlanner:
    """
    The bathydrone path planner.

    Attributes:
        self.bounding: BoundingRegion object to be pathed.
        self.sideScanAngle: angle of the side-scan sonar. (NOT implemented yet)
        self.maxRange: maximum range of the side-scan sonar.
        self.minRange: minimum range of the side-scan sonar.
        self.noRequireParallel: disable parallel pathing requirement.
    """

    def __init__(self, sideScanAngle=30, **system_params):
        self.bounding = None
        self.sideScanAngle = sideScanAngle
        self.maxRange = system_params.get("maxRange", None)
        self.minRange = system_params.get("minRange", None)
        self.maxDepth = system_params.get("maxDepth", None)
        self.noRequireParallel = system_params.get("noRequireParallel", None)

    def get_bounding_polygon(self, **boundingInput):
        """Create a Bounding Polygon Object to Store in the PathPlanner."""
        # TODO add support for depth map
        if "csvName" in boundingInput:
            # fetch from CSV
            setattr(
                self,
                "bounding",
                BoundingRegion(bounding_polygon=None, csvName=boundingInput["csvName"]),
            )
        elif "bounding_polygon" in boundingInput:
            # get from bounding polygon provided directly as an array of tuples.
            setattr(
                self,
                "bounding",
                BoundingRegion(
                    bounding_polygon=boundingInput["bounding_polygon"], csvName=None
                ),
            )

    def generate_path(
        self, polygon_points: List[Tuple[float, float]], path_dist: float, angle: float = 0
    ) -> Tuple[List[List[float]], float, bool, List[Polygon], Polygon]:
        """
        Generate an optimized path through the polygon.

        Args:
            polygon_points (List[Tuple[float, float]]): List of vertices of the polygon.
            path_dist (float): Distance between waypoints.

        Returns:
            Tuple[List[List[float]], float, bool, List[Polygon], Polygon]:
                - chosenPath: List of path points
                - PL: Best path length
                - grid used for path generation
                - Geom: Polygon geometry used for path generation
        """
        # Rotate polygon
        transform = [
            [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
            [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))],
        ]
        rotated_points = [
            list(np.dot(transform, point)) for point in polygon_points
        ]

        # Generate grid and path
        geom = Polygon(rotated_points)
        grid = self.partition(geom, path_dist)

        path, path_length, num_turns = self._generate_path_for_grid(grid)

        # Rotate path back
        inv_transform = [
            [np.cos(np.deg2rad(-angle)), -np.sin(np.deg2rad(-angle))],
            [np.sin(np.deg2rad(-angle)), np.cos(np.deg2rad(-angle))],
        ]
        path = [list(np.dot(inv_transform, point)) for point in path]

        return path, path_length, grid, geom

    def _generate_path_for_grid(self,
        grid: List[Polygon],
    ) -> Tuple[List[List[float]], float, int]:
        """
        Generate a path through the grid cells.

        Args:
            grid (List[Polygon]): List of grid cells.

        Returns:
            Tuple[List[List[float]], float, int]:
                - path: List of path points
                - path_length: Total length of the path
                - num_turns: Number of turns in the path
        """
        path = []
        num_turns = 0
        direction = 1  # 1 = up, -1 = down

        # Get grid boundaries
        x_coords = [cell.bounds[0] for cell in grid]
        min_x = min(x_coords)
        max_x = max(x_coords)
        cell_width = abs(grid[0].bounds[2] - grid[0].bounds[0])

        curr_x = min_x
        while curr_x <= max_x:
            # Get cells in current slice
            slice_cells = [
                cell for cell in grid if abs(cell.bounds[0] - curr_x) < 1e-10
            ]

            if slice_cells:
                num_turns += 1
                path.extend(self._get_slice_path(slice_cells, direction))
                direction *= -1

            curr_x += cell_width

        # Calculate path length
        path_length = self._calculate_path_length(path)

        return path, path_length, num_turns

    @staticmethod
    def _get_slice_path(cells: List[Polygon], direction: int) -> List[List[float]]:
        """
        Generate path points for a vertical slice of cells.

        Args:
            cells (List[Polygon]): List of grid cells in the slice.
            direction (int): Direction of the path (1 = up, -1 = down).

        Returns:
            List[List[float]]: List of path points for the slice.
        """
        if direction == 1:
            start_cell = cells[0]
            end_cell = cells[-1]
        else:
            start_cell = cells[-1]
            end_cell = cells[0]

        return [
            [
                (start_cell.bounds[0] + start_cell.bounds[2]) / 2,
                (start_cell.bounds[1] + start_cell.bounds[3]) / 2,
            ],
            [
                (end_cell.bounds[0] + end_cell.bounds[2]) / 2,
                (end_cell.bounds[1] + end_cell.bounds[3]) / 2,
            ],
        ]

    @staticmethod
    def _calculate_path_length(path: List[List[float]]) -> float:
        """
        Calculate the total length of the path.

        Args:
            path (List[List[float]]): List of path points.

        Returns:
            float: Total length of the path.
        """
        total_length = 0
        for i in range(len(path) - 1):
            dist_x = (path[i][0] - path[i + 1][0]) * 69
            dist_y = (path[i][1] - path[i + 1][1]) * 54.6
            total_length += math.sqrt(dist_x**2 + dist_y**2)
        return total_length

    @staticmethod
    def grid_bounds(geom, delta):
        """
        Input: geom (shapely polygon), delta (distance between path lines)
        Output: grid boundaries from which to draw path.
        """

        minx, miny, maxx, maxy = geom.bounds
        nx = int((maxx - minx) / delta)
        ny = int((maxy - miny) / delta)
        gx, gy = np.linspace(minx, maxx, nx), np.linspace(miny, maxy, ny)

        grid = []
        for i in range(len(gx) - 1):
            for j in range(len(gy) - 1):
                poly_ij = Polygon(
                    [
                        [gx[i], gy[j]],
                        [gx[i], gy[j + 1]],
                        [gx[i + 1], gy[j + 1]],
                        [gx[i + 1], gy[j]],
                    ]
                )
                grid.append(poly_ij)
        return grid

    def partition(self, geom, delta):
        """Define a grid of cells for a polygon."""
        prepared_geom = prep(geom)
        grid = list(filter(prepared_geom.covers, self.grid_bounds(geom, delta)))
        return grid

    @staticmethod
    def listConvexHull(polyPoints):
        """Find the convex hull of a polygon."""
        cHull = ConvexHull(polyPoints)
        listHull = []
        for i in range(len(cHull.vertices)):
            listHull.append(polyPoints[cHull.vertices[i]])
        return listHull

    @staticmethod
    def concavity_checker(polyPoints, polyGeom):
        """Take a polygon, and export an inorder list of SL-concavities"""

        conHullC = polyGeom.convex_hull.get_coordinates()
        conHull = list(conHullC.itertuples(index=False, name=None))

        concavities = [0] * len(polyPoints)
        for i in range(len(polyPoints)):
            if polyPoints[i] in conHull:
                concavities[i] = 0
            else:
                # find bounding hull points
                if i == 0:
                    j = len(polyPoints) - 1
                else:
                    j = i

                while j >= 0:
                    j = j - 1
                    if polyPoints[j] in conHull:
                        b1 = polyPoints[j]
                        break
                j = i
                if i == (len(polyPoints) - 1):
                    j = 0
                while j < len(polyPoints) - 1:
                    j = j + 1
                    if polyPoints[j] in conHull:
                        b2 = polyPoints[j]
                        break
                # Find perpendicular distance from line b1->b2 to point polyPoints[i]
                concavities[i] = abs(
                    (b1[0] - b2[0]) * (b1[1] - polyPoints[i][1])
                    - (b1[1] - b2[1]) * (b1[0] - polyPoints[i][0])
                ) / math.sqrt(math.pow(b1[0] - b2[0], 2) + math.pow(b1[1] - b2[1], 2))
        return concavities

    @staticmethod
    def get_polygon_from_csv_external(csvName):
        """get region of interest from csv (edge detection)"""
        xList = []
        yList = []
        file_path = os.path.join(os.getcwd(), csvName)
        df = pd.read_csv(file_path, usecols=["Latitude", "Longitude"])
        yList = df.Latitude.values.tolist()
        xList = df.Longitude.values.tolist()

        return xList, yList

    @staticmethod
    def plotPath(xList, yList, testPathArr, pathCenters, best_route, best_distance):
        """generate matplotlib plot of the competed path"""
        # PLOTTING SETUP
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)

        # PLOTTING PATHS
        # print(testPathArr)
        ax1.plot(xList, yList, c="red", marker="o", markersize="0.25")
        for i in range(len(testPathArr)):
            x1, y1 = zip(*testPathArr[i])
            ax1.plot(x1, y1, c="green", marker="o", markersize="0.5", zorder=1)

        # PLOTTING VISIT ORDER
        j = 0
        xc2, yc2 = zip(*pathCenters)

        for xIm, yIm in zip(xc2, yc2):
            ax1.text(
                xIm, yIm, str(j), color="grey", fontsize=12, fontweight="bold", zorder=3
            )
            # plt.scatter(xIm, yIm, s=sSize, facecolors='white', edgecolors='grey', zorder=2)
            j += 1

        # SHOW PLOT
        plt.axis("equal")
        plt.show()

class BoundingRegion:
    """
    The bounding polygon for the region of interest.

    Attributes:
        polygonVertices (list of tuples): vertices forming exterior of the bounding polygon.
    """

    def __init__(self, bounding_polygon, csvName):
        if csvName is None:
            assert bounding_polygon is not None, "No polygon provided."
            self.polygonVertices = [tuple(xy_list) for xy_list in bounding_polygon]
        else:
            assert (
                bounding_polygon is None
            ), "Provide either bounding polygon or csv. Set other as None."
            self.polygonVertices = self.get_polygon_from_csv(csvName)

    @staticmethod
    def get_polygon_from_csv(csvName):
        """get region of interest from csv (manual)"""
        file_path = os.path.join(os.getcwd(), "csv", csvName)
        df = pd.read_csv(file_path, usecols=["Latitude", "Longitude"])
        return list(zip(df.Latitude.values.tolist(), df.Longitude.values.tolist()))


def main():
    """
    Example usage of the PathPlanner class.

    1. Create a PathPlanner object.
    2. Set the bounding polygon from csv.
    3. Generate and display a coverage path for the region.
    """
    bounds = [
        (-1.5, -1),
        (0.5, -1),
        (0.5, 2.),
        (-1.5, 2.),
    ]
    my_planner = PathPlanner()
    region = my_planner.get_bounding_polygon(bounding_polygon=bounds)
    best_path, best_path_length, best_grid, best_geom = my_planner.generate_path(bounds, 0.25, angle=0)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # include the first one at the end to close the loop
    bounds.append(bounds[0])
    xList, yList = list(zip(*bounds))
    ax1.plot(xList, yList, c="red", marker="o", markersize="0.25")
    xList, yList = list(zip(*best_path))
    ax1.plot(xList, yList, c="green", marker="o", markersize="0.5", zorder=1)
    plt.axis("equal")
    plt.show()



if __name__ == "__main__":
    main()
