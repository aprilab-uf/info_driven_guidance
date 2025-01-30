#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np


class LawnmowerPath:
    """Creates lawnmower path for the setpoints ("sp") of a square or rectangle with sp1
    as the first initial point as [x,y] input and sp2 as the second point
    also as [x,y] input."""

    # enter sp1 as [x,y]
    # enter sp2 as [x,y]
    def __init__(self, sp1=[0, 0], sp2=[170, 80], dy=40, is_plot=False, is_sim=False):
        self.sp1 = sp1
        self.sp2 = np.transpose(sp2)
        self.sp = [sp1, sp2]
        self.dy = dy  # number of intervals needed (user input)
        self.deltax = 1
        self.is_plot = is_plot
        self.dr = 6
        self.is_sim = is_sim

    def path(self, s0=np.array([0, 0])):
        """TBased on two initial setpoints, creates the remaining two points
        based on square/rectangle geometry
        Inputs:
           s0: initial position x and y of the robot
        """
        # TODO: make it able to end going either up or down
        self.x_distance = self.sp2[0] - self.sp1[0]
        self.y_distance = self.sp2[1] - self.sp1[1]

        self.sp3 = [self.sp1[0] + self.x_distance, self.sp1[1]]
        self.sp4 = [self.sp1[0], self.sp1[1] + self.y_distance]
        self.square = [self.sp1, self.sp2, self.sp3, self.sp4]

        # Sets up variables for the semi-circular cases for the path
        S = []

        h_arr = np.arange(self.sp1[0], self.sp3[0], self.dy)  # circle case
        i_arr = np.arange(
            (self.sp1[0] + (self.dy / 2)), self.sp3[0], self.dy
        )  # circle case

        r = (self.dy - self.sp1[0]) / 4  # radius of the turn
        k = self.sp4[1] - ((1 / 2) * self.dy)  # top semi-circular case
        l = self.sp1[1] + ((1 / 2) * self.dy)  # bottom semi-circular case

        T = np.arange(
            self.sp1[0], (self.sp3[0] + self.dy / 2), self.dy / 2
        )  # x-component (straight)
        # TODO: automatically calculate the number of points or have it as an input
        delta = 0.2
        S = np.arange(
            (self.sp1[1] + (self.dy / 2)), (k), delta
        )  # y-component (straight)

        # TODO: do not have an overlap of green with red and green with blue points
        # TODO: get rid of all the lists by preallocating memory of the np arrays
        xi = []
        yi = []
        xT = []
        yT = []
        xh = []
        yh = []
        X_list = []
        Y_list = []
        ii = 0

        for count, x in enumerate(T):
            count = int(count)
            if count % 2 == 0:
                for y in S:
                    xT.append(x)
                    yT.append(y)
                    plt.scatter(x, y, c="g") if self.is_plot else None
                xT_arr = np.array(xT)
                yT_arr = np.array(yT)
                xT = []
                yT = []

                x0 = h_arr[ii] - r
                x1 = h_arr[ii] + r
                xhh = np.linspace(x0, x1, self.dr)
                yhh = k + np.sqrt(r**2 - (xhh - h_arr[ii]) ** 2)
                xh.append(xhh + (self.dy / 4))
                yh.append(yhh)
                plt.scatter(xhh + (self.dy / 4), yhh, c="b") if self.is_plot else None
                xh_arr = np.concatenate(xh)
                yh_arr = np.concatenate(yh)
                xh = []
                yh = []

                Xtop = np.concatenate((xT_arr, xh_arr))
                Ytop = np.concatenate((yT_arr, yh_arr))
            else:
                for y in np.flip(S):
                    xT.append(x)
                    yT.append(y)
                    plt.scatter(x, y, c="g") if self.is_plot else None
                xT_arr = np.array(xT)
                yT_arr = np.array(yT)
                xT = []
                yT = []

                if ii < len(i_arr) - 1:
                    x0 = i_arr[ii] - r
                    x1 = i_arr[ii] + r
                    # TODO: evenly spaced angles instead of dx (self.dr)
                    xii = np.linspace(x0, x1, self.dr)
                    yii = l - np.sqrt(r**2 - (xii - i_arr[ii]) ** 2)
                    xi.append(xii + (self.dy / 4))
                    yi.append(yii)
                    plt.scatter(
                        xii + (self.dy / 4), yii, c="red"
                    ) if self.is_plot else None
                    xi_arr = np.concatenate(xi)
                    yi_arr = np.concatenate(yi)
                    xi = []
                    yi = []

                    Xbottom = np.concatenate((xT_arr, xi_arr))
                    Ybottom = np.concatenate((yT_arr, yi_arr))
                else:
                    Xbottom = xT_arr
                    Ybottom = yT_arr

                X = np.concatenate((Xtop, Xbottom))
                Y = np.concatenate((Ytop, Ybottom))
                X_list.append(X)
                Y_list.append(Y)
                ii += 1
                if ii == len(i_arr):
                    break

        self.X = np.concatenate(X_list)
        self.Y = np.concatenate(Y_list)
        self.X = self.X.reshape(self.X.size)
        self.Y = self.Y.reshape(self.Y.size)
        # plt.axis("equal") if self.is_plot else None

        # plt.show() if self.is_plot else None
        # TODO: set axis name with dimensions

        # TODO: CHANGE quick fix offset for hardware
        if self.is_sim:
            path = np.array([self.X + s0[0] - 1.4, self.Y + s0[1] - 1.3])
            # rotation matrix to rotate the path by the specified degrees
            angle = np.deg2rad(0)
        else:
            path = np.array([self.X + s0[0] - 1.2, self.Y + s0[1] - 1.5])
            angle = np.deg2rad(-90)
        tf = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        return np.matmul(tf, path)

    def heading(self):
        """Computes the heading of the path at each index
        based on the difference between the next and previous
        points
        """
        X, Y = (self.X, self.Y)
        h = np.zeros(len(X))
        for i in range(1, len(X) - 1):
            h[i] = np.arctan2(Y[i + 1] - Y[i - 1], X[i + 1] - X[i - 1])
        h[0] = h[1]
        h[-1] = h[-2]
        return h

    def velocity(self):
        """Computes the velocity by getting the
        difference in each axis between the next
        and previous points
        """
        X, Y = (self.X, self.Y)
        d = np.zeros((len(X), 2))
        for i in range(1, len(X) - 1):
            d[i] = np.array([X[i + 1] - X[i - 1], Y[i + 1] - Y[i - 1]])
        d[0] = d[1]
        d[-1] = d[-2]
        # normalize the difference
        d[:, 0] = d[:, 0] * self.max_vel / np.linalg.norm(d, axis=1)
        d[:, 1] = d[:, 1] * self.max_vel / np.linalg.norm(d, axis=1)
        return d

    def trajectory(self, max_vel=1):
        """Returns the trajectory as a nd array of
        position, heading and velocity"""
        self.max_vel = max_vel
        X, Y = self.path()
        h = self.heading()
        d = self.velocity()
        return np.array([X, Y, h, d[:, 0], d[:, 1], np.repeat(0, len(X))]).T

    def plot(self):
        """Plots the path"""
        X, Y = self.path()
        plt.scatter(self.X, self.Y, c="r")
        # Plot rectangle boundary knowing sp1 and sp2
        plt.plot(
            [self.sp1[0], self.sp2[0], self.sp2[0], self.sp1[0], self.sp1[0]],
            [self.sp1[1], self.sp1[1], self.sp2[1], self.sp2[1], self.sp1[1]],
            "--k",
        )
        # TODO: title: " trajectory of the robot"
        plt.axis("equal")
        plt.show()


def main():
    sp1 = [0, 0]
    sp2 = [3.5, 3.5]
    dy = 1.5
    is_plot = True
    planning = LawnmowerPath(sp1, sp2, dy, is_plot)
    traj = planning.trajectory(2)
    planning.plot()
    # print(traj.shape)
    return 0


if __name__ == "__main__":
    main()
