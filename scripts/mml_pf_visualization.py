#!/usr/bin/env python

import geometry_msgs.msg
from nav_msgs.msg import Odometry
from mml_guidance.msg import Particle, ParticleMean, ParticleArray
import rospy
from std_msgs.msg import Bool, Float32, Float32MultiArray
from geometry_msgs.msg import PoseStamped, PointStamped
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image
import numpy as np
import math
import matplotlib.gridspec as gridspec
from tf.transformations import euler_from_quaternion

# from sensor_msgs.msg import Joy


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


# extend the display of the visualization in some part of the screen
def move_figure(f, x, y):

    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == "TkAgg":
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == "WXAgg":
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK

        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)
    f.set_size_inches(19, 24.0)


class MML_PF_Visualization:
    def __init__(self):
        self.initialization_finished = False

        self.is_sim = rospy.get_param("/is_sim", True)

        # self.img_cam = mpimg.imread(visualization_path + "/resources/cam.png")

        # number of real time data plotted in matplotlib
        self.data_size = 200
        self.update_size = 6

        # as the spin function is call by an infinite loop in the file _pf_visualization_node, you can define the rate of this loop with and then add
        self.loop_rate = rospy.Rate(100)

        # generation an interactive matplotlib visualization
        plt.ion()  ## Note this correction

        # message container
        self.temp_msg = None
        self.mml_pf_subscriber = rospy.Subscriber(
            "xyTh_estimate", ParticleMean, self.mml_pf_callback
        )
        self.sampled_index_subscriber = rospy.Subscriber(
            "sampled_index", Float32MultiArray, self.si_cb, queue_size=100
        )
        self.mml_pred_pf_subscriber = rospy.Subscriber(
            "xyTh_predictions", ParticleArray, self.mml_pred_pf_callback
        )
        self.mml_errEstimate_subscriber = rospy.Subscriber(
            "err_estimation", geometry_msgs.msg.PointStamped, self.mml_err_callback
        )
        self.mml_entropy_subscriber = rospy.Subscriber(
            "entropy", Float32, self.entropy_callback
        )
        if self.is_sim:
            self.true_position_subscriber = rospy.Subscriber(
                "odom", Odometry, self.odom_callback
            )
            self.mocap_msg = Odometry()
        else:
            self.true_position_subscriber = rospy.Subscriber(
                "odom", PoseStamped, self.odom_callback
            )
            self.mocap_msg = PoseStamped()
        # self.joy_subscriber = rospy.Subscriber("joy", Joy, self.joy_callback,queue_size=100)
        self.noisy_measurements_subscriber = rospy.Subscriber(
            "/noisy_measurement", PointStamped, self.measurement_callback
        )
        self.update_subscriber = rospy.Subscriber(
            "is_update", Bool, self.upd_callback, queue_size=100
        )
        self.fov_subscriber = rospy.Subscriber(
            "fov_coord", Float32MultiArray, self.fov_callback, queue_size=100
        )
        self.des_fov_subscriber = rospy.Subscriber(
            "des_fov_coord", Float32MultiArray, self.des_fov_callback, queue_size=100
        )

        # initializa matlplotlib figure ui
        fig = plt.figure()
        fig2 = plt.figure()
        # define 6 plot holder
        gs = gridspec.GridSpec(ncols=3, nrows=3)

        # move the figure to the top left part of the screen
        move_figure(fig, 0, 0)

        # initialize subplot in the matplotlib figure
        self.fig_ax1 = fig.add_subplot(gs[:, :])
        self.fig_ax1.set_title("Map")
        self.fig_ax2 = fig2.add_subplot(gs[0, 1])
        self.fig_ax2.set_title("Est x vs Real x")
        self.fig_ax3 = fig2.add_subplot(gs[1, 1])
        self.fig_ax3.set_title("Est y vs Real y")
        self.fig_ax4 = fig2.add_subplot(gs[2, 1])
        self.fig_ax4.set_title("Est y vs Real y")
        self.fig_ax5 = fig2.add_subplot(gs[:, 2])
        self.fig_ax5.set_title("Entropy")

        # set the map image boundaries and resize it to respect aspect ratio
        self.fig_ax1.axis("equal")
        self.fig_ax1.set(xlim=(-5.5, 5.5), ylim=(-7, 7))

        # initialize data structure
        # estimate error lists to plot respective to timestamp
        self.xErrList = []
        self.yErrList = []
        self.yawErrList = []
        self.timestamplist = []

        # entropty list to plot respective to timestamp
        self.entropyList = []

        # real position of the robot using mocap posestamped and its timestamp
        self.xReallist = []
        self.yReallist = []
        self.yawReallist = []
        self.timestampReallist = []
        # self.joy_msg = Joy()
        self.update_msg = Bool()
        self.measurement_msg = None
        self.update_msg.data = True
        self.update_t = np.zeros(self.update_size)
        self.goal_pose = np.array([0, 0])
        self.fov = np.zeros((5, 2))
        self.des_fov = np.zeros((5, 2))
        self.particles = np.zeros((1, 2))
        self.K = rospy.get_param("/predict_window", 4)
        self.N_s = rospy.get_param("/num_sampled_particles", 25)
        self.future_parts = np.zeros((self.K, self.N_s, 2))
        self.plot_prediction = False
        self.sampled_index = np.arange(self.N_s)

        self.x_sigmaList = []
        self.y_sigmaList = []
        self.yaw_sigmaList = []

        # Road Network
        self.road_network = rospy.get_param("/road_network", None)
        if self.road_network == None:
            self.road_network = np.array(
                [
                    [-1.25, 1.0],
                    [0.35, 1],
                    [-0.45, 1.8],
                    [-1.25, 1],
                    [-1.25, -0.6],
                    [0.35, 1.0],
                    [0.35, -0.6],
                    [-1.25, -0.6],
                ]
            )

        # initialization flags
        self.plot_flag = False
        self.initialization_finished = True

    def joy_callback(self, msg):

        if (
            msg.buttons[0] == 1
            and abs((msg.header.stamp - self.joy_msg.header.stamp).to_sec()) > 0.1
        ):
            rospy.loginfo("Received an update in the visualization")
            self.joy_msg = msg

    def upd_callback(self, msg):
        if self.initialization_finished == True:
            if msg.data:
                t = rospy.get_time()
                # if(t - self.update_t > 0.3):
                # if(msg.buttons[0]  == 1 and abs((msg.header.stamp - self.joy_msg.header.stamp).to_sec()) > 0.1 ):
                #    print("on")
                #    self.update_msg = msg
                self.update_t[:-1] = self.update_t[1:]
                self.update_t[-1] = t
            self.update_msg = msg

    def fov_callback(self, msg):
        if self.initialization_finished == True:
            self.fov = msg.data
            self.fov = np.reshape(self.fov, (5, 2))

    def des_fov_callback(self, msg):
        if self.initialization_finished == True:
            self.des_fov = msg.data
            self.des_fov = np.reshape(self.des_fov, (5, 2))

    def mml_pf_callback(self, msg):
        if self.initialization_finished == True:
            self.particle_mean = np.array([msg.mean.x, msg.mean.y])
            self.cov = np.array(msg.cov)

            # setup the particle array
            particle_list = []
            for data in msg.all_particle:
                particle_list.append([data.x, data.y])
            self.particles = np.array(particle_list)

            # compute sigma values
            x_sigma = 3 * math.sqrt(msg.cov[0])
            y_sigma = 3 * math.sqrt(msg.cov[4])
            yaw_sigma = 3 * math.sqrt(msg.cov[8])
            self.x_sigmaList.append(x_sigma)
            self.y_sigmaList.append(y_sigma)
            self.yaw_sigmaList.append(yaw_sigma)

            self.plot_flag = True

    def mml_pred_pf_callback(self, msg):
        if self.initialization_finished == True:
            self.plot_prediction = True
            for k in range(self.K):
                particle_list = []
                for data in msg.particle_array[k].all_particle:
                    particle_list.append([data.x, data.y])
                particles = np.array(particle_list)
                self.future_parts[k, :, :] = particles

            self.plot_flag = True

    def si_cb(self, msg):
        if self.initialization_finished == True:
            self.sampled_index = np.array(msg.data).astype(int)

    def mml_err_callback(self, msg):
        if self.initialization_finished == True:
            # estimate error holder update
            self.xErrList.append(msg.point.x)
            self.yErrList.append(msg.point.y)

            # change yaw boundaries to -pi to pi
            errYaw = msg.point.z % (2 * math.pi)
            if errYaw > math.pi:
                errYaw = errYaw - 2 * math.pi
            if errYaw < -math.pi:
                errYaw = errYaw + 2 * math.pi
            # error on yaw update
            self.yawErrList.append(errYaw)
            # according timestamp update
            self.timestamplist.append(rospy.Time.now().to_sec())
            self.plot_flag = True

    def entropy_callback(self, msg):
        if self.initialization_finished == True:
            self.entropyList.append(msg.data)
            self.plot_flag = True

    def odom_callback(self, msg):
        if self.initialization_finished == True:
            # mocap holder update
            self.mocap_msg = msg

    def measurement_callback(self, msg):
        if self.initialization_finished == True:
            # mocap holder update
            self.measurement_msg = msg

    def getCircle(self):
        """generate a python list representing a normalized circle"""
        theta = np.linspace(0, 2 * np.pi, 100)
        circle = np.block([[np.cos(theta)], [np.sin(theta)]])

        return circle

    def plotCov(self, ax, avg, cov):
        """plot the covariance ellipse on top of the particles"""
        # compute eigen values
        y, v = np.linalg.eig(cov)
        # take the real part of the eigen value
        y, v = np.real(y), np.real(v)
        # get all the 3 sigmas values
        r = np.sqrt(7.814 * np.abs(y))  # 5.991 for 95% confidence. 7.814 for 3dof

        # generate a normalized circle
        circle = self.getCircle()
        # compute the ellipse shape
        ellipse = np.matmul(v, (r[:, None] * circle)) + avg[:2, None]
        # pot the ellipse
        self.fig_ax1.plot(ellipse[0], ellipse[1], color="g")

    # def adding_attrited_value_to_plot(self):

    def visualization_spin(self):

        if self.initialization_finished:

            if self.plot_flag:
                # clear and reinitialize all the plots
                self.fig_ax1.clear()
                self.fig_ax2.clear()
                self.fig_ax3.clear()
                self.fig_ax4.clear()
                self.fig_ax5.clear()
                self.fig_ax1.set_title("Map")
                self.fig_ax2.set_title("Est x vs Real x")
                self.fig_ax3.set_title("Est y vs Real y")
                self.fig_ax4.set_title("Est yaw vs Real yaw")
                self.fig_ax5.set_title("Entropy")
                self.fig_ax1.axis("equal")
                self.fig_ax1.set(xlim=(-5.5, 5.5), ylim=(-7, 7))

                # plot the road network
                self.fig_ax1.plot(
                    self.road_network[:, 0],
                    self.road_network[:, 1],
                    marker=".",
                    markersize=5,
                    alpha=0.6,
                    color="k",
                    label="Road Network",
                )

                # plot the particles
                # TODO: change from only sampled to all particles
                # check the particle is not outside of the map, if it is, remove that sampled index
                self.sampled_index = np.array(
                    [
                        i
                        for i in self.sampled_index
                        if self.particles[i, 0] < 5.5
                        and self.particles[i, 0] > -5.5
                        and self.particles[i, 1] < 7
                        and self.particles[i, 1] > -7
                    ]
                )
                self.fig_ax1.scatter(
                    self.particles[self.sampled_index, 0],
                    self.particles[self.sampled_index, 1],
                    marker=".",
                    color="k",
                    label="Sampled !!! Particles ",
                )
                # plot the current estimate position
                self.fig_ax1.plot(
                    self.particle_mean[0],
                    self.particle_mean[1],
                    marker="P",
                    markersize=10.0,
                    color="g",
                    label="Estimated position ",
                )

                # plot spaguetti plots and scatter using the future particles and the sampled index
                # TODO: change from blue to black
                self.fig_ax1.axis("equal")
                self.fig_ax1.set(xlim=(-5.5, 5.5), ylim=(-7, 7))
                if self.plot_prediction:
                    slope = (0.5 - 0.05) / (self.K - 1)
                    for k in range(self.K):
                        self.fig_ax1.scatter(
                            self.future_parts[k, :, 0],
                            self.future_parts[k, :, 1],
                            marker=".",
                            color="b",
                            alpha=0.5 - slope * k,
                        )
                        counter = 0
                        for ii in self.sampled_index:
                            if k == 0:
                                self.fig_ax1.plot(
                                    [
                                        self.particles[ii, 0],
                                        self.future_parts[0, counter, 0],
                                    ],
                                    [
                                        self.particles[ii, 1],
                                        self.future_parts[k, counter, 1],
                                    ],
                                    color="b",
                                    alpha=0.2,
                                )
                            else:
                                self.fig_ax1.plot(
                                    [
                                        self.future_parts[k - 1, counter, 0],
                                        self.future_parts[k, counter, 0],
                                    ],
                                    [
                                        self.future_parts[k - 1, counter, 1],
                                        self.future_parts[k, counter, 1],
                                    ],
                                    color="b",
                                    alpha=0.2,
                                )
                            counter += 1
                            self.fig_ax1.axis("equal")
                            self.fig_ax1.set(xlim=(-5.5, 5.5), ylim=(-7, 7))
                    self.fig_ax1.scatter(
                        self.fov[0, 0],
                        self.fov[0, 1],
                        marker=".",
                        color="b",
                        alpha=0.3,
                        label="Propagated Particles",
                    )  # fake for legend

                self.fig_ax1.axis("equal")
                self.fig_ax1.set(xlim=(-5.5, 5.5), ylim=(-7, 7))
                # plot the real position
                if self.is_sim:
                    self.fig_ax1.plot(
                        self.mocap_msg.pose.pose.position.x,
                        self.mocap_msg.pose.pose.position.y,
                        marker="P",
                        markersize=10.0,
                        color="m",
                        label="True position ",
                    )
                else:
                    self.fig_ax1.plot(
                        self.mocap_msg.pose.position.x,
                        self.mocap_msg.pose.position.y,
                        marker="P",
                        markersize=10.0,
                        color="m",
                        label="True position ",
                    )
                # plot the desired fov
                self.fig_ax1.plot(
                    self.des_fov[:, 0],
                    self.des_fov[:, 1],
                    marker=".",
                    markersize=1.0,
                    color="b",
                    label="Action FOV",
                )
                act_x = (self.des_fov[2, 0] - self.des_fov[0, 0]) / 2.0 + self.des_fov[
                    0, 0
                ]
                act_y = (self.des_fov[1, 1] - self.des_fov[0, 1]) / 2.0 + self.des_fov[
                    0, 1
                ]
                self.fig_ax1.scatter(
                    act_x, act_y, marker="+", color="b", label="Action chosen"
                )
                # plot the fov
                self.fig_ax1.plot(
                    self.fov[:, 0],
                    self.fov[:, 1],
                    marker=".",
                    markersize=1.0,
                    color="r",
                    label="Field of View ",
                )
                quad_x = (self.fov[2, 0] - self.fov[0, 0]) / 2.0 + self.fov[0, 0]
                quad_y = (self.fov[1, 1] - self.fov[0, 1]) / 2.0 + self.fov[0, 1]
                self.fig_ax1.scatter(
                    quad_x, quad_y, marker="+", color="r", label="Quad position "
                )
                # plot the occlusion
                occlusions = rospy.get_param("/occlusions", None)
                if occlusions != None:
                    occ_centers = occlusions[0]
                    widths = occlusions[1]
                    for occ_center, width in zip(occ_centers, widths):
                        x_dim = np.array([-width, width, width, -width, -width])
                        y_dim = np.array([-width, -width, width, width, -width])
                        self.fig_ax1.plot(
                            occ_center[0] + x_dim / 2,
                            occ_center[1] + y_dim / 2,
                            marker=".",
                            markersize=0.5,
                            color="k",
                        )
                    self.fig_ax1.plot(
                        occ_center[0] + x_dim / 2,
                        occ_center[1] + y_dim / 2,
                        marker=".",
                        markersize=0.5,
                        color="k",
                        label="Occlusion ",
                    )
                # self.fig_ax1.plot([], [], marker=">",
                #                   markersize=10., color="black", label="Measurement")
                # add legend
                self.fig_ax1.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))

                # shows the camera picture to indicate measurement
                # if self.update_msg.data:
                #    self.fig_ax1.imshow(self.img_cam, extent=[2.5, 4.5, -3.4, -1.8])

                # plot estimated arrow theta arrow on the map
                # if the update happen 0.25 sec before then use a yellow arrow
                # plot real arrow theta arrow on the map
                if self.is_sim:
                    real_theta = euler_from_quaternion(
                        self.mocap_msg.pose.pose.orientation.x,
                        self.mocap_msg.pose.pose.orientation.y,
                        self.mocap_msg.pose.pose.orientation.z,
                        self.mocap_msg.pose.pose.orientation.w,
                    )[2]
                    self.fig_ax1.arrow(
                        self.mocap_msg.pose.pose.position.x,
                        self.mocap_msg.pose.pose.position.y,
                        0.25 * math.cos(real_theta),
                        0.25 * math.sin(real_theta),
                        width=0.05,
                        color="magenta",
                        label="Estimated yaw",
                    )
                else:
                    real_theta = euler_from_quaternion(
                        self.mocap_msg.pose.orientation.x,
                        self.mocap_msg.pose.orientation.y,
                        self.mocap_msg.pose.orientation.z,
                        self.mocap_msg.pose.orientation.w,
                    )[2]
                    self.fig_ax1.arrow(
                        self.mocap_msg.pose.position.x,
                        self.mocap_msg.pose.position.y,
                        0.25 * math.cos(real_theta),
                        0.25 * math.sin(real_theta),
                        width=0.05,
                        color="magenta",
                        label="Estimated yaw",
                    )
                # plot the measurement
                if self.measurement_msg != None:
                    self.fig_ax1.plot(
                        self.measurement_msg.point.x,
                        self.measurement_msg.point.y,
                        marker="+",
                        markersize=10.0,
                        color="orange",
                        label="Noisy Measurements",
                    )

                # plot particles covariances
                if self.temp_msg != None:
                    self.plotCov(
                        self.fig_ax1,
                        self.particle_mean,
                        self.cov.reshape((3, 3))[:2, :2],
                    )
                    self.fig_ax1.grid(True)

                # trim extra data to keep only the most updated data plotted
                if len(self.timestamplist) > self.data_size:
                    self.xErrList = self.xErrList[-self.data_size :]
                    self.yErrList = self.yErrList[-self.data_size :]
                    self.yawErrList = self.yawErrList[-self.data_size :]
                    self.timestamplist = self.timestamplist[-self.data_size :]
                    self.x_sigmaList = self.x_sigmaList[-self.data_size :]
                    self.y_sigmaList = self.y_sigmaList[-self.data_size :]
                    self.yaw_sigmaList = self.yaw_sigmaList[-self.data_size :]
                    self.entropyList = self.entropyList[-self.data_size :]

                timestamplist = np.copy(self.timestamplist)[0 : self.data_size]
                xErrList = np.copy(self.xErrList)[0 : self.data_size]
                yErrList = np.copy(self.yErrList)[0 : self.data_size]
                yawErrList = np.copy(self.yawErrList)[0 : self.data_size]
                x_sigmaList = np.copy(self.x_sigmaList)[0 : self.data_size]
                y_sigmaList = np.copy(self.y_sigmaList)[0 : self.data_size]
                yaw_sigmaList = np.copy(self.yaw_sigmaList)[0 : self.data_size]
                entropyList = np.copy(self.entropyList)[0 : self.data_size]

                # plot the estimates errors and their sigmas
                if len(timestamplist) == len(xErrList) and len(x_sigmaList) == len(
                    timestamplist
                ):
                    self.fig_ax2.set_ylim(-1, 1)
                    self.fig_ax2.plot(timestamplist, xErrList)
                    self.fig_ax2.plot(timestamplist, (x_sigmaList), color="r")
                    self.fig_ax2.plot(timestamplist, (-x_sigmaList), color="r")
                    self.fig_ax2.grid(True)

                if len(timestamplist) == len(yErrList) and len(y_sigmaList) == len(
                    timestamplist
                ):
                    self.fig_ax3.set_ylim(-1, 1)
                    self.fig_ax3.plot(timestamplist, yErrList)
                    self.fig_ax3.plot(timestamplist, (y_sigmaList), color="r")
                    self.fig_ax3.plot(timestamplist, (-y_sigmaList), color="r")
                if len(self.timestamplist) == len(self.yawErrList) and len(
                    yaw_sigmaList
                ) == len(timestamplist):
                    self.fig_ax4.set_ylim(-0.4, 0.4)
                    self.fig_ax4.plot(timestamplist, yawErrList)
                    self.fig_ax4.plot(timestamplist, (yaw_sigmaList), color="r")
                    self.fig_ax4.plot(timestamplist, (-yaw_sigmaList), color="r")

                # if len(self.timestamplist) == len(self.entropyList) and len(
                #    entropyList
                # ) == len(timestamplist):
                # self.fig_ax5.set_ylim(0,1)
                self.fig_ax5.plot(timestamplist, entropyList[: len(timestamplist)])

                self.plot_flag = False

                # orange bars to show update times
                # for fig in [self.fig_ax2, self.fig_ax3]:
                #    for t in self.update_t:
                #        fig.plot([t,t], [-10,10], color="orange")

            # update all the plots and display them
            plt.show()

            plt.pause(0.01)

            self.loop_rate.sleep()


if __name__ == "__main__":
    rospy.init_node("mml_pf_visualization", anonymous=False)
    mml_pf_visualization = MML_PF_Visualization()

    while not rospy.is_shutdown():
        mml_pf_visualization.visualization_spin()
