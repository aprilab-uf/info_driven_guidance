#!/usr/bin/env python3

""" This node is used to propagate the particles in the future and publish them
    to the topics future_parts1, future_parts2, future_parts3, future_parts4, future_parts5
    This is done to be able to visualize the particles in the future as a spahetti plot
"""
import numpy as np
import rospy
from mml_guidance.msg import Particle, ParticleMean
from ParticleFilter import ParticleFilter
from std_msgs.msg import Float32MultiArray


class Spaguetti:
    def __init__(self):
        self.init_finished = False

        self.prediction_method = "NN"  # 'NN', 'Velocity' or 'Unicycle'

        # Initialization of robot variables
        self.initial_time = rospy.get_time()

        ## PARTICLE FILTER  ##
        # number of particles
        self.N = 800
        self.filter = ParticleFilter(self.N, self.prediction_method)

        ## INFO-DRIVEN GUIDANCE ##
        # Number of future measurements per sampled particle to consider in EER
        # self.N_m = 1  # not implemented yet
        self.N_s = 25  # Number of sampled particles
        self.K = 5  # Time steps to propagate in the future for EER
        self.sampled_index = np.arange(self.N)
        self.sampled_particles = self.filter.particles[:, : self.N_s, :]
        self.sampled_weights = np.ones(self.N_s) / self.N_s
        self.position_following = True

        # ROS stuff
        rospy.loginfo(
            f"Initializing spaguetti node with parameter in prediction method: {self.prediction_method}"
        )

        self.mag_pg_subscriber = rospy.Subscriber(
            "xyTh_estimate", ParticleMean, self.pf_cb
        )

        self.particle_pub = [
            rospy.Publisher("future_parts1", ParticleMean, queue_size=1),
            rospy.Publisher("future_parts2", ParticleMean, queue_size=1),
            rospy.Publisher("future_parts3", ParticleMean, queue_size=1),
            rospy.Publisher("future_parts4", ParticleMean, queue_size=1),
            rospy.Publisher("future_parts5", ParticleMean, queue_size=1),
        ]
        self.sampled_index_pub = rospy.Publisher(
            "sampled_index", Float32MultiArray, queue_size=1
        )

        rospy.loginfo("Number of particles for the Bayes Filter: %d", self.N)
        rospy.sleep(0.1)
        self.init_finished = True

    def propagate_particles(self, event=None):
        """Propagate the particles and publish them
        Output: published propagated particles"""
        t = rospy.get_time() - self.initial_time
        self.sampled_index = np.random.choice(a=self.N, size=self.N_s)
        self.sampled_particles = np.copy(
            self.filter.particles[:, self.sampled_index, :]
        )

        future_parts = np.copy(self.sampled_particles)

        last_future_time = np.copy(self.filter.last_time)
        for k in range(self.K):
            if self.prediction_method == "NN":
                future_parts = self.filter.predict_mml(future_parts)
            elif self.prediction_method == "Unicycle":
                future_parts, last_future_time = self.filter.predict(
                    future_parts,
                    self.filter.weights[self.sampled_index],
                    last_future_time + 0.3,
                    angular_velocity=self.angular_velocity,
                    linear_velocity=self.linear_velocity,
                )
            elif self.prediction_method == "Velocity":
                future_parts, last_future_time = self.filter.predict(
                    future_parts,
                    self.filter.weights[self.sampled_index],
                    last_future_time + 0.3,
                )
            # publish each k step in the future
            mean_msg = ParticleMean()
            # mean_msg.mean.x = self.filter.weighted_mean[0]
            # mean_msg.mean.y = self.filter.weighted_mean[1]
            # mean_msg.mean.yaw = np.linalg.norm(self.filter.weighted_mean[2:4])
            for ii in range(self.N_s):
                particle_msg = Particle()
                particle_msg.x = future_parts[-1, ii, 0]
                particle_msg.y = future_parts[-1, ii, 1]
                # particle_msg.yaw = np.linalg.norm(future_parts[-1, ii, 2:4])
                particle_msg.weight = self.filter.weights[ii]
                mean_msg.all_particle.append(particle_msg)
            # mean_msg.cov = np.diag(self.filter.var).flatten("C")
            self.particle_pub[k].publish(mean_msg)
        # publish the sampled index array
        self.sampled_index_pub.publish(data=self.sampled_index)

        propagate_time = rospy.get_time() - t - self.initial_time
        # print("Propagation time: ", propagate_time)

    def pf_cb(self, msg):
        """Subscribes to get the particles"""
        # setup the particle array
        particle_list = []
        for data in msg.all_particle:
            particle_list.append([data.x, data.y])
        new_particles = np.array(particle_list)

        self.filter.particles[:-1, :, :] = self.filter.particles[
            1:, :, :
        ]  # shift particles in time
        self.filter.particles[-1, :, :] = new_particles  # update particles

    def shutdown(self, event=None):
        # Stop the node when shutdown is called
        rospy.logfatal("Timer expired or user terminated. Stopping the node...")
        rospy.sleep(0.1)
        rospy.signal_shutdown("Timer signal shutdown")
        # os.system("rosnode kill other_node")


if __name__ == "__main__":
    rospy.init_node("spaguetti", anonymous=True)
    guidance = Spaguetti()

    time_to_shutdown = 2000
    rospy.Timer(rospy.Duration(time_to_shutdown), guidance.shutdown, oneshot=True)
    rospy.on_shutdown(guidance.shutdown)

    # Publish topics
    rospy.Timer(rospy.Duration(1.0 / 3.0), guidance.propagate_particles)

    rospy.spin()
