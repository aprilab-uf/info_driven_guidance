import numpy as np
import rospy
import rospkg
import torch

from mml_network.simple_dnn import SimpleDNN
from mml_network.scratch_transformer import ScratchTransformer

FWD_VEL = 0.0
ANG_VEL = 0.0


class ParticleFilter:
    def __init__(self, num_particles=10, prediction_method="NN", is_sim=False):

        # Initialize variables
        deg2rad = lambda deg: np.pi * deg / 180
        self.prediction_method = prediction_method
        # boundary of the lab [[x_min, y_min], [x_max, y_,max]] [m]
        self.AVL_dims = np.array([[-1.7, -1.0], [1.5, 1.6]])
        self.AVL_dims = (
            self.AVL_dims if not is_sim else np.array([[-3.0, -1.2], [1.0, 2.0]])
        )

        if self.prediction_method == "NN" or self.prediction_method == "Transformer":
            self.N_th = 10  # Number of time history particles
            pkg_path = rospkg.RosPack().get_path("mml_guidance")
            self.is_velocity = False
            self.nn_input_size = (
                self.N_th * 2 - 2 if self.is_velocity else self.N_th * 2
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.prediction_method == "NN":
                model_file = pkg_path + "/scripts/mml_network/models/noisy_dnn_best.pth"
                self.motion_model = SimpleDNN(
                    input_size=self.nn_input_size,
                    num_layers=2,
                    nodes_per_layer=80,
                    output_size=2,
                    activation_fn="relu",
                )
            elif self.prediction_method == "Transformer":
                model_file = (
                    pkg_path
                    + "/scripts/mml_network/models/best_noisy_ScratchTransformer.pth"
                )
                self.motion_model = ScratchTransformer(
                    input_dim=2, block_size=10, n_embed=5, n_head=4, n_layer=2
                )

            # load weights
            self.motion_model.load_state_dict(
                torch.load(model_file, map_location=device)
            )
        else:
            self.N_th = 2

        # PF
        self.N = num_particles
        self.weights = np.ones(self.N) / self.N
        self.prev_weights = np.copy(self.weights)
        self.weighted_mean = np.array([0, 0, 0])
        self.is_update = False
        self.is_occlusion = False
        self.neff = self.nEff(self.weights)
        if self.prediction_method == "Velocity":
            self.Nx = 4  # number of states
            self.vmax = 0.7  # m/s
        elif self.prediction_method == "Unicycle":
            self.Nx = 3
            self.measurement_covariance = np.diag([0.01, 0.01, deg2rad(5)])
            self.process_covariance = np.diag([0.01, 0.01, 0.001])
        elif self.prediction_method == "KF":
            self.Nx = 2
            self.measurement_covariance = np.diag([0.01, 0.01])
            self.process_covariance = np.diag([0.01, 0.01, 0.001, 0.001])
        elif self.prediction_method == "NN" or self.prediction_method == "Transformer":
            self.Nx = 2

        if (
            self.prediction_method != "Unicycle" and self.prediction_method != "KF"
        ):  # NN and Velocity
            self.measurement_covariance = np.diag([0.01, 0.01])
            self.process_covariance = np.diag([0.001, 0.001])

        self.noise_inv = np.linalg.inv(self.measurement_covariance[:2, :2])
        self.measurement_history = np.zeros(
            (self.N_th, self.measurement_covariance.shape[0])
        )
        self.particles = self.uniform_sample()
        # Use multivariate normal if you know the initial condition
        # self.particles[-1,:, :2] = np.array([
        #   np.random.multivariate_normal(
        #   np.array([1.3, -1.26]), self.measurement_covariance, self.N
        #   )
        # ])

        # Process noise: q11, q22 is meters of error per meter, q33 is radians of error per revolution
        self.var = np.array(
            [
                self.process_covariance[0, 0],
                self.process_covariance[1, 1],
                self.process_covariance[0, 0],
            ]
        )  # initialization of variance of particles

        # Particles to be resampled whether we have measurements or not (in guidance.py)
        self.resample_index = np.arange(self.N)
        self.initial_time = rospy.get_time()
        self.last_time = 0.0

    def uniform_sample(self) -> np.ndarray:
        """Uniformly samples the particles between min and max values per state.
        Positions are always sampled according to the lab dimensions.
        Output:
            local_particles: N_th sets (time histories) of N particles with Nx states
        """
        if self.prediction_method == "Velocity":
            local_particles = np.random.uniform(
                [self.AVL_dims[0, 0], self.AVL_dims[0, 1], -self.vmax, -self.vmax],
                [self.AVL_dims[1, 0], self.AVL_dims[1, 1], self.vmax, self.vmax],
                (self.N_th, self.N, self.Nx),
            )
        elif self.prediction_method == "Unicycle":
            local_particles = np.random.uniform(
                [self.AVL_dims[0, 0], self.AVL_dims[0, 1], -np.pi],
                [self.AVL_dims[1, 0], self.AVL_dims[1, 1], np.pi],
                (self.N_th, self.N, self.Nx),
            )
        elif (
            self.prediction_method == "NN"
            or self.prediction_method == "KF"
            or self.prediction_method == "Transformer"
        ):
            local_particles = np.random.uniform(
                [self.AVL_dims[0, 0], self.AVL_dims[0, 1]],
                [self.AVL_dims[1, 0], self.AVL_dims[1, 1]],
                (self.N_th, self.N, self.Nx),
            )
        return local_particles

    def pf_loop(
        self,
        noisy_measurement,
        ang_vel=np.array([ANG_VEL]),
        lin_vel=np.array([FWD_VEL]),
    ):
        """Main function of the particle filter where the predict,
        update, resample and estimate steps are called.
        """
        t = rospy.get_time() - self.initial_time

        # update measurement history with noisy_measurement
        self.measurement_history = np.roll(self.measurement_history, -1, axis=0)
        self.measurement_history[-1, :2] = noisy_measurement[:2]

        # Prediction step
        if self.prediction_method == "NN" or self.prediction_method == "Transformer":
            self.particles = self.predict_mml(np.copy(self.particles))
        elif self.prediction_method == "Unicycle":
            self.measurement_history[-1, 2] = noisy_measurement[2]
            self.particles, self.last_time = self.predict(
                self.particles,
                self.last_time,
                angular_velocity=ang_vel,
                linear_velocity=lin_vel,
            )
        elif self.prediction_method == "Velocity":
            dt = t - self.last_time
            estimate_velocity = (
                self.measurement_history[-1, :] - self.measurement_history[-2, :]
            ) * dt

            self.particles, self.last_time = self.predict(
                self.particles,
                self.last_time,
            )

        # Update step
        if self.is_update:
            self.prev_weights = np.copy(self.weights)
            self.weights = self.update(
                self.weights, self.particles, self.measurement_history[-1]
            )

        # Resampling step
        outbounds = self.outside_bounds(self.particles[-1])
        self.neff = self.nEff(self.weights)
        if outbounds > self.N * 0.5:
            # Resample if fraction of particles are outside the lab boundaries
            rospy.logwarn(
                f"{self.outside_bounds(self.particles[-1])} particles outside the lab boundaries. Uniformly resampling."
            )
            self.particles = self.uniform_sample()
        if (
            not self.is_occlusion
        ):  # agent can only tell if it is occluded, not the shape of occlusion
            if (
                self.neff < self.N * 0.99
            ):  # nEff is only infinity when something went wrong
                if (self.neff < self.N * 0.4 or self.neff == self.N) and self.is_update:
                    # most particles are bad, resample from Gaussian around the measurement
                    if self.prediction_method == "Velocity":
                        self.particles[-1, :, :2] = np.random.multivariate_normal(
                            self.measurement_history[-1, :2],
                            self.measurement_covariance,
                            self.N,
                        )
                        # backwards difference for velocities
                        self.particles[-1, :, 2:] = np.random.multivariate_normal(
                            estimate_velocity,
                            dt * self.measurement_covariance,
                            self.N,
                        )
                    else:
                        noise = np.random.multivariate_normal(
                            np.zeros(self.measurement_history.shape[1]),
                            self.measurement_covariance,
                            size=(self.N_th, self.N),
                        )
                        # repeat the measurement history to be the same size as the particles
                        measurement_history_repeated = np.tile(
                            self.measurement_history.reshape(self.N_th, 1, self.Nx),
                            (1, self.N, 1),
                        )
                        self.particles = measurement_history_repeated + noise

                    self.weights = np.ones(self.N) / self.N
                else:
                    # some are good but some are bad, resample
                    self.resample()

        self.weighted_mean, self.var = self.estimate(self.particles[-1], self.weights)
        # print("PF time: ", rospy.get_time() - t - self.initial_time)

    def update(self, weights, particles, noisy_turtle_pose):
        """Updates the belief (weights) of the particle distribution.
        Input: Likelihood of the particles from measurement model and prior belief of the particles
        Output: Updated (posterior) weight of the particles
        """
        weights = weights * self.likelihood(
            particles[-1, :, :], np.tile(noisy_turtle_pose, (self.N, 1))
        )
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
        return weights

    def likelihood(self, particles, y_act):
        """Particles that are closer to the noisy measurements are weighted higher than
        particles which don't match the measurements very well.
        There are two methods to compute this.
        """

        # Method 1: Manual for loop with normal multivariate equation
        shape = particles.shape[0]
        like = np.zeros(shape)
        for ii in range(shape):
            # The factor sqrt(det((2*pi)*measurement_cov)) is not included in the
            # likelihood, but it does not matter since it can be factored
            # and then cancelled out during the normalization or expectation.
            like[ii] = np.exp(
                -0.5
                * (particles[ii, :2] - y_act[ii, :2]).T
                @ self.noise_inv
                @ (particles[ii, :2] - y_act[ii, :2])
            )

        # Method 2: Vectorized using scipy.stats
        # TODO fix to account for different measurements
        # like = stats.multivariate_normal.pdf(
        #    x=particles, mean=y_act[0], cov=self.measurement_covariance
        # )
        return like

    def predict_mml(self, particles):
        """Propagates the current particles through the motion model learned with the
        neural network.
        Input: N_th (number of time histories) sets of particles up to time k-1
        Output: N_th sets of propagated particles up to time k
        """
        # print("particles: ", particles.shape)
        if self.is_velocity:
            current_parts = np.copy(particles[:, :, :2])
            particles = (particles[1:, :, :2] - particles[:-1, :, :2]) / 0.333  # 3 Hz

        next_parts = self.motion_model.predict(
            np.transpose(particles[:, :, :2], (1, 0, 2)).reshape(
                particles.shape[1], self.nn_input_size
            )
        )
        # print("next_parts: ", next_parts)
        if self.is_velocity:
            next_parts = (
                current_parts[-1] + next_parts * 0.333
            )  # x(t+1) = x(t) + v(t) * dt
            particles = current_parts

        particles = np.concatenate(
            (
                particles[1:, :, :2],
                next_parts.reshape(1, next_parts.shape[0], next_parts.shape[1]),
            ),
            axis=0,
        )
        particles[-1, :, :] = self.add_noise(
            particles[-1, :, :], 2 * self.process_covariance
        )
        return particles

    def predict(
        self,
        particles,
        last_time,
        linear_velocity=np.zeros(2),
        angular_velocity=np.zeros(1),
    ):
        """Uses the unicycle model plus noise or the to propagate the belief in the system state.
        The position and orientation of the particles are updated according to the linear
        and angular velocity of the turtlebot.
        In the Velocity method, the particles are propagated according to the velocity of each particle.
        Input: State of the particles at time k-1
        Output: Predicted (propagated) state of the particles up to time k
        """
        t = rospy.get_time() - self.initial_time
        dt = t - last_time

        particles[:-1, :, :] = particles[1:, :, :]  # shift particles in time
        if self.prediction_method == "Unicycle":
            delta_theta = angular_velocity[0] * dt
            particles[-1, :, 2] = (
                particles[-2, :, 2]
                + delta_theta
                + (delta_theta / (2 * np.pi))
                * self.add_noise(
                    np.zeros(particles.shape[1]),
                    self.process_covariance[2, 2],
                    size=particles.shape[1],
                )
            )

            for ii in range(particles.shape[1]):
                if np.abs(particles[-1, ii, 2]) > np.pi:
                    # Wraps angle
                    particles[-1, ii, 2] = (
                        particles[-1, ii, 2] - np.sign(particles[-1, ii, 2]) * 2 * np.pi
                    )

            norm_lin_vel = np.linalg.norm(linear_velocity)
            delta_distance = norm_lin_vel * dt + norm_lin_vel * dt * self.add_noise(
                0, self.process_covariance[0, 0], size=particles.shape[1]
            )
            particles[-1, :, :2] = (
                particles[-2, :, :2]
                + np.array(
                    [
                        delta_distance * np.cos(particles[-1, :, 2]),
                        delta_distance * np.sin(particles[-1, :, 2]),
                    ]
                ).T
            )
        elif self.prediction_method == "Velocity":
            delta_distance = particles[-1, :, 2:] * dt + particles[
                -1, :, 2:
            ] * dt * self.add_noise(
                np.array([0, 0]), self.process_covariance, size=particles.shape[1]
            )
            particles[-1, :, :2] = (
                particles[-2, :, :2] + delta_distance * particles[-1, :, :2]
            )

        last_time = t

        return particles, last_time

    def resample(self):
        """Uses the multinomial resampling algorithm to update the belief in the system state.
        The particles are copied randomly with probability proportional to the weights plus
        some roughening based on the spread of the states.
        Inputs: Updated state of the particles
        Outputs: Resampled updated state of the particles
        """
        N_r = len(self.resample_index)
        if N_r > 0:
            # Normalize weights
            self.weights = (
                self.weights / np.sum(self.weights)
                if np.sum(self.weights) > 0
                else self.weights
            )
            # Copy particles that are weighted higher
            rand_ind = np.random.choice(a=self.N, size=N_r, p=self.weights)
            self.particles[-1, self.resample_index, :] = self.particles[-1, rand_ind, :]
            self.weights[self.resample_index] = self.weights[rand_ind]
            # Roughening. See Bootstrap Filter from Crassidis and Junkins.
            G = 0.2
            E = np.zeros(self.Nx)
            for ii in range(self.Nx):
                E[ii] = np.max(self.particles[-1, :, ii]) - np.min(
                    self.particles[-1, :, ii]
                )
            cov = (G * E * N_r ** (-1 / 3)) ** 2
            P_sigmas = np.diag(cov)

            for ii in self.resample_index:
                self.particles[-1, ii, :] = self.add_noise(
                    self.particles[-1, ii, :], P_sigmas
                )

    def estimate(self, particles, weights):
        """returns mean and variance of the weighted particles"""
        if np.sum(weights) > 0.0:
            weighted_mean = np.average(particles, weights=weights, axis=0)
            # TODO: change in pf_viz to only use 2 covariance
            var = np.zeros_like(self.var)
            var[:2] = np.average(
                (particles[:, :2] - weighted_mean[:2]) ** 2,
                weights=weights,
                axis=0,
            )
            if self.prediction_method == "Unicycle":
                # Component mean in the complex plane to prevent wrong average
                # source: https://www.rosettacode.org/wiki/Averages/Mean_angle#C.2B.2B
                self.yaw_mean = np.arctan2(
                    np.sum(self.weights * np.sin(particles[:, 2])),
                    np.sum(self.weights * np.cos(particles[:, 2])),
                )
                weighted_mean[2] = self.yaw_mean
                var[2] = np.average(
                    (particles[:, 2] - weighted_mean[2]) ** 2,
                    weights=weights,
                    axis=0,
                )
            ## source: Differential Entropy for a Gaussian in Wikipedia
            ## https://en.wikipedia.org/wiki/Differential_entropy
            # H_gauss = (
            #    np.log((2 * np.pi * np.e) ** (3) * np.linalg.det(np.diag(var))) / 2
            # )
            return weighted_mean, var

    def outside_bounds(self, particles):
        """returns the number of particles outside of the lab boundaries"""
        return np.sum(
            (particles[:, 0] < self.AVL_dims[0, 0])
            | (particles[:, 0] > self.AVL_dims[1, 0])
            | (particles[:, 1] < self.AVL_dims[0, 1])
            | (particles[:, 1] > self.AVL_dims[1, 1])
        )

    @staticmethod
    def add_noise(mean, covariance, size=1):
        """Add noise to the mean from a gaussian distribution with covariance matrix"""
        if type(mean) is np.ndarray and type(covariance) is np.ndarray:
            if mean.ndim > 1:
                size = mean.shape[0]
                noise = np.random.multivariate_normal(
                    np.zeros(mean.shape[1]), covariance, size
                )
            else:
                size = mean.shape[0]
                # print('shape of mean: ', mean.shape)
                noise = np.random.multivariate_normal(np.zeros(size), covariance)
        else:
            noise = np.random.normal(0, covariance, size)

        return mean + noise

    @staticmethod
    def nEff(wgts):
        """Compute the number of effective particles
        Source: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
        """
        return 1.0 / np.sum(np.square(wgts))
