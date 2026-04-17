# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
import scipy.linalg


class KalmanFilterXYAH:
    """
    For bytetrack. A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space (x, y, a, h, vx, vy, va, vh) contains the bounding box center position (x, y), aspect
    ratio a, height h, and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location (x, y, a, h) is taken as direct
    observation of the state space (linear observation model).
    """

    def __init__(self):
        """Initialize Kalman filter model matrices with motion and observation uncertainty weights."""
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current state estimate. These weights control
        # the amount of uncertainty in the model.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple:
        """
        Create track from unassociated measurement.

        Args:
            measurement (ndarray): Bounding box coordinates (x, y, a, h) with center position (x, y), aspect ratio a,
                and height h.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector (8 dimensional) and covariance matrix (8x8 dimensional)
                of the new track. Unobserved velocities are initialized to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        Run Kalman filter prediction step.

        Args:
            mean (ndarray): The 8 dimensional mean vector of the object state at the previous time step.
            covariance (ndarray): The 8x8 dimensional covariance matrix of the object state at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector and covariance matrix of the predicted state. Unobserved
                velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        Project state distribution to measurement space.

        Args:
            mean (ndarray): The state's mean vector (8 dimensional array).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            (tuple[ndarray, ndarray]): Returns the projected mean and covariance matrix of the given state estimate.
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        Run Kalman filter prediction step (Vectorized version).

        Args:
            mean (ndarray): The Nx8 dimensional mean matrix of the object states at the previous time step.
            covariance (ndarray): The Nx8x8 covariance matrix of the object states at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector and covariance matrix of the predicted state. Unobserved
                velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> tuple:
        """
        Run Kalman filter correction step.

        Args:
            mean (ndarray): The predicted state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (ndarray): The 4 dimensional measurement vector (x, y, a, h), where (x, y) is the center
                position, a the aspect ratio, and h the height of the bounding box.

        Returns:
            (tuple[ndarray, ndarray]): Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        """
        Compute gating distance between state distribution and measurements. A suitable distance threshold can be
        obtained from `chi2inv95`. If `only_position` is False, the chi-square distribution has 4 degrees of freedom,
        otherwise 2.

        Args:
            mean (ndarray): Mean vector over the state distribution (8 dimensional).
            covariance (ndarray): Covariance of the state distribution (8x8 dimensional).
            measurements (ndarray): An Nx4 matrix of N measurements, each in format (x, y, a, h) where (x, y)
                is the bounding box center position, a the aspect ratio, and h the height.
            only_position (bool, optional): If True, distance computation is done with respect to the bounding box
                center position only. Defaults to False.
            metric (str, optional): The metric to use for calculating the distance. Options are 'gaussian' for the
                squared Euclidean distance and 'maha' for the squared Mahalanobis distance. Defaults to 'maha'.

        Returns:
            (np.ndarray): Returns an array of length N, where the i-th element contains the squared distance between
                (mean, covariance) and `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            return np.sum(z * z, axis=0)  # square maha
        else:
            raise ValueError("Invalid distance metric")

class KalmanFilterXYWH(KalmanFilterXYAH):
    """
    For BoT-SORT. A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space (x, y, w, h, vx, vy, vw, vh) contains the bounding box center position (x, y), width
    w, height h, and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location (x, y, w, h) is taken as direct
    observation of the state space (linear observation model).
    """

    def initiate(self, measurement: np.ndarray) -> tuple:
        """
        Create track from unassociated measurement.

        Args:
            measurement (ndarray): Bounding box coordinates (x, y, w, h) with center position (x, y), width, and height.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector (8 dimensional) and covariance matrix (8x8 dimensional)
                of the new track. Unobserved velocities are initialized to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance) -> tuple:
        """
        Run Kalman filter prediction step.

        Args:
            mean (ndarray): The 8 dimensional mean vector of the object state at the previous time step.
            covariance (ndarray): The 8x8 dimensional covariance matrix of the object state at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector and covariance matrix of the predicted state. Unobserved
                velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance) -> tuple:
        """
        Project state distribution to measurement space.

        Args:
            mean (ndarray): The state's mean vector (8 dimensional array).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            (tuple[ndarray, ndarray]): Returns the projected mean and covariance matrix of the given state estimate.
        """
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance) -> tuple:
        """
        Run Kalman filter prediction step (Vectorized version).

        Args:
            mean (ndarray): The Nx8 dimensional mean matrix of the object states at the previous time step.
            covariance (ndarray): The Nx8x8 covariance matrix of the object states at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector and covariance matrix of the predicted state. Unobserved
                velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement) -> tuple:
        """
        Run Kalman filter correction step.

        Args:
            mean (ndarray): The predicted state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (ndarray): The 4 dimensional measurement vector (x, y, w, h), where (x, y) is the center
                position, w the width, and h the height of the bounding box.

        Returns:
            (tuple[ndarray, ndarray]): Returns the measurement-corrected state distribution.
        """
        return super().update(mean, covariance, measurement)

class KalmanFilterDynamicNoise(KalmanFilterXYWH):
    # Q、 R、 QR
    def __init__(self, mode: str = "QR"):
        super().__init__()
        self._std_weight_position_dynamic = 1.0 / 15
        self._std_weight_velocity_dynamic = 1.0 / 120
        self.mode = mode

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        if self.mode in ["QR", "Q"]:
            std = [
                2 * self._std_weight_position_dynamic * measurement[2],
                2 * self._std_weight_position_dynamic * measurement[3],
                2 * self._std_weight_position_dynamic * measurement[2],
                2 * self._std_weight_position_dynamic * measurement[3],
                10 * self._std_weight_velocity_dynamic * measurement[2],
                10 * self._std_weight_velocity_dynamic * measurement[3],
                10 * self._std_weight_velocity_dynamic * measurement[2],
                10 * self._std_weight_velocity_dynamic * measurement[3],
            ]
        else:
            std = np.ones(8) * 1.0

        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        if self.mode in ["QR", "Q"]:
            std_pos = [
                self._std_weight_position_dynamic * mean[2],
                self._std_weight_position_dynamic * mean[3],
                self._std_weight_position_dynamic * mean[2],
                self._std_weight_position_dynamic * mean[3],
            ]
            std_vel = [
                self._std_weight_velocity_dynamic * mean[2],
                self._std_weight_velocity_dynamic * mean[3],
                self._std_weight_velocity_dynamic * mean[2],
                self._std_weight_velocity_dynamic * mean[3],
            ]
            motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        else:
            motion_cov = np.eye(8) * 0.01

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        mean_proj, cov_proj = super().project(mean, covariance)
        if self.mode in ["QR", "R"]:
            std = [
                self._std_weight_position_dynamic * mean[2],
                self._std_weight_position_dynamic * mean[3],
                self._std_weight_position_dynamic * mean[2],
                self._std_weight_position_dynamic * mean[3],
            ]
            noise_cov = np.diag(np.square(std))
            cov_proj += noise_cov
        return mean_proj, cov_proj

class IMMKalmanFilterXYWH(KalmanFilterXYWH):
    # CV、 CA、 CV+CA、 CV+CA+penalty、 IMM+penalty
    def __init__(self, mode: str = "IMM+penalty"):
        super().__init__()
        self.mode = mode
        self.mu = np.array([0.5, 0.5])
        self.trans_mat = np.array([[0.9, 0.1], [0.1, 0.9]])
        self.CA_motion_mat = self._create_motion_matrix(cv=False)

    def _create_motion_matrix(self, cv=True):
        dt = 1
        F = np.eye(8)
        for i in range(4):
            F[i, i + 4] = dt
        if not cv:
            for i in range(2):
                F[i + 4, i + 6] = dt
        return F

    def predict(self, mean, covariance):
        mean_cv = np.dot(self._motion_mat, mean)
        mean_ca = np.dot(self.CA_motion_mat, mean)
        cov_cv = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + np.eye(8) * 0.01
        cov_ca = np.linalg.multi_dot((self.CA_motion_mat, covariance, self.CA_motion_mat.T)) + np.eye(8) * 0.02

        if self.mode == "CV":
            return mean_cv, cov_cv
        elif self.mode == "CA":
            return mean_ca, cov_ca
        elif self.mode.startswith("IMM"):
            likelihood_cv = self._compute_likelihood(mean_cv, cov_cv)
            likelihood_ca = self._compute_likelihood(mean_ca, cov_ca)
            likelihoods = np.array([likelihood_cv, likelihood_ca])

            self.mu = self.trans_mat @ self.mu * likelihoods
            self.mu /= np.sum(self.mu)

            mean_fused = self.mu[0] * mean_cv + self.mu[1] * mean_ca
            cov_fused = self.mu[0] * cov_cv + self.mu[1] * cov_ca
            return mean_fused, cov_fused
        else:  # "CV+CA" or "CV+CA+penalty"
            return 0.5 * (mean_cv + mean_ca), 0.5 * (cov_cv + cov_ca)

    def associate_cost(self, prev_acc, curr_acc):
        if "penalty" not in self.mode:
            return 1.0
        acc_change = np.linalg.norm(np.array(curr_acc) - np.array(prev_acc))
        return np.exp(acc_change)

    def _compute_likelihood(self, mean, covariance):
        innovation = mean[:4]
        S = covariance[:4, :4]
        return np.exp(-0.5 * np.dot(innovation.T, np.linalg.inv(S)).dot(innovation))


class IMMKalmanFilterDynamic(KalmanFilterDynamicNoise):
    def __init__(self, noise_mode: str = "QR", motion_mode: str = "CV+CA+penalty"):
        super().__init__(mode=noise_mode)
        self.motion_mode = motion_mode
        self.mu = np.array([0.5, 0.5])
        self.trans_mat = np.array([[0.9, 0.1], [0.1, 0.9]])
        self.CA_motion_mat = self._create_motion_matrix(cv=False)

    def _create_motion_matrix(self, cv=True):
        dt = 1
        F = np.eye(8)
        for i in range(4):
            F[i, i + 4] = dt
        if not cv:
            for i in range(2):
                F[i + 4, i + 6] = dt
        return F

    def predict(self, mean, covariance):
        mean_cv, cov_cv = super().predict(mean, covariance), None
        mean_ca = np.dot(self.CA_motion_mat, mean)
        std_pos = self._std_weight_position_dynamic * mean[2:4].repeat(2)
        std_vel = self._std_weight_velocity_dynamic * mean[2:4].repeat(2)
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        cov_ca = np.linalg.multi_dot((self.CA_motion_mat, covariance, self.CA_motion_mat.T)) + motion_cov

        if self.motion_mode == "CV":
            return mean_cv, covariance
        elif self.motion_mode == "CA":
            return mean_ca, cov_ca
        elif self.motion_mode.startswith("IMM"):
            likelihood_cv = self._compute_likelihood(mean_cv, covariance)
            likelihood_ca = self._compute_likelihood(mean_ca, cov_ca)
            likelihoods = np.array([likelihood_cv, likelihood_ca])
            self.mu = self.trans_mat @ self.mu * likelihoods
            self.mu /= np.sum(self.mu)
            mean_fused = self.mu[0] * mean_cv + self.mu[1] * mean_ca
            cov_fused = self.mu[0] * covariance + self.mu[1] * cov_ca
            return mean_fused, cov_fused
        else:
            return 0.5 * (mean_cv + mean_ca), 0.5 * (covariance + cov_ca)

    def _compute_likelihood(self, mean, covariance):
        innovation = mean[:4]
        S = covariance[:4, :4]
        return np.exp(-0.5 * np.dot(innovation.T, np.linalg.inv(S)).dot(innovation))






