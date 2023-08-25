import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag


class KalmanBoxTracker:
    def __init__(self, bbox):
        """
        Initialize the tracker using the initial bounding box.
        bbox: A 1D array of shape (4,) representing [x, y, w, h]
        """
        # Define the Kalman filter
        self.missed = 0
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # State transition matrix (constant velocity model)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        # Observation matrix (we observe position but not velocity)
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )

        # Initial state
        self.kf.x[:4] = np.reshape(bbox, (4, 1))
        self.kf.x[4:] = 0

        # Covariance matrices
        self.kf.P *= 10
        self.kf.Q = block_diag(np.diag([1, 1, 1, 1]), np.diag([1, 1, 1, 1]))
        self.kf.R = np.diag([1, 1, 1, 1]) * 10

    def predict(self):
        """
        Predict the next state.
        """
        self.kf.predict()
        return self.kf.x[:4]

    def update(self, bbox):
        """
        Update the state using the observed bounding box.
        """
        self.kf.update(bbox)
        return self.kf.x[:4]
