import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanFilter3D(object):
    """
    3D Kalman Filter for tracking objects in 3D space.
    The state space is 10-dimensional: (x, y, z, theta, l, w, h, vx, vy, vz)
    The measurement space is 7-dimensional: (x, y, z, theta, l, w, h)
    """

    def __init__(self, bbox3D, ID=None):
        self.initial_pos = bbox3D  # Initial observation [x, y, z, theta, l, w, h]
        self.id = ID
        self.time_since_update = 0
        self.hits = 1  # number of total hits including the first detection

        self.kf = KalmanFilter(dim_x=10, dim_z=7)

        # State transition matrix (F)
        # x' = x + vx, y' = y + vy, z' = z + vz
        # All other states (theta, l, w, h, vx, vy, vz) remain constant (decay=1)
        # x, y, z, theta, l, w, h, vx, vy, vz
        self.kf.F = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # x
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],  # y
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],  # z
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # theta
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # l
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # w
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # h
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # vx
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # vy
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]   # vz
        ])

        # Measurement function (H)
        # We observe (x, y, z, theta, l, w, h) directly
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        ])

        # Measurement uncertainty (R)
        # self.kf.R[0:,0:] *= 10.

        # Initial state uncertainty (P)
        self.kf.P[7:, 7:] *= 1000.  # High uncertainty for initial velocity
        self.kf.P *= 10.

        # Process uncertainty (Q)
        self.kf.Q[7:, 7:] *= 0.01  # Low uncertainty for constant velocity assumption

        # Initialize state
        self.kf.x[:7] = self.initial_pos.reshape((7, 1))

    def predict(self):
        self.kf.predict()
        # Normalize theta to be within range if needed, though usually handled in matching
        # self.kf.x[3] = self.within_range(self.kf.x[3])

    def update(self, bbox3D):
        self.kf.update(bbox3D)
        
    def get_state(self):
        return self.kf.x.reshape((-1))
    
    def get_velocity(self):
        return self.kf.x[7:]
