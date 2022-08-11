import json
import pickle
from datetime import datetime

import numpy as np
from .evaluation import StepInfoData


class Controller:
    """Holds all controller's information"""

    def __init__(self, K, norm, q, r, P, status, poles):
        self.K = K
        self.norm = norm
        self.q = q
        self.r = r
        self.P = P
        self.status = status
        self.poles = poles
        self.stepinfo = None
        self.u_max_var = None

    def __str__(self):
        return (
            "Center: {0:.2f}\n"
            "Radius: {1:.2f}\n"
            "Hinf Norm: {2:.2f}\n"
            "Status: {3}\n"
            "Poles: {4}".format(self.q, self.r, self.norm, self.status, self.poles)
        )

    def __repr__(self):
        return str(self)

    def todict(self):
        return {
            "K": self.K.tolist(),
            "norm": float(self.norm),
            "q": float(self.q),
            "r": float(self.r),
            "P": self.P.tolist(),
            "status": self.status,
            "poles": [str(p) for p in self.poles.tolist()],
            "u_max_var": float(self.u_max_var),
            "steopinfo": self.stepinfo.todict(),
        }

    @classmethod
    def fromdict(cls, d):
        c = cls(
            np.array(d["K"]),
            d["norm"],
            d["q"],
            d["r"],
            np.array(d["P"]),
            d["status"],
            np.array([complex(p) for p in d["poles"]]),
        )
        c.u_max_var = d["u_max_var"]
        c.stepinfo = StepInfoData(**d["steopinfo"])
        return c


def now():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def save(file_name, obj):

    with open(file_name, "wb") as f:
        pickle.dump(obj, f)


def load(file_name):

    with open(file_name, "rb") as f:
        obj = pickle.load(f)

    return obj


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)

    return data


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def load_controllers_json(path):
    data = load_json(path)
    return [Controller.fromdict(d) for d in data]


def make_ref(ref, N):
    return np.tile(ref, (N, 1))


def make_inv_ref(ref1, ref2, N):
    a = np.tile(ref1, (int(N / 2), 1))
    b = np.tile(ref2, (int(N / 2), 1))

    return np.vstack((a, b))


def get_clpoles(Ad, Bd, K):

    Z = np.zeros((3, 3))
    I = np.eye(3)
    B2 = np.block([[Bd], [Z]])
    A = np.block([[Ad, Z], [I, I]])

    return np.linalg.eig(A + B2.dot(K))[0]


def get_clmat(Ad, Bd, Cd, Dd, K):

    Z = np.zeros((3, 3))
    I = np.eye(3)
    A = np.block([[Ad, Z], [I, I]])
    B1 = np.vstack((I, Z))
    B2 = np.vstack((Bd, Z))
    C1 = np.block([[I, Z], [Z, Z]])
    D11 = np.block([[Z], [Z]])
    D12 = np.block([[Z], [I]])

    Acl = A + B2.dot(K)
    Bcl = B1
    Ccl = C1 + D12.dot(K)
    Dcl = D11

    return Acl, Bcl, Ccl, Dcl


def load_square_traj():
    """One meter side, 20cm spaced marks"""
    traj = np.array(
        [
            [0, 0, 0],
            [0.2, 0, 0],
            [0.4, 0, 0],
            [0.6, 0, 0],
            [0.8, 0, 0],
            [1, 0, 0],
            [1, 0.2, 0],
            [1, 0.4, 0],
            [1, 0.6, 0],
            [1, 0.8, 0],
            [1, 1, 0],
            [0.8, 1, 0],
            [0.6, 1, 0],
            [0.4, 1, 0],
            [0.2, 1, 0],
            [0, 1, 0],
            [0, 0.8, 0],
            [0, 0.6, 0],
            [0, 0.4, 0],
            [0, 0.2, 0],
            [0, 0, 0],
        ]
    )
    return traj


def load_8_traj():

    angles = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180])
    angles1 = np.hstack((angles, np.flip(-angles[:-1], 0)))
    angles1 = np.pi / 180 * angles1
    angles2 = np.hstack((np.flip(angles, 0), -angles[1:]))
    angles2 = np.pi / 180 * angles2
    circle1 = np.zeros(shape=(len(angles1), 3))
    circle1[:, 0] = np.cos(angles1)
    circle1[:, 1] = np.sin(angles1)

    circle2 = np.zeros(shape=(len(angles2), 3))
    circle2[:, 0] = np.cos(angles2)
    circle2[:, 1] = np.sin(angles2)

    circle1 = circle1 + np.array([1, 0, 0])
    circle2 = circle2 + np.array([3, 0, 0])
    traj = np.vstack((np.array([0, 0, 0]), circle2[0:-1], circle1))

    return traj


def load_circle_J_traj():
    from scipy.io import loadmat

    return loadmat("data/circle_traj_J.mat")["traj"]
