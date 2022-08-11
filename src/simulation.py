from scipy.integrate import odeint
from scipy.io import loadmat
import numpy as np
from . import system_data as sysdat


def system(y, t, u, sys):
    """Only linear terms"""
    y = y.reshape(3, 1)
    u = u.reshape(3, 1)
    dydt = np.array(sys.A.dot(y) + sys.B.dot(u))

    return dydt.flatten()


# TODO: Review this func
def system_nl(y, t, u, sys, nldat):

    delta = np.array([1.0, 1.0, 1.0]).reshape(3, 1)
    vs = np.array([1.0, 1.0, 1.0]).reshape(3, 1)
    E = np.diag(np.exp(-(np.abs(y) ** delta) / vs))

    F = sys.A.dot(y) + sys.B.dot(u)
    Fa = (nldat.Kc + nldat.Kf.dot(E)).dot(np.sign(y))
    if (np.abs(Fa) > np.abs(F)).any():
        Fa = -F
    dydt = np.array(F + Fa)

    return dydt.flatten()


class Simulation:
    def __init__(self, lti_sys, Ts, nonlinear=False, nldat=None):
        self.Ts = Ts
        self.states = np.zeros((3, 1))
        self.wheel_vel = np.zeros((3, 1))
        self.sys = lti_sys
        self.nonlinear = nonlinear
        if nonlinear:
            self.func = system_nl
        else:
            self.func = system
        self.nldat = nldat

    def simulation_step(self, u):
        t = [0, self.Ts]
        if self.nonlinear:
            y1 = odeint(
                self.func, self.states.flatten(), t, args=(u, self.sys, self.nldat)
            )
        else:
            y1 = odeint(self.func, self.states.flatten(), t, args=(u, self.sys))
        states = y1[1, :]
        states = states.reshape(3, 1)
        self.states = states
        self.wheel_vel = sysdat.S2W.dot(states)


def sim_closed_loop(reference, sim, K):

    N = len(reference)

    states_vec = np.empty((N, 3))
    control_signal = np.empty((N, 3))

    aug_states_old = np.zeros(shape=(6, 1))
    aug_states = np.zeros(shape=(6, 1))
    time = np.arange(0, N * sim.Ts, sim.Ts)

    for i in range(N):

        states = sim.states.reshape((3, 1))
        ref = reference[i, :].reshape((3, 1))

        aug_states[:3] = states - ref.reshape(3, 1)
        aug_states[3:] = aug_states_old[3:] + aug_states_old[:3]

        u = K.dot(aug_states)

        u[(u > sysdat.actuator_saturation)] = sysdat.actuator_saturation

        sim.simulation_step(u)

        aug_states_old = aug_states

        control_signal[i, :] = u.flatten()
        states_vec[i, :] = states.flatten()

    return states_vec, control_signal, time


def update_pose(states, pose_old, Ts):
    """Update the pose of the robot."""

    d = states[0] * Ts
    dn = states[1] * Ts
    dtheta = states[2] * Ts

    xm_ = pose_old[0]
    ym_ = pose_old[1]
    theta_ = pose_old[2]

    theta = theta_ + dtheta

    if dtheta == 0:
        xm = xm_ + d * np.cos(theta_) - dn * np.sin(theta_)
        ym = ym_ + d * np.sin(theta_) + dn * np.cos(theta_)

    else:
        xsin = np.sin(theta_ + dtheta / 2.0) / dtheta
        xcos = np.cos(theta_ + dtheta / 2.0) / dtheta

        xm = (
            xm_
            + (d * np.sin(dtheta) + dn * (np.cos(dtheta) - 1)) * xcos
            - (d * (1 - np.cos(dtheta)) + dn * np.sin(dtheta)) * xsin
        )
        ym = (
            ym_
            + (d * np.sin(dtheta) + dn * (np.cos(dtheta) - 1)) * xsin
            + (d * (1 - np.cos(dtheta)) + dn * np.sin(dtheta)) * xcos
        )

    pose = np.array([xm, ym, theta])

    return pose


class Trajectory:
    def __init__(self, traj, v_nav, radius=0.05):
        self.traj = traj
        self.ind = 0
        self.v_nav = v_nav
        self.radius = radius

    def update(self, pose):

        next_point = self.traj[self.ind, :]
        d = np.sqrt((next_point[0] - pose[0]) ** 2 + (next_point[1] - pose[1]) ** 2)
        if d < self.radius:
            self.ind = self.ind + 1
            if self.ind >= self.traj.shape[0]:
                return None
            else:
                next_point = self.traj[self.ind, :]

        xr = pose[0]
        yr = pose[1]
        theta = pose[2]
        R = np.array(
            [
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        phi = np.arctan2(next_point[1] - yr, next_point[0] - xr)
        e = np.array(
            [self.v_nav * np.cos(phi), self.v_nav * np.sin(phi), next_point[2] - theta]
        )
        ref = R.dot(e)

        return ref


def trajectory_sim(trajectory, v_nav, sim, K):

    traj = Trajectory(trajectory, v_nav)
    Ts = sim.Ts
    aug_states_old = np.zeros(shape=(6, 1))
    aug_states = np.zeros(shape=(6, 1))
    control_signal = np.zeros(shape=(3, 1))
    # pose_old = np.zeros(3)
    pose_old = trajectory[0]

    N = 5000
    control_signal_vec = np.empty(shape=(N, 3))
    states_vec = np.empty(shape=(N, 3))
    #    wheels_vel_vec = np.empty(shape=(N, 3))
    pose_vec = np.empty(shape=(N, 3))
    ref_vec = np.empty(shape=(N, 3))

    i = 0
    while True:

        states = sim.states.reshape((3, 1))
        pose = update_pose(states, pose_old, Ts)
        ref = traj.update(pose)

        if ref is None:
            break

        aug_states[:3] = states - ref.reshape(3, 1)
        aug_states[3:] = aug_states_old[3:] + aug_states_old[:3]

        control_signal = K.dot(aug_states)
        control_signal[
            (control_signal > sysdat.actuator_saturation)
        ] = sysdat.actuator_saturation

        states_vec[i, :] = states.flatten()
        control_signal_vec[i, :] = control_signal.flatten()
        ref_vec[i, :] = ref.flatten()
        pose_vec[i, :] = pose.flatten()

        sim.simulation_step(control_signal)

        aug_states_old = aug_states
        pose_old = pose
        i += 1

    return states_vec[:i], control_signal_vec[:i], ref_vec[:i], pose_vec[:i]


if __name__ == "__main__":

    sim = Simulation(sysdat.csys, sysdat.Ts)

    u = np.array([0, 1, -1])
    sim.simulation_step(u)
    print(sim.states)

    sim = Simulation(sysdat.csys, sysdat.Ts, nonlinear=True, nldat=sysdat.nldat)
    sim.simulation_step(u)
    print(sim.states)
