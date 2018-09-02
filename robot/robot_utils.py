import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)) + '/src')
from simulation import Simulation
import time
from scipy.io import loadmat, savemat
import numpy as np
from serial_com import RecvData
import system_data as sysdat
from vis_data import triple_plot, triple_plot2
from utils import save, load
import matplotlib.pyplot as plt

def now():
    """
    Returns system current time (until seconds)
    """
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


class SimulationAdapter:

    def __init__(self, csys, Ts):
        self.sim = Simulation(csys, Ts)

    def send_control_signal(self, control_signal):
        self.sim.simulation_step(control_signal)

    def receive_message(self):
        wheels_vel = self.sim.wheel_vel.flatten()
        data = RecvData()
        data.m1_vel = wheels_vel[0]
        data.m2_vel = wheels_vel[1]
        data.m3_vel = wheels_vel[2]

        return data

    def init_serial(self):
        print('Simulation Initialized!')

    def stop_motors(self):
        self.send_control_signal(np.array([0, 0, 0]))
        

def vis_experiment(file_name=None):
    if file_name is None:
        file_name = get_recent()

    data = load(file_name)
    control = data['control_signal']
    states = data['states']
    wheels = data['wheels_vel']
    pose = data['pose']
    ref = data['reference']

    # triple_plot(states, 'States')
    triple_plot(control, 'Control Signals')
    triple_plot(wheels, 'Wheels Velocities')
    # triple_plot(ref, 'reference')
    triple_plot2(states, ref, 'states', 'reference')
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pose[:, 0], pose[:, 1])
    fig.suptitle('Position')
    plt.grid(True)

    plt.show()


def get_recent():
    results_folder = 'experiments'
    files = os.listdir(results_folder)
    files.sort()
    return os.path.join(results_folder, files[-1])


def load_controler(file_path, pprint=False):
    controller = loadmat(file_path)
    if pprint:
        print('Controller Loaded:')
        print(controller)
    return controller['K']


class DataLogger:

    def __init__(self, data_folder, controller):
        N = 5000
        self.control_signal_vec = np.empty(shape=(N, 3))
        self.states_vec = np.empty(shape=(N, 3))
        self.wheels_vel_vec = np.empty(shape=(N, 3))
        self.pose_vec = np.empty(shape=(N, 3))
        self.ref_vec = np.empty(shape=(N, 3))
        self.iter = 0
        self.data_folder = data_folder
        self.controller = controller

    def update(self, control_signal, states, wheels_vel, pose, ref):

        i = self.iter
        self.control_signal_vec[i, :] = control_signal.flatten()
        self.states_vec[i, :] = states.flatten()
        self.wheels_vel_vec[i, :] = wheels_vel.flatten()
        self.pose_vec[i, :] = pose.flatten()
        self.ref_vec[i, :] = ref.flatten()
        self.iter += 1

    def close(self, suffix):
        i = self.iter

        if not os.path.isdir(self.data_folder):
            os.mkdir(self.data_folder)
        out = {'states': self.states_vec[:i],
               'wheels_vel': self.wheels_vel_vec[:i],
               'control_signal': self.control_signal_vec[:i],
               'reference': self.ref_vec[:i],
               'pose': self.pose_vec[:i],
               'controller': self.controller
                            }
        test_name = os.path.join(self.data_folder, now() + '_' + suffix + '.pkl')
        
        save(test_name, out)
        












if __name__ == '__main__':
    
    com = SimulationAdapter(sysdat.csys, sysdat.Ts)
    com.send_control_signal(np.array([1, 1, 1]))
    data = com.receive_message()
    print(data)