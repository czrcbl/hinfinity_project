import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/src')
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/robot')
import numpy as np
import time
import system_data as sysdat
from serial_com import RobotCom
from robot_utils import SimulationAdapter, vis_experiment, DataLogger
from utils import (load, save, load_square_traj, load_circle_J_traj, make_ref)
from simulation import Trajectory, update_pose

def main():
    
    #For metadata only
    controller_file = 'controller0.pkl'
    
    controller = controller_file.split('.')[0]
    
    Ts = sysdat.Ts
    time.sleep(5)
    
    verbose = False

    is_ref = False
    if is_ref:
         reference = make_ref([0.6, 0.0, 0], 100)
        # reference = make_ref([0.0, 0.6, 0], 40)
        # reference = make_ref([0.0, 0.0, 2], 40)
        # reference = make_S_traj()
        # reference = make_quad_traj()
#        reference = make_inv_ref([0.6, 0.0, 0], [-0.6, 0.0, 0], 100)
    else:
        traj = Trajectory(load_square_traj(), 0.3)
        # traj = Traj(load_circle_J_traj(), 0.3)

    # Init Serial
    try:
        com = RobotCom()
        suffix = controller + '_experiment'
    except Exception as e:
        print(e)
        suffix = controller + '_simulation'
        com = SimulationAdapter(sysdat.csys, Ts)

    # Init Variables
    cont = load('data/' + controller_file)
    K = cont.K
    
    aug_states_old = np.zeros(shape=(6, 1))
    aug_states = np.zeros(shape=(6, 1))
    control_signal = np.zeros(shape=(3, 1))
    pose_old = np.zeros(3)
    logger = DataLogger('experiments', cont)

    com.init_serial()

    i = 0
    while True:

        print('Sample {}'.format(i))

        data = com.receive_message()
        wheels_vel = np.array([data.m1_vel, data.m2_vel, data.m3_vel]).reshape(3,1)

        tic = time.time()
        states = sysdat.W2S.dot(wheels_vel)
        pose = update_pose(states, pose_old, Ts)

        if is_ref:
            try:
                ref = reference[i, :]
            except IndexError:
                ref = None
        else:
            ref = traj.update(pose)

        if ref is None:
            break

        aug_states[:3] = states - ref.reshape(3, 1)
        aug_states[3:] = aug_states_old[3:] + aug_states_old[:3]

        control_signal = K.dot(aug_states)

        if verbose:
            print('wheels_vel', wheels_vel)
            print('States', states)
            print('pose', pose)
            print('Control Signal:', control_signal)

        print('delay', time.time() - tic)

        com.send_control_signal(control_signal)

        try:
            logger.update(control_signal, states, wheels_vel, pose, ref)
        except IndexError:
            break

        aug_states_old = aug_states
        pose_old = pose
        i += 1

    com.stop_motors()
    logger.close(suffix)

    print('Visualizing data:')
    vis_experiment()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        # com.send_control_signal(np.array([0, 0, 0]))
        raise e
