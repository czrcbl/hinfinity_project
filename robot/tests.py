import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/src')
from serial_com import RobotCom
import system_data as sysdat 
import numpy as np
from scipy.io import savemat
import time


def test_sampling_time():

    Ts = sysdat.Ts
    try:
        com = RobotCom()
        print('Port Openned:', com.serial.name)
    except Exception as e:
        raise e

    com.init_serial()
    control_signal = np.array([1, 1, 1])
    tac = time.time()
    for n in range(100):

        data = com.receive_message()
        tic = time.time()
        if n > 0:
            delay = tic - tac
            print('delay', delay)
            # Check if the sampling time is within a 10% error
        tac = tic

        print(data.m1_vel)
        print(data.m2_vel)
        print(data.m3_vel)
        com.send_control_signal(control_signal)

    com.stop_motors()


def test_movement(control_signal):

    try:
        com = RobotCom()
        print('Port Openned:', com.serial.name)
    except Exception as e:
        print(e)

    com.init_serial()
    N = 100 # Number of samples
    for _ in range(N):
        data = com.receive_message()
        vel = np.array([data.m1_vel, data.m2_vel, data.m3_vel])
        print('wheel', vel)
        states = wheel2states(vel)
        print('States', states)
        com.send_control_signal(control_signal)

    com.stop_motors()

def test_frontal_movement():

    control_signal = np.array([0, 2, -2])
    print('Testing frontal movement.')
    test_movement(control_signal)


def test_normal_movement():

    control_signal = np.array([-2, 1, 1])
    print('Testing normal movement.')
    test_movement(control_signal)


def test_angular_movement():

    control_signal = np.array([1, 1, 1])
    print('Testing angular movement.')
    test_movement(control_signal)



if __name__ == '__main__':

    test_sampling_time()
    time.sleep(2)
    test_frontal_movement()
    time.sleep(2)
    test_normal_movement()
    time.sleep(2)
    test_angular_movement()



# N = 20
# states_vec = np.empty(shape=(N, 3))
# control_signal_vec = np.empty(shape=(N, 3))
# wheel_vel_vec = np.empty(shape=(N, 3))

# test_n = 2

# if test_n == 0:
#     # 1 Deslocamento frontal
#     init_serial(ser)
#     control_signal = np.array([0, 2, -2])
#     for n in range(N):
#         data = recv_message(ser)
#         vel = np.array([data.m1_vel, data.m2_vel, data.m3_vel])
#         print('wheel', vel)
#         states = wheel2states(vel)
#         print('states', states)
#         send_control_signal(control_signal, ser)
#         wheel_vel_vec[n, :] = vel.flatten()
#         states_vec[n, :] = states.flatten()
#         control_signal_vec[n, :] = control_signal.flatten()

#     send_control_signal(np.array([0, 0, 0]), ser)
#     savemat('openloop/frontal', {'control': control_signal_vec,
#                                  'states': states_vec,
#                                  'wheel_vel:': wheel_vel_vec})

# elif test_n == 1:
#     # 2 Deslocamento normal
#     init_serial(ser)
#     control_signal = np.array([-2, 1, 1])
#     for n in range(N):
#         data = recv_message(ser)
#         vel = np.array([data.m1_vel, data.m2_vel, data.m3_vel])
#         wheel_vel_vec[n, :] = vel.flatten()
#         print('wheel', vel)
#         states = wheel2states(vel)
#         states_vec[n, :] = states.flatten()
#         print('states', states)
#         send_control_signal(control_signal, ser)
#         control_signal_vec[n, :] = control_signal.flatten()

#     send_control_signal(np.array([0, 0, 0]), ser)
#     savemat('openloop/normal', {'control': control_signal_vec,
#                                 'states': states_vec,
#                                 'wheel_vel:': wheel_vel_vec})

# elif test_n == 2:
#     # 3 Deslocamento angular
#     init_serial(ser)
#     control_signal = np.array([1, 1, 1])
#     for n in range(N):
#         data = recv_message(ser)
#         vel = np.array([data.m1_vel, data.m2_vel, data.m3_vel])
#         wheel_vel_vec[n, :] = vel.flatten()
#         print('wheel', vel)
#         states = wheel2states(vel)
#         states_vec[n, :] = states.flatten()
#         print('states', states)
#         send_control_signal(control_signal, ser)
#         control_signal_vec[n, :] = control_signal.flatten()

#     send_control_signal(np.array([0, 0, 0]), ser)
#     savemat('openloop/angular', {'control': control_signal_vec,
#                                  'states': states_vec,
#                                  'wheel_vel:': wheel_vel_vec})
