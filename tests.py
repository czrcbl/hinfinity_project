import system_data as sysdat
import numpy as np
from simulation import Simulation
import control
import matplotlib.pyplot as plt
from utils import step_info
import sys


MIN_FLOAT = sys.float_info.min


def test_2nd_step_info(xi, wn):
    
    A = np.array([[0, 1], [-wn**2, -2*xi*wn]])
    B = np.array([[0], [wn**2]])
    C = np.array([[1, 0]])
    D = np.array([[0]])
    
    system = control.ss(A, B, C, D)
    st = 4/(xi*wn)
    os = np.exp(-xi*np.pi/(MIN_FLOAT + np.sqrt(1 - xi**2)))
    #system = sysdat.Sys(A, B, C, D)
    
    time, yout = control.step_response(system)

    si = step_info(time, yout)
    
    print('xi:', xi, 'wn:', wn)
    print('Theoretical / Measured')
    print(st, si.SettlingTime)
    print(os, si.Overshoot)
    print()
    
    
def test_simulation():
    from system_data import Ad, Bd, Cd, Dd
    sim = Simulation(sysdat.csys, sysdat.Ts)
    
    u = np.array([0, 0.6, -0.6]).reshape((3, 1))
    x = np.zeros((3,1))
    for i in range(100):
        x = Ad.dot(x) + Bd.dot(u)
        y = Cd.dot(x) + Dd.dot(u)
        sim.simulation_step(u)
        assert np.linalg.norm(y - sim.states) < 1e-6, 'Simulation missmatch'
    
    print('test_simulation passed', np.linalg.norm(y - sim.states))
    
    
if __name__ == '__main__':
    
    wn = 2
    xi = 0.5
    test_2nd_step_info(xi, wn)
    
    wn = 2
    xi = 1
    test_2nd_step_info(xi, wn)
    
    wn = 4
    xi = 0.1
    test_2nd_step_info(xi, wn)
    
    test_simulation()

    
