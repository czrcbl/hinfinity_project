import numpy as np
import cvxpy as cvx
from utils import save, step_info, make_inv_ref
from simulation import Simulation, sim_closed_loop
from utils import get_clpoles


class OptException(Exception):
    pass


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
        return 'Center: {0:.2f}\n'\
                'Radius: {1:.2f}\n'\
                'Hinf Norm: {2:.2f}\n'\
                'Status: {3}\n'\
                'Poles: {4}'.format(self.q, self.r, self.norm, self.status, self.poles)
        
    def __repr__(self):
        return str(self)
        

def project(Ad, Bd, Cd, Dd, q, r, solver=cvx.SCS):
    """Perform the project for z = [y; u], for a discrete time LTI system.
    
    Parameters:
    -----------
        Ad, Bd, Cd, Dd : Discrete time system state space matrices.
        q : cicular region center, over the x axis.
        r : circular region radius.
        solver : a string with the cvx solver name.
    
    Returns:
    --------
        K : gain matrix of the controler.
        Hinf_norm : Hinf norm of the closed loop system.
        Pop : Lyapunov matrix
        status : Solver status.
    """
    
    Z = np.zeros((3,3))
    I = np.eye(3)
    A = np.block([[Ad, Z], [I, I]])
    B1 = np.vstack((I, Z))
    B2 = np.vstack((Bd, Z))
    C1 = np.block([[I, Z], [Z, Z]])
    D11 = np.block([[Z], [Z]])
    D12 = np.block([[Z], [I]])
    C2 = np.block([I, Z])
    D21 = Z
    D22 = Z

    return hinf_project_pole_alloc(A, B1, B2, C1, C2, D11, D12, D21, D22, q, r, solver)


def hinf_project_pole_alloc(A, B1, B2, C1, C2, D11, D12, D21, D22, q, r, solver=cvx.SCS):
    """ 
    Perform hinf project with pole allocation in a circular subregion, for
    discrete time LTI systems.
    
    Parameters:
    -----------
        A, B1, ... D22 : Generalized form system matrices.
        q : cicular region center, over the x axis.
        r : circular region radius.
        solver: a string with the cvx solver name.
    
    Returns:
    --------
        K : gain matrix of the controler.
        Hinf_norm : Hinf norm of the closed loop system.
        Pop : Lyapunov matrix.
        status : Solver status.
    """
    
    assert r > 0, 'r must be positive.'
    assert np.abs(q) + r < 1, 'the region must be inside the unit circle.'
    
    tol = 1e-20
    n = A.shape[0]
    
    L = cvx.Variable((B2.shape[1], n))
    P = cvx.Variable((n, n))
    gamma2 = cvx.Variable()
    
    LMI1 = cvx.bmat([
            [P, A*P + B2*L, B1, np.zeros((B1.shape[0], D11.shape[0]))],
            [P*A.T + L.T * B2.T, P, np.zeros((P.shape[0], B1.shape[1])), P*C1.T + L.T*D12.T],
            [B1.T, np.zeros((B1.shape[1], P.shape[1])), np.eye(B1.shape[1]), D11.T],
            [np.zeros((C1.shape[0], B1.shape[0])), C1*P + D12*L, D11, gamma2*np.eye(D11.shape[0])]
              ])
    
    cons1 = LMI1 >> tol
    
    LMI2 = cvx.bmat([
            [-r*P, -q*P + A*P + B2*L],
            [-q*P + P*A.T + L.T*B2.T, -r*P]
            ])
    
    cons2 = LMI2 << -tol
    
    cons3 = gamma2 >= tol
    
    cons4 = P == P.T
    
    cons5 = P >> tol
    
    prob = cvx.Problem(cvx.Minimize(gamma2), constraints=[cons1, cons2, cons3, cons4, cons5])
    prob.solve(solver=solver)
    
    status = prob.status
    if not status in [cvx.OPTIMAL_INACCURATE, cvx.OPTIMAL]:
        #variable.value will be None, better trow an exception
        raise OptException(f'Problem is {status}')
        
    Hinf_norm = np.sqrt(gamma2.value)
    Pop = P.value
    Lop = L.value
    
    K = Lop.dot(np.linalg.inv(Pop))
    
    return K, Hinf_norm, Pop, status


def multiple_projects(Ad, Bd, Cd, Dd, csys, Ts, q_step=0.05, r_step=0.05, solver=cvx.SCS):
    """
    Project and evaluate a set of controllers for multiple values of 
    radiuses and centers
    """
    ref = make_inv_ref([0.6, 0, 0], [-0.6, 0, 0], 100)
    
    centers = np.arange(0, 1, q_step)
    radius = np.arange(0.05, 1, r_step)
    
    controllers = []
    
    for q in centers:
        for r in radius:
            if np.abs(q) + r < 1:
                print('Center: %f, Radius: %f' % (q, r))
                try:
                    K, norm, P, status = project(Ad, Bd, Cd, Dd, q, r, solver=solver)
                    c = Controller(K, norm, q, r, P, status, get_clpoles(Ad, Bd, K))
                    states, control_signal, time = sim_closed_loop(ref, Simulation(csys, Ts), K)
                    info = step_info(time[:40], states[:40, 0])
                    c.stepinfo = info
                    c.u_max_var = np.max(np.abs(control_signal[:-1, :] - control_signal[1:, :]))
                    
                except OptException as e:
                    # Save the status
                    status = str(e).split()[-1]
                    c = Controller(None, None, c, r, None, status, None)
                except cvx.error.SolverError:
                    status = 'solver_error'
                    c = Controller(None, None, c, r, None, status, None)
                    
                print(status)                    
                controllers.append(c)
                    
    return controllers
    
    
if __name__ == '__main__':
    
    import system_data as sysdat
    from system_data import Ad, Bd, Cd, Dd
    
    controllers = multiple_projects(Ad, Bd, Cd, Dd, sysdat.csys, sysdat.Ts, q_step=0.05, r_step=0.05, solver=cvx.SCS)
    save('data/controllers_solver-SCS_all.pkl', controllers)
#    eval_controllers('data/controllers_solver-SCS_all.pkl', sysdat.csys, sysdat.Ts, controllers)

    
#    controllers = multiple_project(Ad, Bd, Cd, Dd, solver=cvx.MOSEK)
#    eval_controllers('data/controllers_solver-MOSEK_all.pkl', sysdat.csys, sysdat.Ts, controllers)

 
