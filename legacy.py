import streamlit as st
import os
import sys
sys.path.append(os.getcwd() + '/src')
import numpy as np
import matplotlib.pyplot as plt
from utils import load, make_ref, make_inv_ref, get_clpoles, step_info, load_square_traj, load_8_traj, load_circle_J_traj
from simulation import Simulation, sim_closed_loop, trajectory_sim
import system_data as sysdat
from system_data import Ad, Bd, Cd, Dd
from vis_data import triple_plot, triple_plot2, plot_poles, scatter_controllers, plot_pose
from project import project, Controller
import pandas as pd

st.markdown("# Legacy 🎈")
st.sidebar.markdown("# Legacy 🎈")

all_controllers = load('data/controllers_solver-SCS_all.pkl')
print('Number of projects:', len(all_controllers))

df = pd.DataFrame({'status': [c.status for c in all_controllers]})
df_g = df.groupby('status')['status'].count()
st.write(df_g)
df_g.plot.barh()


q = 0.0
r = 0.99999
K, baseline_norm, P, status = project(Ad, Bd, Cd, Dd, q, r)

print('Optimization status:', status)
print('Minimal Hinf norm %.2f' % baseline_norm)

poles = get_clpoles(Ad, Bd, K)
fig = plot_poles(poles, q, r)
st.pyplot(fig)
st.write(poles)


ref = make_inv_ref([0.6, 0, 0], [-0.6,0,0], 100)
states, control_signal, time = sim_closed_loop(ref, Simulation(sysdat.csys, sysdat.Ts), K)
baseline_info = step_info(time[:40], states[:40, 0])

baseline_u_max_var = np.max(np.abs(control_signal[:-1, :] - control_signal[1:, :]))
st.write(f'Max. Control Signal Var: {baseline_u_max_var}')

st.write(baseline_info)
fig = triple_plot2(states, ref, 'States', 'Reference', figsize=(8, 10))
st.pyplot(fig)
fig = triple_plot(control_signal, 'Control Signal', figsize=(8, 10))
st.pyplot(fig)

controllers = [c for c in all_controllers if not c.poles is None]

settling_time = np.array([c.stepinfo.SettlingTime for c in controllers])
over_shoot = np.array([c.stepinfo.Overshoot for c in controllers])
u_var = np.array([c.u_max_var for c in controllers])
norm = np.array([c.norm for c in controllers])

arrs = [norm, settling_time, over_shoot, u_var]
titles = [r'$H_{\infty} Norm$', 'Settling Time', 'Overshoot', 'Control Signal Max. Var,']
fig  = scatter_controllers(arrs, titles)
st.pyplot(fig)

idx = np.where((st < 1.4) & (u_var < 6) & (norm < 25))[0]
filtered_arrs = [arr[idx] for arr in arrs]
fig = scatter_controllers(filtered_arrs, titles)
st.pyplot(fig)

c = controllers[idx[-1]]
print(c)
print('Control Signal Max. Var.:', c.u_max_var)
print('Status:', c.status)