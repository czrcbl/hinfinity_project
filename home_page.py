import streamlit as st
import pandas as pd
st.set_page_config(page_title="Home Page", layout="wide")

TITTLE = "# 🕮 Home Page - Theory"

def robot_description():

    st.markdown("## Robot Description")
    
    st.write("A scheme of the the forces and torques acting over the robots' center of mass.")
    st.image('figures/diagrama_dinamica_english.png', width=600)
    st.markdown(r"""
    The forces and torque in the robot's center of mass are given by the following equations:
    $$
    M\dot{v}(t) = F_v(t) - F_{va}(t) \\
    M\dot{v_n}(t) = F_{v_n}(t) - F_{v_na}(t) \\
    J\dot{\omega}(t) = \Gamma(t) - \Gamma_a(t)
    $$

    where $ F_v(t)$, $ F_{v_n}(t)$ and $\Gamma(t)$ correspond to the resultant forces and torque due to the traction forces of each wheel.

    Each wheel is attached to a identical DC motor. 
    Let $r$ be the radius of each wheel, the traction force on the wheel $i$ is given by:
    $$
    F_{mi}(t) = \frac{T_i(t)}{r_i}
    $$

    $T_i(t)$ is the torque produced by the motor $i$, and it can be related to the input voltage by the equations:
    $$
    u_i(t) =L_{ai}\dfrac{di_{ai}(t)}{dt}+R_{ai}i_{ai} + k_{vi}\omega_{mi}(t), \\
    T_i = l_ik_{ti}i_{ai}.
    $$

    $L_{ai}$ and $R_{ai}$ are the inductance end the resistance, respectively, of the motor winding.
    $K_{ti}$ is the motor's torque constant,  $K_{vi}$ is the *emf* constant and $l_i$ is the axis reduction factor.
    
    The scalar velocity wheel $i$ ($v_{ri}$) is related to its rotor's angular velocity ($\omega_{mi}$) by:
    $$
    v_{ri} = \frac{r_i\omega_{mi}}{l_i}.
    $$

    On the modeling process, four friction phenomena were considered: the Coulomb 
    friction (dynamic friction); the viscous friction, that is modeled as a linear 
    function of the 
    velocity; the Stiction, that models the friction when the system is at rest; 
    and the Stribeck effect, that corresponds to the passage of static to dynamic 
    friction for low velocities. 
    
    Next figure presents a diagram where the four effects can be visualized.
    """)
    st.image("figures/frictions.png", width=600)

    st.markdown(r"""
    The frontal and normal friction forces; and the friction torque are given by: 

    $$
    F_{va} = B_vv + [C_v + (F_{sv} - C_v)e^{\frac{-\lvert v 
            \rvert^{\delta_{s}}}{v_{sv}}}]sign(v), \\
    F_{v_na} = B_{v_n}v_n + [C_{v_n} + (F_{s{v_n}} - C_{v_n})e^{\frac{-\lvert 
            v_n
            \rvert^{\delta_{s}}}{v_{sv_n}}}]sign(v_n), \\
    
    \Gamma_a = B_{\omega}\omega + [C_{\omega} + (F_{s\omega} - 
    C_{\omega})e^{\frac{-\lvert \omega
            \rvert^{\delta_{s}}}{\omega_{s\omega}}}]sign(\omega),
    $$
    in which $B_v$, $B_{v_n}$ and $B_\omega$ are the viscous friction coefficients 
    related to each defined direction. The Coulomb friction coefficients are 
    represented by $C_v$, $C_{v_n}$ and $C_w$.
    The Stiction forces and torque are given by $F_{sv}$, $F_{sv_{n}}$ 
    and $F_{s\omega}$, and the Stribeck velocities are $v_{sv}$, $v_{sv_n}$ and 
    $v_{s\omega}$.  $\delta_{s}$ is a adjustment parameter.

    Based on the equations that describe the system's dynamic model, the following state space representation was derived:
    $$
    \dot{\xi}(t) = A_c\xi(t) + B_cu(t) + (K_c + GE(t))sign(\xi), \\
    y(t) = C_c\xi(t) + D_cu(t),
    $$
    where the vector $u(t) = \begin{bmatrix} u_1(t) & u_2(t) & u_3(t) 
    \end{bmatrix}^T$ corresponds to the input voltage of the motors.

    ## Discrete Time Model

    The plant was discretized using a ZOH with a sample time of 60 ms. In the process, the nonlinear terms, inherent to 
    the friction phenomena, were modeled as a perturbation $w_{k}$ that acts 
    directly 
    over the states. The state space representation obtained has the following form:
    $$
    \xi_{k+1} = A_d\xi_{k} + B_du_{k} + w_{k}, \\
    y_{k} = C_d\xi_{k} + D_du_{k}
    $$

    ## Augmented Model

    In order to obtain null tracking error for constant references, an integral action should be added to the plant.

    In the \hinf{} project, this was achieved by the adoption of following new state vector:
    $$
    x_{k} = \begin{bmatrix}
    e_{k} \\ \eta_{k},
    \end{bmatrix},
    $$
    where:
    $$
    e_{k} = y_{k} - r_{k}, \\
    \eta_{k + 1} = \eta_{k} + e_{k},
    $$
    and $r_k$ is the reference signal. 
    Thus, the new state space representation obtained is given by:
    $$
    x_{k+1} = A_{aug}x_{k} + B_{aug}u_{k} + B_ww_k, \\
    y_{k} = C_{aug}x_{k} + D_{aug}u_{k}.
    $$

    And the new state matrices, as a function of the discrete time state matrices, 
    are given by:
    $$
    A_{aug} =  \begin{bmatrix}
    A_d & 0 \\
    I & I
    \end{bmatrix},
    B_{aug} = \begin{bmatrix}
    B_d \\
    0
    \end{bmatrix},\\
    C_{aug} = \begin{bmatrix}
    C_d & 0
    \end{bmatrix},
    D_{aug} = D_d,
    B_w = \begin{bmatrix}
    I \\
    0
    \end{bmatrix}
    $$
    $I$ is a identity matrix and $0$ is a square matrix of zeros, both of $3^{rd}$ order.
    """)

def project_description():

    st.markdown("## Controller Description")
    
    st.image('figures/forma_geral.png', width=600)

    st.markdown(r"""
    
    ## $\mathcal{H}_\infty$ Controller


    On this technique formulation, the generalized form is adopted, it is represented on Figure \ref{fig:forma_geral}. 
    In this form, $u$ is the vector of control inputs; $y$ is the vector of outputs; $w$ is the vector of external disturbances and $z$ is the vector of controlled outputs.
    On this setting, we want to project an optimal controller regarding the $\mathcal{H}_\infty$ norm of $H_{zw}(s)$, the transfer function  from the input $w$ to the output $z$.

    The respective state space representation in terms of the described signals is: 
    $$
    x_{[k+1]} = Ax_{[k]} + B_1w_{[k]} + B_2u_{[k]}, \\
    z_{[k]} = C_1x_{[k]} + D_{11}w_{[k]} + D_{12}u_{[k]}, \\
    y_{[k]} = C_2x_{[k]} + D_{21}w_{[k]} + D_{22}u_{[k]}.
    $$

    In this work, the $z$ signal is defined as follows:
    $$
    z =  \begin{bmatrix}
    y \\ 
    u
    \end{bmatrix},
    $$
    i.e, the effects of the perturbations over the output and the control signal should be reduced.

    In order to obtain a state feedback controller, that guarantees the inequality $\lVert H_{zw} \rVert_\infty < \gamma$ the following LMI must be feasible:
    
    $$
    \begin{bmatrix}
    P & AP+B_2L & B_1 & 0 \\
    PA^T+L^TB_2^T & P & 0 & PC_1^T + L^TD_{12}^T \\
    B_1^T & 0 & I & D_{11}^T \\
    0 & C_1P + D_{12}L & D_{11} & \gamma^2I
    \end{bmatrix} > 0,
    $$
    where matrices $P \in \mathbb{R}^{6\times6}$ and $L \in \mathbb{R}^{3\times6}$ are the optimization variables.

    An optimal controller regarding the $\mathcal{H}_\infty$ norm of the transfer matrix 
    $H_{zw}(s)$ can be obtained by solving the following convex optimization 
    problem: minimize $\gamma^2$ subject to the LMI defined in the previous equation. The controller's gain matrix is obtained by:
    $$
    K = LP^{-1}.
    $$

    The system's poles can be allocated in a circular subregion of the complex plane with center $q$ over the real axis, and radius $r$.
    This can be achieved by adding to the anterior optimization problem the following LMI:
    $$
    \begin{bmatrix}
    -rP & -qP + AP + B_2L \\
    -qP + PA^T + L^TB_2^T& -rP
    \end{bmatrix} < 0 
    $$

    By allocating the poles, it is possible to achieve a trade-off between the 
    system's transitory response and the magnitude of the $\mathcal{H}_\infty$ norm.
    """)

def project_bibliography():

    st.markdown("## Bibliography")

    bib_df = pd.read_csv('assets/bibliography.csv', dtype={"Publication Year": str})
    bib_df = bib_df[bib_df["Author"].notna()]
    st.dataframe(bib_df)


page_names_to_funcs = {
    "Robot Description": robot_description,
    "Controller Description": project_description,
    "Bibliography": project_bibliography
}

st.markdown(TITTLE)

home_expander = st.expander("Introduction", expanded=True)
home_expander.markdown("""
    ### Welcome to the Hinf Controller project for and omnidirectional robot!
    #### This a Python port of my Control Engineering Capstone project, originally done in Matlab.
    
    #### The motivation for this project is to show that almost everything can be done with Python, even a Control System project, area that is dominated by Matlab. 
       
    * This page will cover the theoretical topics of the project.
    * The topics can be switched on the sidebar.  
    * For playing with project data, choose the corresponding page on the top of the sidebar.
""")
home_expander.markdown("---")
st.sidebar.markdown(TITTLE)
st.sidebar.markdown("## Topics")

selected_page = st.sidebar.radio("Select a topic", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()


