import streamlit as st
import pandas as pd

st.set_page_config(page_title="Home Page")

TITLE = "# 📚 Home Page - Theory"
SUBTITLE = r"$\mathcal{H}_\infty$ Control for an Omnidirectional Robot"


def robot_description():

    st.markdown("## Robot Description")

    st.write(
        "A diagram of the forces and torques acting on the robot's center of mass."
    )
    st.image("figures/diagrama_dinamica_english.png", width=600)
    st.markdown(
        r"""
    The forces and torque at the robot's center of mass are given by the following equations
    (Conceição, Moreira & Costa, 2009; Conceição, Correia & Martinez, 2016):
    $$
    M\dot{v}(t) = F_v(t) - F_{va}(t) \\
    M\dot{v_n}(t) = F_{v_n}(t) - F_{v_na}(t) \\
    J\dot{\omega}(t) = \Gamma(t) - \Gamma_a(t)
    $$

    where $F_v(t)$, $F_{v_n}(t)$ and $\Gamma(t)$ are the resultant forces and torque produced by the traction forces of the wheels.

    Each wheel is driven by an identical DC motor.
    Letting $r_i$ be the radius of wheel $i$, its traction force is given by:
    $$
    F_{mi}(t) = \frac{T_i(t)}{r_i}
    $$

    $T_i(t)$ is the torque produced by motor $i$, which is related to the input voltage by:
    $$
    u_i(t) =L_{ai}\dfrac{di_{ai}(t)}{dt}+R_{ai}i_{ai} + k_{vi}\omega_{mi}(t), \\
    T_i = l_ik_{ti}i_{ai}.
    $$

    $L_{ai}$ and $R_{ai}$ are the inductance and the resistance of the motor winding, respectively.
    $k_{ti}$ is the motor's torque constant, $k_{vi}$ is the back-EMF constant and $l_i$ is the gear reduction ratio.

    The linear velocity of wheel $i$ ($v_{ri}$) is related to its rotor's angular velocity ($\omega_{mi}$) by:
    $$
    v_{ri} = \frac{r_i\omega_{mi}}{l_i}.
    $$

    Four friction phenomena were considered in the model (Olsson et al., 1998): Coulomb friction
    (dynamic friction); viscous friction, modeled as a linear function of the
    velocity; stiction, the friction acting while the system is at rest;
    and the Stribeck effect, the transition from static to dynamic
    friction at low velocities.

    The next figure illustrates the four effects.
    """
    )
    st.image("figures/frictions.png", width=600)

    st.markdown(
        r"""
    The frontal and normal friction forces and the friction torque are given by:

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
    in each direction. The Coulomb friction coefficients are
    $C_v$, $C_{v_n}$ and $C_\omega$.
    The stiction forces and torque are $F_{sv}$, $F_{sv_{n}}$
    and $F_{s\omega}$, and the Stribeck velocities are $v_{sv}$, $v_{sv_n}$ and
    $\omega_{s\omega}$. $\delta_{s}$ is an empirical shape parameter.

    From the equations of the dynamic model, the following state-space representation was derived:
    $$
    \dot{\xi}(t) = A_c\xi(t) + B_cu(t) + (K_c + GE(t))sign(\xi), \\
    y(t) = C_c\xi(t) + D_cu(t),
    $$
    where the vector $u(t) = \begin{bmatrix} u_1(t) & u_2(t) & u_3(t) 
    \end{bmatrix}^T$ corresponds to the input voltage of the motors.

    ## Discrete Time Model

    The plant was discretized using a zero-order hold (ZOH) with a sampling period of 60 ms
    (Åström & Wittenmark, 2013).
    In the process, the nonlinear friction terms were modeled as a disturbance $w_{k}$
    acting directly on the states. The resulting state-space representation is:
    $$
    \xi_{k+1} = A_d\xi_{k} + B_du_{k} + w_{k}, \\
    y_{k} = C_d\xi_{k} + D_du_{k}
    $$

    ## Augmented Model

    To achieve zero steady-state tracking error for constant references, integral action is added to the plant.

    In the $\mathcal{H}_\infty$ design, this is done by adopting the following augmented state vector:
    $$
    x_{k} = \begin{bmatrix}
    e_{k} \\ \eta_{k}
    \end{bmatrix},
    $$
    where:
    $$
    e_{k} = y_{k} - r_{k}, \\
    \eta_{k + 1} = \eta_{k} + e_{k},
    $$
    and $r_k$ is the reference signal.
    The augmented state-space representation is then:
    $$
    x_{k+1} = A_{aug}x_{k} + B_{aug}u_{k} + B_ww_k, \\
    y_{k} = C_{aug}x_{k} + D_{aug}u_{k},
    $$
    where the augmented matrices, written in terms of the discrete-time matrices,
    are:
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
    $I$ and $0$ denote the $3\times3$ identity and zero matrices.
    """
    )


def project_description():

    st.markdown("## Controller Description")

    st.image("figures/forma_geral.png", width=600)

    st.markdown(
        r"""
    
    ## $\mathcal{H}_\infty$ Controller


    This formulation uses the generalized plant form shown in the figure above (Zhou, Doyle & Glover, 1996).
    In this form, $u$ is the vector of control inputs, $y$ is the vector of measured outputs, $w$ is the vector of external disturbances and $z$ is the vector of controlled outputs.
    The goal is to design a controller that minimizes the $\mathcal{H}_\infty$ norm of $H_{zw}(z)$, the transfer function from the input $w$ to the output $z$.

    The corresponding state-space representation is:
    $$
    x_{k+1} = Ax_{k} + B_1w_{k} + B_2u_{k}, \\
    z_{k} = C_1x_{k} + D_{11}w_{k} + D_{12}u_{k}, \\
    y_{k} = C_2x_{k} + D_{21}w_{k} + D_{22}u_{k}.
    $$

    In this work, the $z$ signal is defined as follows:
    $$
    z =  \begin{bmatrix}
    y \\ 
    u
    \end{bmatrix},
    $$
    i.e., the controller should reduce the effect of the disturbances on both the output and the control signal.

    A state-feedback controller $u_k = Kx_k$ that guarantees $\lVert H_{zw} \rVert_\infty < \gamma$ exists if the following LMI is feasible
    (Gahinet & Apkarian, 1994; De Oliveira, Geromel & Bernussou, 2002):
    
    $$
    \begin{bmatrix}
    P & AP+B_2L & B_1 & 0 \\
    PA^T+L^TB_2^T & P & 0 & PC_1^T + L^TD_{12}^T \\
    B_1^T & 0 & I & D_{11}^T \\
    0 & C_1P + D_{12}L & D_{11} & \gamma^2I
    \end{bmatrix} > 0,
    $$
    where the symmetric matrix $P \in \mathbb{R}^{6\times6}$ and the matrix $L \in \mathbb{R}^{3\times6}$ are the optimization variables.

    The controller that minimizes the $\mathcal{H}_\infty$ norm of $H_{zw}(z)$ is obtained by
    solving a convex optimization problem: minimize $\gamma^2$ subject to the LMI above.
    The controller's gain matrix is then:
    $$
    K = LP^{-1}.
    $$

    The closed-loop poles can be placed in a disk of the complex plane with center $q$ on the real axis and radius $r$.
    This is achieved by adding the following LMI to the previous optimization problem
    (Chilali & Gahinet, 1996; Chilali, Gahinet & Apkarian, 1999):
    $$
    \begin{bmatrix}
    -rP & -qP + AP + B_2L \\
    -qP + PA^T + L^TB_2^T& -rP
    \end{bmatrix} < 0 
    $$

    Constraining the pole locations makes it possible to trade off the system's
    transient response against the achieved $\mathcal{H}_\infty$ norm.
    """
    )


def project_bibliography():

    st.markdown("## Bibliography")
    bib_df = pd.read_csv("assets/bibliography.csv", dtype={"Publication Year": str})
    st.dataframe(bib_df)


def main():
    page_names_to_funcs = {
        "Robot Description": robot_description,
        "Controller Description": project_description,
        "Bibliography": project_bibliography,
    }

    st.markdown(TITLE)
    st.markdown(SUBTITLE)

    st.sidebar.markdown(TITLE)
    st.sidebar.markdown(SUBTITLE)

    tabs = st.tabs(
        [
            "Introduction",
            "Robot Description",
            "Controller Description",
            "Test Videos",
            "Bibliography",
        ]
    )

    with tabs[0]:
        # home_expander = st.expander("Introduction", expanded=True)

        st.image("figures/robo_foto.jpg")
        st.markdown(
            """
            ### Welcome to the H∞ control project for an omnidirectional robot!
            #### This is a Python port of my Control Engineering capstone project, originally developed in MATLAB.

            #### The motivation for this project is to show that almost everything can be done with Python, even control system design, a field dominated by MATLAB.
        """
        )
        with open("assets/TFG_Cezar_Lemos.pdf", "rb") as f:
            data = f.read()

        st.download_button(
            "Download the Work PDF (Portuguese)",
            data=data,
            file_name="TFG_Cezar_Lemos.pdf",
        )
        # st.markdown("---")

    with tabs[1]:
        robot_description()

    with tabs[2]:
        project_description()

    with tabs[3]:
        st.markdown(
            r"""
        ## Square Trajectory  
        $V_{navigation} = 0.3 m/s$
        """
        )
        st.video("assets/traj_quad_vnav03.mp4")

        st.markdown(
            r"""
        ## 8 Trajectory  
        $V_{navigation} = 0.3 m/s$
        """
        )
        st.video("assets/traj_8_vnav03_2.mp4")
    with tabs[4]:
        project_bibliography()


if __name__ == "__main__":
    main()
