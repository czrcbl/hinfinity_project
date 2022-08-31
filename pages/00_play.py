import json
from dataclasses import field
from pathlib import Path
import cv2

# Specify canvas parameters in application
import numpy as np
import pandas as pd
import src.system_data as sysdat
import streamlit as st
from PIL import Image
from src import system_data as sysdat
from src.simulation import Simulation, trajectory_sim
from src.utils import load, load_controllers_json
from src.vis_data import plot_poles, plot_pose, triple_plot2, SimulationDrawer
from streamlit_drawable_canvas import st_canvas

DEFAULT_FIELD_WH = 5
DEFAULT_CANVAS_WH = 500

TITTLE = "# 🎮 Play with the Simulation"


@st.cache
def load_controllers():

    import sys

    sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "src"))
    all_controllers = load_controllers_json("data/controlers_test-q=0.1-r=0.1.json")
    return all_controllers


def generate_start_frame(field_wh):
    """Generate grid image for the canvas"""
    WH = 1000
    grid_dist = int(WH / field_wh)
    frame_template = (255 * np.ones((WH, WH, 3))).astype(np.uint8)

    for start in range(0, WH, grid_dist):
        frame_template = cv2.line(frame_template, (start, 0), (start, WH), (0, 0, 0), 1)
        frame_template = cv2.line(frame_template, (0, start), (WH, start), (0, 0, 0), 1)

    return frame_template


def objects2points(objects, CONV_FACTOR):

    points = []
    for idx, line in objects.iterrows():
        if line.type == "line":
            points.append(
                [
                    (line.left + line.x1) * CONV_FACTOR,
                    (line.top + line.y1) * CONV_FACTOR,
                    0,
                ]
            )
            points.append(
                [
                    (line.left + line.x2) * CONV_FACTOR,
                    (line.top + line.y2) * CONV_FACTOR,
                    0,
                ]
            )
            points.insert(
                1,
                [
                    (points[0][0] + points[1][0]) / 2,
                    (points[0][1] + points[1][1]) / 2,
                    0,
                ],
            )
        elif line.type == "circle" and line.radius < 10:
            points.append([line.left * CONV_FACTOR, line.top * CONV_FACTOR, 0])
        elif line.type == "rect":
            points.append([line.left * CONV_FACTOR, line.top * CONV_FACTOR, 0])
            points.append(
                [(line.left + line.width) * CONV_FACTOR, line.top * CONV_FACTOR, 0]
            )
            points.append(
                [
                    (line.left + line.width) * CONV_FACTOR,
                    (line.top + line.height) * CONV_FACTOR,
                    0,
                ]
            )
            points.append(
                [line.left * CONV_FACTOR, (line.top + line.height) * CONV_FACTOR, 0]
            )
        elif line.type == "circle":
            for angle in range(-180, 180, 30):
                points.append(
                    [
                        (
                            line.left
                            + line.radius
                            + line.radius * np.cos(angle * np.pi / 180)
                        )
                        * CONV_FACTOR,
                        (line.top + line.radius * np.sin(angle * np.pi / 180))
                        * CONV_FACTOR,
                        0,
                    ]
                )
        elif line.type == "path":
            # st.write(line.path)
            for point in json.loads(line.path.replace("'", '"'))[::3]:
                # st.write(point)
                points.append([point[1] * CONV_FACTOR, point[2] * CONV_FACTOR, 0])

    return points


@st.cache(allow_output_mutation=True)
def do_simulation(traj, v_nav, sim, K):

    states_vec, control_signal_vec, ref_vec, pose_vec = trajectory_sim(
        traj, v_nav, sim, K
    )
    return states_vec, control_signal_vec, ref_vec, pose_vec


def main():

    st.markdown(TITTLE)
    st.sidebar.markdown(TITTLE)

    stroke_width = 1
    point_display_radius = 3
    stroke_color = "#000000"
    bg_color = "#FFFFFF"
    realtime_update = True

    # tab_canvas, tab_controller, tab_video, tab_plots = st.tabs(
    #     ["Path Drawing", "Controller Data", "Simulation Video", "Simulation Plots"]
    # )

    # with tab_canvas:

    st.sidebar.markdown(
        """
        ### Simulation parameters
    """
    )

    v_nav_reference = st.sidebar.number_input("Reference Velocity (m/s)", value=0.6)

    st.sidebar.markdown(
        """
        #### Controller Selection Parameters

        Here follows a list of many projected controllers for various center and radius configurations, you can select one based on its performance to use for the simulation.
    """
    )

    only_optimal = st.sidebar.checkbox("Only show optimal controllers", value=True)
    all_controllers = load_controllers()
    controllers = [
        c
        for c in all_controllers
        if c.poles is not None and (not only_optimal or c.status == "optimal")
    ]
    settling_time = np.array([c.stepinfo.SettlingTime for c in controllers])
    over_shoot = np.array([c.stepinfo.Overshoot for c in controllers])
    u_var = np.array([c.u_max_var for c in controllers])
    norm = np.array([c.norm for c in controllers])

    max_st = st.sidebar.number_input("Max settling time (s)", value=settling_time.max())
    max_norm = st.sidebar.number_input("Max Hinf norm", value=norm.max())
    max_u_var = st.sidebar.number_input(
        "Max Control Signal Variation (V)", value=u_var.max()
    )

    idx = np.where((settling_time < max_st) & (u_var < max_u_var) & (norm < max_norm))[
        0
    ]
    filt_controlers = sorted([controllers[ix] for ix in idx], key=lambda c: c.norm)

    c = st.sidebar.selectbox(
        f"Filtered Controllers (total = {len(filt_controlers)})",
        filt_controlers,
        format_func=lambda c: f"Norm: {c.norm:.2f}, Center: {c.q:.2f}, Radius: {c.r:.2f}",
    )

    tab_controller = st.expander("Controller Data", expanded=False)

    tab_controller.text(str(c))

    tab_controller.text(str(c.stepinfo))

    tab_controller.markdown(
        "The blue circle represents the constraint on the poles position."
    )
    fig = plot_poles(c.poles, c.q, c.r)
    tab_controller.pyplot(fig)

    st.markdown(
        """
        ### Path Drawing Canvas

        The drawing can be reset on the trash can below the canvas.
    """
    )
    # lock_drawing = st.checkbox("Lock Drawing")

    cols = st.columns([2, 2])
    field_wh = cols[0].number_input(
        "Field Width and Height (m)",
        value=DEFAULT_FIELD_WH,
        min_value=3,
        max_value=10,
    )
    # drawing_mode = cols[1].selectbox(
    #     "Drawing tool:", ("freedraw", "point", "line", "rect", "circle")
    # )
    drawing_mode = "freedraw"

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.fromarray(generate_start_frame(field_wh)),
        update_streamlit=realtime_update,
        height=DEFAULT_CANVAS_WH,
        width=DEFAULT_CANVAS_WH,
        drawing_mode=drawing_mode,
        # initial_drawing=st.session_state["json_data"]
        # if "json_data" in st.session_state and lock_drawing
        # else None,
        point_display_radius=point_display_radius if drawing_mode == "point" else 0,
        display_toolbar=True,
        key="canvas",
    )

    objects = None

    if canvas_result.json_data is not None:
        objects = pd.json_normalize(
            canvas_result.json_data["objects"]
        )  # need to convert obj to str because PyArrow
        for col in objects.select_dtypes(include=["object"]).columns:

            objects[col] = objects[col].astype("str")
        # st.dataframe(objects)

    if objects is None or len(objects) == 0:
        st.warning("Draw Trajectory First")
        st.stop()

    points = objects2points(objects, field_wh / DEFAULT_CANVAS_WH)

    with st.spinner():
        sim = Simulation(sysdat.csys, sysdat.Ts)

        trajectory = np.array(points)
        trajectory[:, 1] = field_wh - trajectory[:, 1]

        # st.write(traj)

        states_vec, control_signal_vec, ref_vec, pose_vec = do_simulation(
            trajectory, v_nav_reference, sim, c.K
        )

        sim_drawer = SimulationDrawer(
            pose_vec, states_vec, image_wh=2048, field_wh=field_wh
        )

        conv_video_path = sim_drawer.generate_video(skip_frames=5)

    st.markdown(
        """
        ## Robot Movement Video """
    )

    st.video(conv_video_path, format="MPEG-4")

    fig = plot_pose(pose_vec, trajectory, size=field_wh)
    st.pyplot(fig)

    fig = triple_plot2(states_vec, ref_vec, "States", "Reference", figsize=(8, 10))
    st.pyplot(fig)


main()
