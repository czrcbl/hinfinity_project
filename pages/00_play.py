import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from PIL import Image
from src.simulation import trajectory_sim, Simulation
import src.system_data as sysdat
from src.utils import load
from src.project import Controller
# Specify canvas parameters in application
import numpy as np
import json
from pathlib import Path
from src.vis_data import plot_pose, triple_plot2, plot_poles
from src import system_data as sysdat

FIELD_DIM = 10
CANVAS_DIM = 700
CONV_FACTOR = FIELD_DIM / CANVAS_DIM

TITTLE = "# 🎮 Play with the Simulation"

@st.cache
def load_controllers():

    import sys
    sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "src"))
    all_controllers = load('data/controllers_solver-SCS_all.pkl')
    return all_controllers

def main():

    st.markdown(TITTLE)
    st.sidebar.markdown(TITTLE)

    # Specify canvas parameters in application
    # drawing_mode = st.sidebar.selectbox(
    #     "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
    # )

    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("freedraw", "point", "line", "rect", "circle")
    )
    
    # stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_width = 1
    # if drawing_mode == 'point':
    #     point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
    point_display_radius = 3
    # stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    stroke_color = '#000000'
    # bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    bg_color = "#FFFFFF"
    # bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

    # realtime_update = st.sidebar.checkbox("Update in realtime", True)
    realtime_update = True


        

    st.markdown("""
        ### Path Drawing Canvas

        Draw the robot path, the side of the canvas has a dimension of 10 meters.

        Choose the drawing tool on the side bar, the drawing can be reset on the trash can below the canvas.
    """)
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        # background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=700,
        width=700,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        display_toolbar=True,
        key="canvas",
    )

    # Do something interesting with the image data and paths
    # if canvas_result.image_data is not None:
    #     st.image(canvas_result.image_data)
    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")
        # st.dataframe(objects)


    st.markdown("""
        ### Simulation parameters
    """)

    v_nav = st.number_input("Reference Velocity (m/s)", value=0.6)


    all_controllers = load_controllers()
    controllers = [c for c in all_controllers if not c.poles is None]
    settling_time = np.array([c.stepinfo.SettlingTime for c in controllers])
    over_shoot = np.array([c.stepinfo.Overshoot for c in controllers])
    u_var = np.array([c.u_max_var for c in controllers])
    norm = np.array([c.norm for c in controllers])
    

    st.markdown("""
        #### Controller Selection Parameters

        Here follows a list of many projected controllers for various center and radius configurations, you can select one based on its performance to use for the simulation.
    """)
    cols = st.columns(3)
    max_st = cols[0].number_input("Max settling time (s)", value=settling_time.max())
    max_norm = cols[1].number_input("Max norm", value=norm.max())
    max_u_var = cols[2].number_input("Max Control Signal Variation (V)", value=u_var.max())
    
    idx = np.where((settling_time < max_st) & (u_var < max_u_var) & (norm < max_norm))[0]
    filt_controlers = sorted([controllers[ix] for ix in idx], key=lambda c: c.norm)
    
    c = st.selectbox("Filtered Controllers", filt_controlers,  format_func=lambda c: f"Stataus: {c.status} - Norm: {c.norm:.2f} - SettlingTime: {c.stepinfo.SettlingTime:.2f} - Overshoot: {c.stepinfo.Overshoot:.2f} - Umax_var: {c.u_max_var:.2f}")
    # c = controllers[idx[-1]]

    st.markdown("The blue circle represents the constraint on the poles position.")
    fig = plot_poles(c.poles, c.q, c.r)
    st.pyplot(fig)
    st.markdown("""
        ### Result Plots
    """)

    if len(objects) == 0:
        st.write("Draw Trajectory First")
        return
    points = []
    for idx, line in objects.iterrows():
        if line.type == "line":
            points.append([(line.left + line.x1) * CONV_FACTOR, ( line.top + line.y1) * CONV_FACTOR, 0])
            points.append([(line.left + line.x2) * CONV_FACTOR, ( line.top + line.y2) * CONV_FACTOR, 0])
            points.insert(1, [(points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2, 0])
        elif line.type == "circle" and line.radius < 10:
            points.append([line.left * CONV_FACTOR, line.top * CONV_FACTOR, 0])
        elif line.type == "rect":
            points.append([line.left * CONV_FACTOR, line.top * CONV_FACTOR, 0])
            points.append([(line.left + line.width) * CONV_FACTOR, line.top * CONV_FACTOR, 0])
            points.append([(line.left + line.width) * CONV_FACTOR , (line.top + line.height) * CONV_FACTOR , 0])
            points.append([line.left * CONV_FACTOR, (line.top + line.height)* CONV_FACTOR , 0])
        elif line.type == "circle":
            for angle in range(-180, 180, 30):
                points.append([(line.left + line.radius + line.radius *  np.cos(angle * np.pi / 180)) * CONV_FACTOR , (line.top  + line.radius * np.sin(angle * np.pi / 180)) * CONV_FACTOR, 0])
        elif line.type == "path":
            # st.write(line.path)
            for point in json.loads(line.path.replace("\'", "\""))[::3]:
                # st.write(point)
                points.append([point[1] * CONV_FACTOR, point[2] * CONV_FACTOR, 0])


    
    


    sim = Simulation(sysdat.csys, sysdat.Ts)
    traj = np.array(points)
    traj[:, 1] = FIELD_DIM - traj[:, 1]
    # st.write(traj)
    states_vec, control_signal_vec, ref_vec, pose_vec = trajectory_sim(traj, v_nav, sim, c.K)
    
    fig = plot_pose(pose_vec, traj, size=10)
    st.pyplot(fig)
    

    fig = triple_plot2(states_vec, ref_vec, 'States', 'Reference', figsize=(8, 10))
    st.pyplot(fig)


    def create_robot_annimations(pose_vec):
        import cv2
        import os
        import subprocess
        os.makedirs("temp", exist_ok=True)
        cf = 2048/10
        size = (2048, 2048, 3)
        frame_template = np.zeros(size, dtype=np.uint8)
        video_frames = []
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        temp_video_path = "temp/sim_video.mp4"
        conv_video_path = "temp/sim_video_conv.mp4"
        writer = cv2.VideoWriter(temp_video_path, fourcc, 1 / sysdat.Ts, (2048, 2048))
        for v in pose_vec:
            frame = frame_template.copy()
            print(v)
            frame = cv2.circle(frame, (int(cf * v[0]), int(cf * v[1])), 100, (255, 255, 255), -1)
            # video_frames.append(frame)
            writer.write(frame)
        writer.release()

        convertedVideo = "./testh264.mp4"
        subprocess.call(args=f"ffmpeg -y -i {temp_video_path} -c:v libx264 {conv_video_path}".split(" "))
        # video = np.array(video_frames)
        # st.write(video.shape)
        st.video(conv_video_path, format="MPEG-4")

    create_robot_annimations(pose_vec)



main()