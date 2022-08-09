import streamlit as st

st.set_page_config(page_title="Home Page", layout="wide")

def robot_description():

    st.markdown("## Robot Description")
    
    st.image('figures/diagrama_dinamica_english.png', width=600)


def project_description():

    st.markdown("## Project Description")
    
    st.image('figures/forma_geral.png', width=600)


# def page2():
#     st.markdown("# Page 2 ❄️")
#     st.sidebar.markdown("# Page 2 ❄️")

# def page3():
#     st.markdown("# Page 3 🎉")
#     st.sidebar.markdown("# Page 3 🎉")

page_names_to_funcs = {
    "Robot Description": robot_description,
    "Project Description": project_description,
}

st.markdown("# Home Page")

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
st.sidebar.markdown("# Home Page")
st.sidebar.markdown("---")
st.sidebar.markdown("## Topics")

selected_page = st.sidebar.radio("Select a topic", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()


