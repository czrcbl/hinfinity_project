import streamlit as st


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
st.markdown("---")
st.sidebar.markdown("# Home Page")
st.sidebar.markdown("---")
st.sidebar.markdown("## Topics")

selected_page = st.sidebar.radio("Select a topic", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()


