import streamlit as st

#---PAGE SET-UP -----

know_me_page  = st.Page(page = "Views/know_me.py", title='About me', icon=":material/account_circle:", default=True)

project_1_page  = st.Page(page = "Views/Guide.py", title='Guide', icon="🤖")

project_2_page  = st.Page(page = "Views/calculator.py", title='Option Price Calculator', icon=":material/bar_chart:")



# pg = st.navigation(pages=[know_me_page, project_1_page, project_2_page])

pg = st.navigation({
    "Info":[know_me_page],
    "Projects":[project_1_page, project_2_page],
})

pg.run()


st.sidebar.text('Made with 💟 by Avani')

