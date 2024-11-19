import streamlit as st
from forms.contact import contact_form

@st.dialog('Share Your thoughts')
def show_contact_form():
    contact_form()

col1, col2 = st.columns(2, gap="small", vertical_alignment="center")

with col1:
    st.image('./assets/Profile_image-modified.png',width=230)
with col2:
    st.title('Avani Pandya', anchor=False)
    st.text("Exploring Finance with Streamlit !")
    
    if st.button('✉️ Share Your Thoughts!'):
        show_contact_form()