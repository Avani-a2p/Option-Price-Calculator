import streamlit as st
import re
import requests

def is_valid_email(email):
    # Regular expression for a valid email
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_pattern, email) is not None
WEBHOOK_URL = st.secrets["WEBHOOK_URL"]

def contact_form(): 
    with st.form("contact form"):
        name = st.text_input("First Name")
        email = st.text_input("Email Adress")
        message = st.text_input("Your Message")
        submit_button = st.form_submit_button("Submit")
    
        
        if submit_button:
            # st.success('Message sent successfully !')
            if not WEBHOOK_URL:
                st.error('Email service is not working. Please try later.',icon="⏱️")
                st.stop()
            if not name:
                st.error('Please provide your name.',icon="👩‍💻")
                st.stop()
            if not email:
                st.error('Please provide your email address.',icon="📩")
                st.stop()
            if not is_valid_email(email):
                st.error('Please provide proper email address.', icon="📧")
                st.stop()
            if not message:
                st.error('Please provide message.', icon="💭")
                st.stop()

            data = {"email":email, "name":name, "message":message}
            response = requests.post(WEBHOOK_URL, json=data)

            if response.status_code == 200:
                st.success('Your message sent successfully 🥳!', icon ="🚀")
            else:
                st.error('There was an error sending your message !', icon='⚠️')
            