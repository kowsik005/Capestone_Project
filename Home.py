import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(page_title="FAKEBUSTER AI Assistant", layout="wide")

# --- HEADER / HERO SECTION ---
st.markdown(
    """
    <div style="text-align:center;">
        <h1>FAKEBUSTER - AI Assistant for Media Verification</h1>
        <h3>Your AI-powered partner for detecting misinformation, deepfakes & fakenews</h3>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")

# --- MODULE SELECTION ---
st.markdown("### Choose a Module to Get Started")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Fakenews Analysis", use_container_width=True):
        st.switch_page("pages/Fakenews.py")

with col2:
    if st.button(" Deepfake Photo Analysis", use_container_width=True):
        st.switch_page("pages/Deepfake.py")

st.write("---")

st.write("---")

# --- FEEDBACK & CONTACT SECTION ---
st.markdown("##  Share Your Feedback")

with st.form("feedback_form"):
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    comments = st.text_area("Your Feedback / Suggestions")
    submitted = st.form_submit_button("Submit Feedback")
    if submitted:
        st.success(" Thank you for your feedback! We will get back to you soon.")

st.write("---")

# --- CONTACT INFO ---
st.markdown(
    """
    ### Contact Us  
    -  Email: fakebuster.ai@gmail.com  
    -  Phone: +91 8870641534  
    -  Website: [Fakebuster Project]  
    """
)

# --- FOOTER ---
st.write("---")
st.markdown(
    """
    <div style="text-align:center; color:gray;">
        © 2025 FAKEBUSTER | All Rights Reserved
    </div>
    """,
    unsafe_allow_html=True
)
