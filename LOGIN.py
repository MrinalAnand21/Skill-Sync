import streamlit as st
import hashlib
import base64
from time import sleep
import psycopg2
from psycopg2 import OperationalError
import bcrypt


#DB connect 
def connect_db():
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="pgAdmin",
            host="localhost",
            port="5432"
        )
        return conn
    except OperationalError as e:
        print(f"Error connecting to the database: {e}")
        return None

connect_db()

# User registration function
def register_user(username, password):
    conn = connect_db()
    cursor = conn.cursor()
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s);", (username, hashed))
        conn.commit()
        st.success("User Registered Successfully!")
    except Exception as e:
        st.error("Registration Failed. Username Already Exists!")
    finally:
        conn.close()

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Use cache_data instead of experimental_memo
@st.cache_data 
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load your image (ensure the path is correct)
img = get_img_as_base64("static_files/bg3.jpg")

# CSS for background image and styles
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{img}");
    background-size: cover;
}}
.custom-header {{
    font-size: 26px;  /* Font size */ 
    border-radius: 10px;  /* Rounded corners */
    text-align: center;  /* Center text */
    margin-top: 20px;  /* Space above the header */
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);  /* Optional: add shadow for depth */
    font-weight: bold;
}}
.title {{
    background-color: rgba(0, 0, 0, 0.5);
    padding: 10px;
    border-radius: 10px;
    text-align: center;  /* Center the title */
    font-family: 'Candara', sans-serif;  /* Use Candara font */
    font-size: 50px;  /* Title font size */
    font-weight: bold;  /* Make font bold */
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);  /* Optional: add shadow for depth */
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the application title
st.markdown('<div class="title">🎓 SkillSync 📚<br><span style="font-size: 0.5em; vertical-align: baseline;">Personalized Learning with Generative AI</span></div>', unsafe_allow_html=True)

# Use custom styling for the header
st.markdown('<div class="custom-header">User Login</div>', unsafe_allow_html=True)

# Initialize user storage
if "users" not in st.session_state:
    st.session_state.users = {}
if "store" not in st.session_state:
    st.session_state.store = {}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

# User login
username = st.text_input("Username")
password = st.text_input("Password", type="password")

# Function to check username and password
def check(username, password):
    conn = connect_db()
    
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT password FROM users WHERE username = %s;", (username,))
        result = cursor.fetchone()
        
        if result:
            # Fetch the stored hashed password, which is in hexadecimal format
            stored_hashed_password_hex = result[0]
            # Convert the hex string to bytes
            stored_hashed_password = bytes.fromhex(stored_hashed_password_hex[2:])  # Skip the '\x' part
            
            # Check the provided password against the stored hashed password
            if bcrypt.checkpw(password.encode('utf-8'), stored_hashed_password):
                return True
            else:
                st.error("Entered Incorrect Password ")
                return False
        else:
            st.error("Username Not Found")
            return False  # Username not found
    except Exception as e:
        print(f"Error during database operation: {e}")
        st.error("An error occurred while checking the credentials.")
        return False
    finally:
        cursor.close()
        conn.close()

col1,col2 = st.columns([7, 1])  # Create two columns
with col1:
    if st.button("Log In"):
        try:
            # Check for special test user login
            if username == "test_user" and password == "test_user":
                st.session_state.logged_in = True
                st.session_state.username = username  # Store username in session state
                st.success("Logged in successfully as test user!")
                sleep(0.5)
                st.session_state.page = "USER_PAGE"  # Redirect to content
                st.session_state.custom_pdf = False
                st.switch_page("pages/USER_PAGE.py")
            elif check(username,password) or username in st.session_state.users and st.session_state.users[username] == hashlib.sha256(password.encode()).hexdigest():
                st.session_state.logged_in = True
                st.session_state.username = username  # Store username in session state
                st.success("Logged in successfully!")
                sleep(0.5)
                st.session_state.page = "USER_PAGE"  # Redirect to content
                st.session_state.custom_pdf = False
                st.switch_page("pages/USER_PAGE.py")
        except KeyError as e:
            st.error(f"Error: {str(e)} - User data may not be initialized.")
            st.session_state.users = {}  # Initialize users if needed

# with col2:    
# Sign Up button logic  
if st.button("New User ? Signup"):
    if username == "" or password == "":
        st.error("Please provide both username and password!")
    else:
        registration_success = register_user(username, password)
        if registration_success:
            # Only store the user in session state if registration was successful
            st.session_state.users[username] = hash_password(password)
            st.session_state.store[username] = {}
            st.success("User created successfully! You can now log in.")
