import streamlit as st
import mysql.connector
from mysql.connector import Error
import os
import hashlib
# Database configuration
DB_CONFIG = {
    'host': os.environ["DB_HOST"],
    'user': os.environ["DB_USER"],
    'password': os.environ["DB_PASSWORD"],
    'database': os.environ["DB_NAME"]
}

st.set_page_config(
    page_title="English Learning App",
    page_icon="📚",
    layout="centered"
)

HIDE_ST_STYLE = """
    <style>
    div[data-testid="stToolbar"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    div[data-testid="stDecoration"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    #MainMenu {
        visibility: hidden;
        height: 0%;
    }
    header {
        visibility: hidden;
        height: 0%;
    }
    footer {
        visibility: hidden;
        height: 0%;
    }
    div[data-testid=stSidebar] {
        background-color: #3fa0bf;  /* 淡い青色 */
        color: #000000;  /* 文字色を黒に設定（必要に応じて変更） */
    }
    .appview-container .main .block-container {
        padding-top: 1rem;
        padding-right: 3rem;
        padding-left: 3rem;
        padding-bottom: 1rem;
    }
    .reportview-container {
        padding-top: 0rem;
        padding-right: 3rem;
        padding-left: 3rem;
        padding-bottom: 0rem;
    }
    header[data-testid="stHeader"] {
        z-index: -1;
    }
    div[data-testid="stToolbar"] {
        z-index: 100;
    }
    div[data-testid="stDecoration"] {
        z-index: 100;
    }
    </style>
"""

st.markdown(HIDE_ST_STYLE, unsafe_allow_html=True)

def init_db():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create progress table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_progress (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                score INT DEFAULT 0,
                total_correct INT DEFAULT 0,
                total_attempts INT DEFAULT 0,
                last_session TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Create history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                japanese_text TEXT,
                english_text TEXT,
                is_correct BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_settings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                difficulty VARCHAR(10) DEFAULT 'normal',
                themes JSON,
                use_dummy_words BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        conn.commit()
    except Error as e:
        st.error(f"Database error: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
# User authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        password_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
            (username, password_hash)
        )
        
        user_id = cursor.lastrowid
        cursor.execute(
            "INSERT INTO user_progress (user_id) VALUES (%s)",
            (user_id,)
        )
        
        conn.commit()
        return True
    except Error as e:
        st.error(f"Error creating user: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def verify_user(username, password):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        password_hash = hash_password(password)
        cursor.execute(
            "SELECT * FROM users WHERE username = %s AND password_hash = %s",
            (username, password_hash)
        )
        
        user = cursor.fetchone()
        return user
    except Error as e:
        st.error(f"Error verifying user: {e}")
        return None
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# Login/Signup UI
def show_auth_ui():
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                user = verify_user(username, password)
                if user:
                    st.session_state.user = user
                    st.session_state.authenticated = True
                    st.success("Successfully logged in!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    with tab2:
        with st.form("signup_form"):
            new_username = st.text_input("Choose Username")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit = st.form_submit_button("Sign Up")
            
            if submit:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters long")
                else:
                    if create_user(new_username, new_password):
                        st.success("Account created successfully! Please log in.")
                    else:
                        st.error("Username already exists")

def create_sidebar():
    with st.sidebar:
        if st.session_state.get("authenticated"):
            st.title("メニュー / Menu")
            # modes = [
            #     {"label": "学習モード / Study Mode", "key": "study"},
            #     {"label": "チャットモード / Chat Mode", "key": "chat"},
            #     {"label": "英会話モード / Speaking Talking Mode", "key": "stt"},
            #     {"label": "育成モード / Nurturing Mode", "key": "Monster"}
            # ]
            modes = [
                {"label": "学習モード / Study Mode", "key": "study"},
                {"label": "チャットモード / Chat Mode", "key": "chat"},
                {"label": "英会話モード / Speaking Talking Mode", "key": "stt"}
            ]

            for mode in modes:
                if st.button(
                    mode["label"],
                    type="primary" if st.session_state.get("mode", "study") == mode["key"] else "secondary"
                ):
                    st.session_state.mode = mode["key"]
                    st.rerun()
            # Mode selection
            # st.write("### モード選択 / Mode Selection")
            # if st.button("学習モード / Study Mode", 
            #             type="primary" if st.session_state.get("mode", "study") == "study" else "secondary"):
            #     st.session_state.mode = "study"
            #     st.rerun()
            
            # if st.button("チャットモード / Chat Mode",
            #             type="primary" if st.session_state.get("mode", "study") == "chat" else "secondary"):
            #     st.session_state.mode = "chat"
            #     st.rerun()

            # if st.button("英会話モード / Speaking Talking Mode",
            #             type="primary" if st.session_state.get("mode", "study") == "stt" else "secondary"):
            #     st.session_state.mode = "stt"
            #     st.rerun()

            # if st.button("育成モード / Nurturing Mode",
            #             type="primary" if st.session_state.get("mode", "study") == "Monster" else "secondary"):
            #     st.session_state.mode = "Monster"
            #     st.rerun()
                

def create_logoutsidebar():
    with st.sidebar:
        # Show user info if logged in
        if st.session_state.get("authenticated"):
            st.write("---")
            st.write(f"### ユーザー / User")
            st.write(f"🧑‍💻 {st.session_state.user['username']}")
            
            if st.button("ログアウト / Logout"):
                st.session_state.clear()
                st.rerun()

                
def main():
    st.session_state.image_flag = None
    # Initialize database
    init_db()
    
    # Set default mode if not set
    if 'mode' not in st.session_state:
        st.session_state.mode = "study"
    if 'Flag_serachimag' not in st.session_state:
        st.session_state.Flag_serachimag = False
    if 'image_results' not in st.session_state:
        st.session_state.image_results = []
    if 'learnimage' not in st.session_state:
            st.session_state.learnimage = None
    if 'generate_toggle' not in st.session_state:
        st.session_state.generate_toggle = False
    # if 'last_mode' not in st.session_state:
    #     st.session_state.last_mode = None

    # Check if mode has changed
    # if st.session_state.mode != st.session_state.last_mode:
    #     reset_mode_specific_state(st.session_state.mode)
    #     st.session_state.last_mode = st.session_state.mode
    
    # Create sidebar
    create_sidebar()
    
    # Check authentication
    if 'authenticated' not in st.session_state or not st.session_state.authenticated:
        show_auth_ui()
    else:
        # Display appropriate UI based on mode
        if st.session_state.mode == "study":   
            from contents import study_mode
            study_mode.render()
            #create_app_ui()
            
        elif st.session_state.mode == "chat":
            from contents import chat_mode
            chat_mode.render()
            #create_chat_ui()

        elif st.session_state.mode == "stt":
            from contents import stt_mode
            stt_mode.render()
            #STT()

        # elif st.session_state.mode == "Monster":
        #     from contents import Monster_mode
        #     Monster_mode.render()
    
    create_logoutsidebar()


if __name__ == "__main__":
    main()

