import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import hashlib
import json
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Page configuration
st.set_page_config(page_title="Scholar Sync", page_icon="🎓", layout="wide")

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'users_db' not in st.session_state:
    # Demo users (in production, use real database)
    st.session_state.users_db = {
        'student@college.edu': {
            'password': hashlib.sha256('password123'.encode()).hexdigest(),
            'name': 'Demo Student',
            'branch': 'CSE',
            'semester': '6'
        }
    }
if 'profiles_db' not in st.session_state:
    st.session_state.profiles_db = {}
if 'groups_db' not in st.session_state:
    st.session_state.groups_db = {}
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.model = None
    st.session_state.scaler = None

# Sample training data generator
def generate_sample_data(n_samples=50):
    """Generate synthetic student data for model training"""
    np.random.seed(42)
    data = {
        'programming': np.random.randint(1, 6, n_samples),
        'dsa': np.random.randint(1, 6, n_samples),
        'os': np.random.randint(1, 6, n_samples),
        'dbms': np.random.randint(1, 6, n_samples),
        'cn': np.random.randint(1, 6, n_samples),
        'projects': np.random.randint(0, 6, n_samples),
        'study_time': np.random.choice([0, 1, 2], n_samples),  # 0=Morning, 1=Afternoon, 2=Evening
        'learning_style': np.random.choice([0, 1, 2, 3], n_samples),  # Visual, Auditory, Kinesthetic, Reading
        'goal': np.random.choice([0, 1, 2, 3], n_samples),  # Placement, Research, CGPA, Competitive
        'cgpa': np.random.uniform(6.0, 10.0, n_samples)
    }
    return pd.DataFrame(data)

# Train ML Model
def train_clustering_model(n_clusters=5):
    """Train KMeans clustering model on sample data"""
    # Generate sample data
    df = generate_sample_data(50)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    # Train KMeans
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(X_scaled)
    
    return model, scaler

# Predict group for new student
def predict_group(student_profile, model, scaler):
    """Predict which group a student should belong to"""
    # Convert profile to feature vector
    features = [
        student_profile['programming'],
        student_profile['dsa'],
        student_profile['os'],
        student_profile['dbms'],
        student_profile['cn'],
        student_profile['projects'],
        student_profile['study_time'],
        student_profile['learning_style'],
        student_profile['goal'],
        student_profile['cgpa']
    ]
    
    # Scale and predict
    features_scaled = scaler.transform([features])
    group_id = model.predict(features_scaled)[0]
    
    return int(group_id)

# Send email notification
def send_email_notification(to_email, student_name, group_id, members, goal):
    """Send email to student about their group assignment"""
    # Note: For demo purposes, just show what would be sent
    # In production, configure SMTP with real credentials
    
    subject = "Scholar Sync - Your Study Group Assignment 🎓"
    
    body = f"""
Hi {student_name},

Congratulations! You've been successfully assigned to a study group.

📚 Group Details:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Group ID: {group_id}
Members: {', '.join(members)}
Common Goal: {goal}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your group has been formed based on:
✓ Similar skill levels
✓ Compatible learning styles
✓ Shared academic goals
✓ Matching study preferences

Next Steps:
1. Connect with your group members
2. Set up your first meeting
3. Share contact information
4. Define your study schedule

Good luck and happy learning! 🚀

Best regards,
Scholar Sync Team
    """
    
    return subject, body

# Hash password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Authentication functions
def login_user(email, password):
    """Authenticate user login"""
    if email in st.session_state.users_db:
        if st.session_state.users_db[email]['password'] == hash_password(password):
            st.session_state.logged_in = True
            st.session_state.user_data = {
                'email': email,
                'name': st.session_state.users_db[email]['name'],
                'branch': st.session_state.users_db[email]['branch'],
                'semester': st.session_state.users_db[email]['semester']
            }
            return True
    return False

def register_user(name, email, password, branch, semester):
    """Register new user"""
    if email in st.session_state.users_db:
        return False
    
    st.session_state.users_db[email] = {
        'password': hash_password(password),
        'name': name,
        'branch': branch,
        'semester': semester
    }
    return True

# Main App
def main():
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            margin: 1rem 0;
        }
        .group-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header"><h1>🎓 Scholar Sync</h1><p>Smart Grouping for Smarter Learning</p></div>', unsafe_allow_html=True)
    
    # Check if logged in
    if not st.session_state.logged_in:
        show_login_page()
    else:
        show_dashboard()

def show_login_page():
    """Display login/registration page"""
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    
    with tab1:
        st.subheader("Login to Scholar Sync")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            email = st.text_input("College Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", use_container_width=True):
                if login_user(email, password):
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Try: student@college.edu / password123")
            
            st.info("**Demo Credentials:**\nEmail: student@college.edu\nPassword: password123")
    
    with tab2:
        st.subheader("Create New Account")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            name = st.text_input("Full Name", key="reg_name")
            email = st.text_input("College Email", key="reg_email")
            password = st.text_input("Password", type="password", key="reg_password")
            branch = st.selectbox("Branch", ["CSE", "ECE", "EEE", "Mechanical", "Civil", "IT"], key="reg_branch")
            semester = st.selectbox("Semester", ["1", "2", "3", "4", "5", "6", "7", "8"], key="reg_semester")
            
            if st.button("Register", use_container_width=True):
                if name and email and password:
                    if register_user(name, email, password, branch, semester):
                        st.success("✅ Registration successful! Please login.")
                    else:
                        st.error("❌ Email already exists.")
                else:
                    st.error("❌ Please fill all fields.")

def show_dashboard():
    """Display main dashboard after login"""
    # Sidebar
    with st.sidebar:
        st.markdown(f"### Welcome, {st.session_state.user_data['name']}! 👋")
        st.markdown(f"**Branch:** {st.session_state.user_data['branch']}")
        st.markdown(f"**Semester:** {st.session_state.user_data['semester']}")
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_data = {}
            st.rerun()
    
    # Check if user has profile
    user_email = st.session_state.user_data['email']
    has_profile = user_email in st.session_state.profiles_db
    
    if not has_profile:
        show_profile_form()
    else:
        show_group_dashboard()

def show_profile_form():
    """Display student profile form"""
    st.subheader("📋 Complete Your Profile")
    st.write("Help us understand your skills and preferences to match you with the perfect study group!")
    
    with st.form("profile_form"):
        st.markdown("#### 💻 Technical Skills (Rate 1-5)")
        col1, col2 = st.columns(2)
        
        with col1:
            programming = st.slider("Programming (Python, Java, C++)", 1, 5, 3)
            dsa = st.slider("Data Structures & Algorithms", 1, 5, 3)
            os = st.slider("Operating Systems", 1, 5, 3)
        
        with col2:
            dbms = st.slider("Database Management", 1, 5, 3)
            cn = st.slider("Computer Networks", 1, 5, 3)
            projects = st.slider("Project Experience", 0, 5, 2)
        
        st.markdown("#### 📚 Study Preferences")
        col3, col4 = st.columns(2)
        
        with col3:
            study_time = st.selectbox("Preferred Study Time", 
                                     ["Morning (6AM-12PM)", "Afternoon (12PM-6PM)", "Evening (6PM-12AM)"])
            learning_style = st.selectbox("Learning Style", 
                                         ["Visual (Videos, Diagrams)", "Auditory (Discussions)", 
                                          "Kinesthetic (Hands-on)", "Reading/Writing"])
        
        with col4:
            goal = st.selectbox("Primary Goal", 
                               ["Placement Preparation", "Research Focus", "CGPA Improvement", "Competitive Coding"])
            cgpa = st.number_input("Current CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
        
        submitted = st.form_submit_button("🚀 Submit Profile & Find My Group", use_container_width=True)
        
        if submitted:
            # Map categorical values to numeric
            study_time_map = {"Morning (6AM-12PM)": 0, "Afternoon (12PM-6PM)": 1, "Evening (6PM-12AM)": 2}
            learning_style_map = {"Visual (Videos, Diagrams)": 0, "Auditory (Discussions)": 1, 
                                 "Kinesthetic (Hands-on)": 2, "Reading/Writing": 3}
            goal_map = {"Placement Preparation": 0, "Research Focus": 1, "CGPA Improvement": 2, "Competitive Coding": 3}
            
            profile = {
                'programming': programming,
                'dsa': dsa,
                'os': os,
                'dbms': dbms,
                'cn': cn,
                'projects': projects,
                'study_time': study_time_map[study_time],
                'learning_style': learning_style_map[learning_style],
                'goal': goal_map[goal],
                'cgpa': cgpa,
                'study_time_label': study_time,
                'learning_style_label': learning_style,
                'goal_label': goal
            }
            
            # Train model if not already trained
            if not st.session_state.model_trained:
                with st.spinner("🤖 Training ML model..."):
                    model, scaler = train_clustering_model()
                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    st.session_state.model_trained = True
            
            # Predict group
            group_id = predict_group(profile, st.session_state.model, st.session_state.scaler)
            
            # Save profile
            user_email = st.session_state.user_data['email']
            st.session_state.profiles_db[user_email] = profile
            
            # Add to group
            if group_id not in st.session_state.groups_db:
                st.session_state.groups_db[group_id] = []
            
            st.session_state.groups_db[group_id].append({
                'email': user_email,
                'name': st.session_state.user_data['name'],
                'branch': st.session_state.user_data['branch'],
                'goal': goal
            })
            
            st.success("✅ Profile saved! Finding your perfect group...")
            st.rerun()

def show_group_dashboard():
    """Display group assignment dashboard"""
    user_email = st.session_state.user_data['email']
    user_profile = st.session_state.profiles_db[user_email]
    
    # Find user's group
    user_group_id = None
    for group_id, members in st.session_state.groups_db.items():
        if any(m['email'] == user_email for m in members):
            user_group_id = group_id
            break
    
    if user_group_id is None:
        st.error("Group not found. Please contact admin.")
        return
    
    # Display group info
    st.subheader("🎉 Your Study Group Assignment")
    
    group_members = st.session_state.groups_db[user_group_id]
    member_names = [m['name'] for m in group_members]
    common_goal = group_members[0]['goal']
    
    # Group card
    st.markdown(f"""
    <div class="group-card">
        <h2>Group {user_group_id + 1}</h2>
        <h4>👥 Members: {', '.join(member_names)}</h4>
        <h4>🎯 Common Goal: {common_goal}</h4>
        <p>✅ Matched based on skills, learning style, and study preferences</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <h3>👥 {len(member_names)}</h3>
            <p>Group Members</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <h3>⏰ {user_profile['study_time_label'].split()[0]}</h3>
            <p>Preferred Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <h3>🧠 {user_profile['learning_style_label'].split()[0]}</h3>
            <p>Learning Style</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Member details
    st.subheader("📊 Group Member Profiles")
    
    for member in group_members:
        with st.expander(f"👤 {member['name']} - {member['branch']}"):
            if member['email'] in st.session_state.profiles_db:
                profile = st.session_state.profiles_db[member['email']]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Technical Skills:**")
                    st.write(f"- Programming: {'⭐' * profile['programming']}")
                    st.write(f"- DSA: {'⭐' * profile['dsa']}")
                    st.write(f"- OS: {'⭐' * profile['os']}")
                
                with col2:
                    st.write("**Preferences:**")
                    st.write(f"- CGPA: {profile['cgpa']}")
                    st.write(f"- Study Time: {profile['study_time_label']}")
                    st.write(f"- Learning: {profile['learning_style_label']}")
    
    # Email notification
    st.subheader("📧 Email Notification")
    
    if st.button("📨 Send Group Details to My Email", use_container_width=True):
        subject, body = send_email_notification(
            user_email,
            st.session_state.user_data['name'],
            user_group_id + 1,
            member_names,
            common_goal
        )
        
        # Show email preview
        st.success("✅ Email notification prepared!")
        with st.expander("📬 Email Preview"):
            st.text(f"To: {user_email}")
            st.text(f"Subject: {subject}")
            st.text("---")
            st.text(body)
        
        st.info("💡 In production, this email would be automatically sent via SMTP (Gmail/SendGrid)")

# Run app
if __name__ == "__main__":
    main()