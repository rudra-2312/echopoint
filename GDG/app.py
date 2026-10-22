import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime
import os
from dotenv import load_dotenv

# NLP & Analytics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# AI
import google.generativeai as genai

# --- CONFIGURATION & INITIALIZATION ---
load_dotenv()
st.set_page_config(page_title="ECHOPOINT", layout="wide")

# Path to your specific permanent file
DATA_FILE = "data/reportOct.csv"

# Initialize Session State
if 'page' not in st.session_state:
    st.session_state.page = 'student'
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Initialize Gemma
API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemma-4-26b-a4b-it')
else:
    st.error("API Key not found. Please check your .env file.")

# --- DATA PERSISTENCE LAYER ---

def save_complaint(text):
    """Appends a new complaint to your master file, matching your schema."""
    if not os.path.exists(DATA_FILE):
        # If file is missing, create it with header
        new_data = pd.DataFrame({
            'feedback_id': [1],
            'timestamp': [datetime.now()],
            'category': ['Uncategorized'],
            'location': ['Unknown'],
            'feedback_text': [text],
            'severity': ['medium']
        })
        new_data.to_csv(DATA_FILE, index=False)
    else:
        # Append to existing file
        new_id = 0
        try:
            existing = pd.read_csv(DATA_FILE)
            new_id = existing['feedback_id'].max() + 1
        except: pass
        
        new_data = pd.DataFrame({
            'feedback_id': [new_id],
            'timestamp': [datetime.now()],
            'category': ['Uncategorized'],
            'location': ['Unknown'],
            'feedback_text': [text],
            'severity': ['medium']
        })
        new_data.to_csv(DATA_FILE, mode='a', header=False, index=False)

def load_master_data():
    """Loads your specific report file."""
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    else:
        st.error(f"File not found: {DATA_FILE}. Please ensure your file is in the 'data' folder.")
        return pd.DataFrame(columns=['feedback_id', 'timestamp', 'category', 'location', 'feedback_text', 'severity'])

# --- ANALYTICS ENGINE ---

def run_hybrid_analysis(df):
    """Uses existing 'category' column if available, otherwise clusters."""
    if df.empty: return df, []
    
    # If the CSV already has categories, we use them for grouping
    # Otherwise, we run the clustering logic
    if 'category' in df.columns and df['category'].nunique() > 1:
        summary = []
        for cat in df['category'].unique():
            group = df[df['category'] == cat]
            summary.append({"category": cat, "count": len(group), "samples": group['feedback_text'].iloc[:3].tolist()})
        return df, sorted(summary, key=lambda x: x['count'], reverse=True)
    
    # Fallback to clustering if categories are missing or all 'Uncategorized'
    category_map = {
        'Connectivity': ['wifi', 'internet', 'connection', 'network', 'slow'],
        'Facilities/Cleaning': ['canteen', 'food', 'table', 'floor', 'clean', 'cafeteria', 'dining'],
        'Academic/Portal': ['portal', 'assignment', 'upload', 'student', 'login'],
        'Classroom/Lab': ['lab', 'classroom', 'projector', 'equipment', 'room']
    }
    def assign_category(text):
        text = str(text).lower()
        for category, keywords in category_map.items():
            if any(kw in text for kw in keywords): return category
        return "Unknown/Other"

    df['detected_category'] = df['feedback_text'].apply(assign_category)
    summary = []
    for cat in df['detected_category'].unique():
        group = df[df['detected_category'] == cat]
        summary.append({"category": cat, "count": len(group), "samples": group['feedback_text'].iloc[:3].tolist()})
    return df, sorted(summary, key=lambda x: x['count'], reverse=True)

def detect_anomalies(df):
    if df.empty: return None, pd.DataFrame()
    df_daily = df.set_index('timestamp').resample('D').size().reset_index(name='counts')
    if len(df_daily) < 3: return None, df_daily
    mean_val, std_val = df_daily['counts'].mean(), df_daily['counts'].std()
    df_daily['z_score'] = (df_daily['counts'] - mean_val) / std_val
    anomalies = df_daily[df_daily['z_score'] > 1.5]
    return anomalies, df_daily

def generate_knowledge_graph(df, summary):
    G = nx.Graph()
    entities = ["Wi-Fi", "Library", "Portal", "Cafeteria", "Staff", "Equipment", "Dorms"]
    for item in summary:
        cat = item['category']
        G.add_node(cat, type='category')
        cat_text = " ".join(df[df.get('detected_category', df['category']) == cat]['feedback_text'].astype(str)).lower()
        for ent in entities:
            if ent.lower() in cat_text:
                G.add_node(ent, type='entity')
                G.add_edge(cat, ent)
    if len(G.nodes) == 0: return go.Figure()
    pos = nx.spring_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
    node_x, node_y, node_text, node_color = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]; node_x.append(x); node_y.append(y); node_text.append(node)
        node_color.append('#1f77b4' if G.nodes[node].get('type') == 'category' else '#ff7f0e')
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center",
                            marker=dict(size=22, color=node_color, line_width=2), hoverinfo='text')
    return go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, margin=dict(b=0,l=0,r=0,t=40), title="Knowledge Graph"))

def get_gemma_analysis(summary, anomalies):
    summary_str = "\n".join([f"- {s['category']}: {s['count']} reports." for s in summary])
    anomaly_str = f"Anomalies on {anomalies['timestamp'].dt.date.tolist()}." if anomalies is not None and not anomalies.empty else "None."
    prompt = f"Analyze these findings for administrators:\n{summary_str}\n{anomaly_str}\n\nFormat: 1. Explanation, 2. Evidence, 3. Interpretation, 4. Action, 5. Limitations."
    try: return model.generate_content(prompt).text
    except: return "AI service currently unavailable."

# --- UI ROUTING ---

def student_page():
    st.title("📝 Student Feedback Portal")
    st.write("Your voice matters. Submit your concerns below.")
    with st.container():
        complaint = st.text_area("Describe your issue:", placeholder="e.g., The Wi-Fi in the library is very slow...", height=150)
        if st.button("Submit Feedback"):
            if complaint.strip():
                save_complaint(complaint)
                st.success("Thank you! Your feedback has been securely submitted.")
            else:
                st.error("Please enter some text.")
    st.divider()
    if st.button("🔑 Admin Login"):
        st.session_state.page = 'login'
        st.rerun()

def login_page():
    st.title("🔐 Administrator Access")
    with st.form("login_form"):
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if user == "admin" and pw == "echo123":
                st.session_state.logged_in = True
                st.session_state.page = 'dashboard'
                st.rerun()
            else:
                st.error("Invalid credentials")
    if st.button("⬅️ Back to Student Portal"):
        st.session_state.page = 'student'
        st.rerun()

def admin_dashboard():
    st.title("📡 ECHOPOINT Admin Dashboard")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = 'student'
        st.rerun()

    df = load_master_data()
    if df.empty:
        st.warning("No data found in the master file.")
        return

    with st.spinner("Processing Intelligence..."):
        df, summary = run_hybrid_analysis(df)
        anomalies, daily_counts = detect_anomalies(df)
        kg_fig = generate_knowledge_graph(df, summary)

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Active Categories", len(summary))
    col3.metric("Critical Anomalies", "YES" if (anomalies is not None and not anomalies.empty) else "NO")

    st.divider()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Analytics & Severity", "🌐 Knowledge Graph", "🤖 AI Briefing"])

    with tab1:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.subheader("Daily Volume")
            fig_line = px.line(daily_counts, x='timestamp', y='counts', template="plotly_white")
            if anomalies is not None and not anomalies.empty:
                fig_line.add_scatter(x=anomalies['timestamp'], y=anomalies['counts'], mode='markers', marker=dict(size=14, color='red'), name="Anomaly")
            st.plotly_chart(fig_line, use_container_width=True)
        
        with c2:
            st.subheader("Severity Levels")
            # Use the 'severity' column from your CSV
            fig_sev = px.pie(df, names='severity', color='severity', 
                             color_discrete_map={'high':'red', 'medium':'orange', 'low':'green'},
                             hole=0.4)
            st.plotly_chart(fig_sev, use_container_width=True)
            
        with c3:
            st.subheader("Hotspots (Location)")
            loc_counts = df['location'].value_counts().reset_index()
            loc_counts.columns = ['location', 'count']
            fig_loc = px.bar(loc_counts, x='count', y='location', orientation='h', color='count')
            st.plotly_chart(fig_loc, use_container_width=True)

        st.subheader("Detailed Category Breakdown")
        for s in summary:
            with st.expander(f"{s['category']} ({s['count']} reports)"):
                for sample in s['samples']: st.write(f"• {sample}")

    with tab2:
        st.subheader("Entity-Issue Relationship Map")
        st.plotly_chart(kg_fig, use_container_width=True)

    with tab3:
        if st.button("Generate AI Intelligence Briefing"):
            with st.spinner("Consulting Gemma..."):
                st.markdown(get_gemma_analysis(summary, anomalies))
        else:
            st.info("Click the button to generate the briefing.")

# --- MAIN ROUTER ---
if st.session_state.page == 'student':
    student_page()
elif st.session_state.page == 'login':
    login_page()
elif st.session_state.page == 'dashboard':
    if st.session_state.logged_in:
        admin_dashboard()
    else:
        st.session_state.page = 'student'
        st.rerun()