import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import os
from datetime import datetime
from dotenv import load_dotenv

# NLP & Analytics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# AI
import google.generativeai as genai

# --- CONFIGURATION & INITIALIZATION ---
load_dotenv()
st.set_page_config(page_title="ECHOPOINT", layout="wide")

# Ensure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# Initialize Session State for Navigation and Login
if 'page' not in st.session_state:
    st.session_state.page = 'student'
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Initialize Gemma
API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemma-4-26b-a4b-it')

# --- DATA PERSISTENCE LAYER ---

def get_current_month_filename():
    """Returns filename for the current month, e.g., data/feedback_2024_10.csv"""
    return f"data/feedback_{datetime.now().strftime('%Y_%m')}.csv"

def save_complaint(text):
    """Appends a new complaint to the current month's CSV."""
    filename = get_current_month_filename()
    new_data = pd.DataFrame({
        'timestamp': [datetime.now()],
        'feedback_text': [text]
    })
    
    # If file doesn't exist, write with header; otherwise, append without header
    if not os.path.exists(filename):
        new_data.to_csv(filename, index=False)
    else:
        new_data.to_csv(filename, mode='a', header=False, index=False)

def load_current_data():
    """Loads the CSV for the current month. Returns empty DF if none exists."""
    filename = get_current_month_filename()
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return pd.DataFrame(columns=['timestamp', 'feedback_text'])

# --- ANALYTICS ENGINE (The Core Logic) ---

def run_hybrid_analysis(df):
    if df.empty: return df, []
    
    category_map = {
        'Connectivity': ['wifi', 'internet', 'connection', 'network', 'slow'],
        'Facilities/Cleaning': ['canteen', 'food', 'table', 'floor', 'clean', 'cafeteria', 'dining'],
        'Academic/Portal': ['portal', 'assignment', 'upload', 'student', 'login'],
        'Classroom/Lab': ['lab', 'classroom', 'projector', 'equipment', 'room']
    }

    def assign_category(text):
        text = text.lower()
        for category, keywords in category_map.items():
            if any(kw in text for kw in keywords): return category
        return "Unknown/Other"

    df['detected_category'] = df['feedback_text'].apply(assign_category)

    # Clustering for unknowns
    unknown_mask = df['detected_category'] == "Unknown/Other"
    unknown_df = df[unknown_mask].copy()
    if len(unknown_df) >= 3:
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(unknown_df['feedback_text'])
        kmeans = KMeans(n_clusters=min(len(unknown_df), 3), random_state=42, n_init=10)
        unknown_df['cluster_id'] = kmeans.fit_predict(X)
        unknown_df['detected_category'] = unknown_df['cluster_id'].apply(lambda x: f"Discovery Group {x}")
        df.loc[unknown_mask, 'detected_category'] = unknown_df['detected_category']

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
        cat_text = " ".join(df[df['detected_category'] == cat]['feedback_text']).lower()
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
        x, y = pos[node]
        node_x.append(x); node_y.append(y); node_text.append(node)
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

# --- PAGE ROUTING FUNCTIONS ---

def student_page():
    st.title("📝 Student Feedback Portal")
    st.write("Your feedback helps us improve the campus experience.")
    
    with st.container():
        complaint = st.text_area("Describe your issue/complaint in detail:", placeholder="e.g., The Wi-Fi in the library is very slow...", height=150)
        if st.button("Submit Feedback"):
            if complaint.strip():
                save_complaint(complaint)
                st.success("Thank you! Your feedback has been securely submitted.")
            else:
                st.error("Please enter some text before submitting.")

    st.divider()
    if st.button("🔑 Admin Login"):
        st.session_state.page = 'login'
        st.rerun()

def login_page():
    st.title("🔐 Administrator Access")
    with st.form("login_form"):
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            # TEMPORARY CREDENTIALS
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

    df = load_current_data()
    
    if df.empty:
        st.warning(f"No feedback recorded for the current month ({datetime.now().strftime('%B %Y')}).")
        return

    # Run Analysis
    with st.spinner("Processing intelligence..."):
        df, summary = run_hybrid_analysis(df)
        anomalies, daily_counts = detect_anomalies(df)
        kg_fig = generate_knowledge_graph(df, summary)

    # Dashboard Layout
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Submissions", len(df))
    col2.metric("Issue Categories", len(summary))
    col3.metric("Anomalies", "Detected" if (anomalies is not None and not anomalies.empty) else "None")

    tab1, tab2, tab3 = st.tabs(["📈 Trends & Patterns", "🌐 Knowledge Graph", "🤖 AI Briefing"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Volume Trend")
            fig_line = px.line(daily_counts, x='timestamp', y='counts', template="plotly_white")
            if anomalies is not None and not anomalies.empty:
                fig_line.add_scatter(x=anomalies['timestamp'], y=anomalies['counts'], mode='markers', marker=dict(size=14, color='red'), name="Anomaly")
            st.plotly_chart(fig_line, use_container_width=True)
        with c2:
            st.subheader("Distribution")
            st.plotly_chart(px.pie(pd.DataFrame(summary), values='count', names='category', hole=0.4), use_container_width=True)
        
        st.subheader("Detailed Breakdown")
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
            st.info("Click to generate briefing.")

# --- MAIN ROUTER ---

if st.session_state.page == 'student':
    student_page()
elif st.session_state.page == 'login':
    login_page()
elif st.session_state.page == 'dashboard':
    if st.session_state.logged_in:
        admin_dashboard()
    else:
        st.error("Unauthorized access.")
        st.session_state.page = 'student'
        st.rerun()