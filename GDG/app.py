import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# NLP & Analytics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# AI
import google.generativeai as genai

# --- CONFIGURATION ---
load_dotenv()
st.set_page_config(page_title="ECHOPOINT | Institutional Intelligence", layout="wide")

# Initialize Gemma
API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemma-4-26b-a4b-it')
else:
    st.error("API Key not found. Please check your .env file.")

# --- DATA ENGINE ---

def generate_synthetic_data(rows=150):
    """Generates a dataset with high-quality patterns for demonstration."""
    np.random.seed(42)
    templates = [
        "The Wi-Fi in the library is extremely slow.",
        "Cannot connect to the campus Wi-Fi in the dorms.",
        "The student portal is lagging during registration.",
        "Assignment upload keeps failing on the portal.",
        "The laboratory equipment in Room 302 is broken.",
        "Classroom projectors are not working.",
        "Administrative staff were unhelpful at the registrar.",
        "The cafeteria food quality has dropped.",
        "Library study rooms are always fully booked.",
        "Internet connectivity issues in the cafeteria.",
        "The canteen tables are very dirty.",
        "Need more cleaning in the dining hall.",
        "The portal login is taking forever.",
        "Wi-Fi signal is weak in the north wing."
    ]
    
    data = {
        'timestamp': [datetime.now() - timedelta(days=np.random.randint(0, 30)) for _ in range(rows)],
        'feedback_text': [np.random.choice(templates) for _ in range(rows)]
    }
    df = pd.DataFrame(data)
    
    # Inject Anomaly: Wi-Fi spike
    anomaly_dates = [datetime.now() - timedelta(days=np.random.uniform(0, 2)) for _ in range(15)]
    df = pd.concat([df, pd.DataFrame({'timestamp': anomaly_dates, 'feedback_text': ["Wi-Fi is completely down!"] * 15})], ignore_index=True)
    
    return df.sort_values('timestamp')

def run_hybrid_analysis(df):
    """
    Hybrid Intelligence:
    1. Deterministic Layer: Keyword mapping for high accuracy on known issues.
    2. Discovery Layer: K-Means clustering for unknown/unmapped issues.
    """
    # 1. Deterministic Layer (Keyword Mapping)
    category_map = {
        'Connectivity': ['wifi', 'internet', 'connection', 'network', 'online', 'slow'],
        'Facilities/Cleaning': ['canteen', 'food', 'table', 'floor', 'clean', 'cafeteria', 'cleaning', 'dining'],
        'Academic/Portal': ['portal', 'assignment', 'upload', 'student', 'login', 'registration'],
        'Classroom/Lab': ['lab', 'classroom', 'projector', 'equipment', 'room']
    }

    def assign_category(text):
        text = text.lower()
        for category, keywords in category_map.items():
            if any(kw in text for kw in keywords):
                return category
        return "Unknown/Other"

    df['detected_category'] = df['feedback_text'].apply(assign_category)

    # 2. Discovery Layer (Cluster the 'Unknowns')
    unknown_mask = df['detected_category'] == "Unknown/Other"
    unknown_df = df[unknown_mask].copy()

    if len(unknown_df) >= 3:
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(unknown_df['feedback_text'])
        n_clusters = min(len(unknown_df), 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        unknown_df['cluster_id'] = kmeans.fit_predict(X)
        unknown_df['detected_category'] = unknown_df['cluster_id'].apply(lambda x: f"Discovery Group {x}")
        df.loc[unknown_mask, 'detected_category'] = unknown_df['detected_category']

    # 3. Prepare Summary
    summary = []
    for cat in df['detected_category'].unique():
        group = df[df['detected_category'] == cat]
        summary.append({
            "category": cat,
            "count": len(group),
            "samples": group['feedback_text'].iloc[:3].tolist()
        })
    
    summary = sorted(summary, key=lambda x: x['count'], reverse=True)
    return df, summary

def detect_anomalies(df):
    df_daily = df.set_index('timestamp').resample('D').size().reset_index(name='counts')
    if len(df_daily) < 3: return None, df_daily
    mean_val, std_val = df_daily['counts'].mean(), df_daily['counts'].std()
    df_daily['z_score'] = (df_daily['counts'] - mean_val) / std_val
    anomalies = df_daily[df_daily['z_score'] > 1.5]
    return anomalies, df_daily

def generate_knowledge_graph(df, summary):
    """Maps categories to campus entities to show relationships."""
    G = nx.Graph()
    entities = ["Wi-Fi", "Library", "Portal", "Cafeteria", "Staff", "Equipment", "Dorms", "Classroom"]
    
    for item in summary:
        cat = item['category']
        G.add_node(cat, type='category')
        
        # Find entities related to this category in the actual text
        cat_text = " ".join(df[df['detected_category'] == cat]['feedback_text']).lower()
        for ent in entities:
            if ent.lower() in cat_text:
                G.add_node(ent, type='entity')
                G.add_edge(cat, ent)

    pos = nx.spring_layout(G, k=0.5)
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

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center",
        marker=dict(size=22, color=node_color, line_width=2), hoverinfo='text'
    )

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=40),
        title="Knowledge Graph: Connecting Issues to Campus Entities"
    ))
    return fig

def get_gemma_analysis(summary, anomalies):
    summary_str = "\n".join([f"- {s['category']}: {s['count']} reports. Samples: {'; '.join(s['samples'])}" for s in summary])
    anomaly_str = f"Anomalies detected on {anomalies['timestamp'].dt.date.tolist()}." if anomalies is not None and not anomalies.empty else "No anomalies."
    
    prompt = f"""
    You are an Institutional Feedback Intelligence Analyst. 
    Analyze these findings and provide a professional briefing for University Administrators.
    
    FINDINGS:
    {summary_str}
    {anomaly_str}

    Format your response exactly as:
    1. **Concise Explanation**
    2. **Supporting Evidence**
    3. **Cautious Interpretation**
    4. **Suggested Investigation Steps**
    5. **Limitations**
    """
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Error generating AI insight: {str(e)}"

# --- UI LAYOUT ---
st.title("📡 ECHOPOINT")
st.caption("Institutional Feedback Intelligence | Hybrid Pattern Discovery")

with st.sidebar:
    st.header("Data Control")
    source = st.radio("Source:", ["Synthetic Data", "Upload CSV"])
    uploaded_file = st.file_uploader("Upload CSV", type="csv") if source == "Upload CSV" else None

if source == "Synthetic Data":
    df = generate_synthetic_data()
elif uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
else:
    st.warning("Please select a data source.")
    st.stop()

# Processing
with st.spinner("Running Intelligence Pipeline..."):
    df, summary = run_hybrid_analysis(df)
    anomalies, daily_counts = detect_anomalies(df)
    kg_fig = generate_knowledge_graph(df, summary)

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Feedback", len(df))
col2.metric("Resolved Categories", len(summary))
col3.metric("Anomalies Found", "Yes" if (anomalies is not None and not anomalies.empty) else "No")

st.divider()

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Trends & Patterns", "🌐 Knowledge Graph", "🤖 AI Briefing"])

with tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Daily Volume Trends")
        fig_line = px.line(daily_counts, x='timestamp', y='counts', template="plotly_white")
        if anomalies is not None and not anomalies.empty:
            fig_line.add_scatter(x=anomalies['timestamp'], y=anomalies['counts'], mode='markers', marker=dict(size=14, color='red'), name="Anomaly")
        st.plotly_chart(fig_line, use_container_width=True)
    with c2:
        st.subheader("Issue Distribution")
        st.plotly_chart(px.pie(pd.DataFrame(summary), values='count', names='category', hole=0.4), use_container_width=True)
    
    st.subheader("Detailed Category Breakdown")
    for s in summary:
        with st.expander(f"{s['category']} ({s['count']} reports)"):
            for sample in s['samples']:
                st.write(f"• *\"{sample}\"*")

with tab2:
    st.subheader("Entity-Issue Relationship Map")
    st.write("Inspired by Graphify: This graph visualizes how institutional issues connect to physical campus entities.")
    st.plotly_chart(kg_fig, use_container_width=True)

with tab3:
    if st.button("Generate AI Intelligence Briefing"):
        with st.spinner("Consulting Gemma..."):
            st.markdown(get_gemma_analysis(summary, anomalies))
    else:
        st.info("Click the button to generate the administrator briefing.")