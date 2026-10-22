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
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemma-4-26b-a4b-it')
else:
    st.error("API Key not found. Please check your .env file.")

# --- DATA ENGINE ---

def generate_synthetic_data(rows=130):
    np.random.seed(42)
    templates = [
        "The Wi-Fi in the library is extremely slow.",
        "Cannot connect to the campus Wi-Fi in the dorms.",
        "The student portal is lagging during registration.",
        "Assignment upload keeps failing on the portal.",
        "The laboratory equipment in Room 302 is broken.",
        "Classroom projectors are not working in the library.",
        "Administrative staff were unhelpful at the registrar.",
        "The cafeteria food quality has dropped.",
        "Library study rooms are always fully booked.",
        "Internet connectivity issues in the cafeteria."
    ]
    data = {
        'timestamp': [datetime.now() - timedelta(days=np.random.randint(0, 30)) for _ in range(rows)],
        'feedback_text': [np.random.choice(templates) for _ in range(rows)]
    }
    df = pd.DataFrame(data)
    # Inject Anomaly
    anomaly_dates = [datetime.now() - timedelta(days=np.random.uniform(0, 2)) for _ in range(15)]
    df = pd.concat([df, pd.DataFrame({'timestamp': anomaly_dates, 'feedback_text': ["Wi-Fi is completely down!"] * 15})], ignore_index=True)
    return df.sort_values('timestamp')

def run_clustering(df, n_clusters=4):
    if len(df) < n_clusters: return df, []
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['feedback_text'])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)
    cluster_summary = []
    for i in range(n_clusters):
        group = df[df['cluster'] == i]
        cluster_summary.append({"cluster_id": i, "count": len(group), "samples": group['feedback_text'].iloc[:3].tolist()})
    return df, cluster_summary

def detect_anomalies(df):
    df_daily = df.set_index('timestamp').resample('D').size().reset_index(name='counts')
    if len(df_daily) < 3: return None, df_daily
    mean_val, std_val = df_daily['counts'].mean(), df_daily['counts'].std()
    df_daily['z_score'] = (df_daily['counts'] - mean_val) / std_val
    anomalies = df_daily[df_daily['z_score'] > 1.5]
    return anomalies, df_daily

def generate_knowledge_graph(df, clusters):
    """Creates a graph mapping Clusters to common Entities (Keywords)."""
    G = nx.Graph()
    
    # Define common entities we want to track
    entities = ["Wi-Fi", "Library", "Portal", "Cafeteria", "Staff", "Equipment", "Dorms"]
    
    for c in clusters:
        cluster_node = f"Issue Group {c['cluster_id']}"
        G.add_node(cluster_node, type='cluster')
        
        # Check which entities appear in this cluster's text
        cluster_text = " ".join(df[df['cluster'] == c['cluster_id']]['feedback_text'])
        for entity in entities:
            if entity.lower() in cluster_text.lower():
                G.add_node(entity, type='entity')
                G.add_edge(cluster_node, entity)

    # Convert NetworkX to Plotly
    pos = nx.spring_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')

    node_x, node_y, node_text, node_color = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        # Color clusters differently from entities
        node_color.append('skyblue' if G.nodes[node].get('type') == 'cluster' else 'orange')

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center",
        marker=dict(size=20, color=node_color, line_width=2), hoverinfo='text'
    )

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=0),
        title="Knowledge Graph: Issue-Entity Relationships"
    ))
    return fig

def get_gemma_analysis(clusters, anomalies):
    cluster_str = "\n".join([f"- Group {c['cluster_id']}: {c['count']} reports. Samples: {'; '.join(c['samples'])}" for c in clusters])
    anomaly_str = f"Anomalies on {anomalies['timestamp'].dt.date.tolist()} with counts: {anomalies['counts'].tolist()}." if anomalies is not None and not anomalies.empty else "No anomalies."
    
    prompt = f"Analyze these institutional feedback patterns and provide a briefing for administrators:\n\nFINDINGS:\n{cluster_str}\n{anomaly_str}\n\nUse format: 1. Explanation, 2. Evidence, 3. Interpretation, 4. Action Steps, 5. Limitations."
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Error: {str(e)}"

# --- UI ---
st.title("📡 ECHOPOINT")
st.caption("Institutional Feedback Intelligence | Knowledge Graph Enabled")

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

with st.spinner("Running Intelligence Pipeline..."):
    df_clustered, clusters = run_clustering(df)
    anomalies, daily_counts = detect_anomalies(df)
    kg_fig = generate_knowledge_graph(df, clusters)

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Feedback", len(df))
col2.metric("Issue Groups", len(clusters))
col3.metric("Anomalies", "YES" if (anomalies is not None and not anomalies.empty) else "NO")

st.divider()

# Tabs for Organization
tab1, tab2, tab3 = st.tabs(["📈 Trends & Patterns", "🌐 Knowledge Graph", "🤖 AI Briefing"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Daily Volume")
        fig_line = px.line(daily_counts, x='timestamp', y='counts')
        if anomalies is not None and not anomalies.empty:
            fig_line.add_scatter(x=anomalies['timestamp'], y=anomalies['counts'], mode='markers', marker=dict(size=12, color='red'), name="Anomaly")
        st.plotly_chart(fig_line, use_container_width=True)
    with c2:
        st.subheader("Issue Distribution")
        st.plotly_chart(px.pie(pd.DataFrame(clusters), values='count', names='cluster_id'), use_container_width=True)
    
    st.subheader("Detailed Cluster Breakdown")
    for c in clusters:
        with st.expander(f"Group {c['cluster_id']} ({c['count']} reports)"):
            for s in c['samples']: st.write(f"• {s}")

with tab2:
    st.subheader("Entity-Issue Relationship Map")
    st.write("This graph visualizes how different issue clusters are linked to specific campus entities (e.g., Wi-Fi $\\rightarrow$ Library).")
    st.plotly_chart(kg_fig, use_container_width=True)

with tab3:
    if st.button("Generate AI Intelligence Briefing"):
        with st.spinner("Consulting Gemma..."):
            st.markdown(get_gemma_analysis(clusters, anomalies))
    else:
        st.info("Click the button to generate the briefing.")