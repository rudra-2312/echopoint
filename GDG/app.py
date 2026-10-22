import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
    # Using the model identifier you provided
    model = genai.GenerativeModel('gemma-4-26b-a4b-it')
else:
    st.error("API Key not found. Please check your .env file.")

# --- DATA ENGINE ---

def generate_synthetic_data(rows=120):
    """Generates a dataset with a clear pattern and an anomaly."""
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
        "Internet connectivity issues in the cafeteria."
    ]
    
    data = {
        'timestamp': [datetime.now() - timedelta(days=np.random.randint(0, 30)) for _ in range(rows)],
        'feedback_text': [np.random.choice(templates) for _ in range(rows)]
    }
    df = pd.DataFrame(data)
    
    # INJECT ANOMALY: A sudden spike of Wi-Fi complaints in the last 2 days
    anomaly_dates = [datetime.now() - timedelta(days=np.random.uniform(0, 2)) for _ in range(15)]
    anomaly_text = ["Wi-Fi is completely down!"] * 15
    df = pd.concat([df, pd.DataFrame({'timestamp': anomaly_dates, 'feedback_text': anomaly_text})], ignore_index=True)
    
    return df.sort_values('timestamp')

def run_clustering(df, n_clusters=4):
    """Groups feedback using TF-IDF and K-Means."""
    if len(df) < n_clusters:
        return df, []

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['feedback_text'])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)
    
    # Extract representative feedback for each cluster
    cluster_summary = []
    for i in range(n_clusters):
        group = df[df['cluster'] == i]
        cluster_summary.append({
            "cluster_id": i,
            "count": len(group),
            "samples": group['feedback_text'].iloc[:3].tolist()
        })
    return df, cluster_summary

def detect_anomalies(df):
    """Detects unusual spikes in complaint volume using Z-Score."""
    df_daily = df.set_index('timestamp').resample('D').size().reset_index(name='counts')
    
    if len(df_daily) < 3:
        return None, df_daily

    mean_val = df_daily['counts'].mean()
    std_val = df_daily['counts'].std()
    
    # Z-score > 2 is a standard statistical anomaly
    df_daily['z_score'] = (df_daily['counts'] - mean_val) / std_val
    anomalies = df_daily[df_daily['z_score'] > 1.5] # Lowered threshold to 1.5 for demo visibility
    
    return anomalies, df_daily

def get_gemma_analysis(clusters, anomalies):
    """Sends analytical findings to Gemma for reasoning."""
    cluster_str = ""
    for c in clusters:
        cluster_str += f"- Group {c['cluster_id']}: {c['count']} reports. Samples: {'; '.join(c['samples'])}\n"
    
    anomaly_str = "No significant anomalies detected."
    if anomalies is not None and not anomalies.empty:
        anomaly_str = f"Detected unusual activity on {anomalies['timestamp'].dt.date.tolist()} with counts: {anomalies['counts'].tolist()}."

    prompt = f"""
    You are an Institutional Feedback Intelligence Analyst. 
    Below are the processed findings from the feedback system. 
    DO NOT invent numbers. Use only the provided evidence.

    FINDINGS:
    {cluster_str}
    {anomaly_str}

    Provide a briefing for University Administrators in this exact format:
    1. **Concise Explanation**: (What is happening?)
    2. **Supporting Evidence**: (Cite the counts/patterns found)
    3. **Cautious Interpretation**: (What might be the underlying cause?)
    4. **Suggested Investigation Steps**: (What should they do next?)
    5. **Limitations**: (What is unknown?)
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating AI insight: {str(e)}"

# --- DASHBOARD UI ---

st.title("📡 ECHOPOINT")
st.caption("Institutional Feedback Intelligence System | Powered by Gemma")

# Sidebar
with st.sidebar:
    st.header("Data Control")
    source = st.radio("Source:", ["Synthetic Data", "Upload CSV"])
    if source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
    else:
        uploaded_file = None

# Main Logic
if source == "Synthetic Data":
    df = generate_synthetic_data()
elif uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
else:
    st.warning("Please select a data source.")
    st.stop()

# 1. Processing
with st.spinner("Analyzing patterns..."):
    df_clustered, clusters = run_clustering(df)
    anomalies, daily_counts = detect_anomalies(df)

# 2. Display Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Feedback", len(df))
col2.metric("Issue Groups", len(clusters))
col3.metric("Anomalies Detected", "Yes" if (anomalies is not None and not anomalies.empty) else "No")

st.divider()

# 3. Visualizations
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("📈 Complaint Trends")
    fig = px.line(daily_counts, x='timestamp', y='counts', title="Daily Feedback Volume")
    if anomalies is not None and not anomalies.empty:
        fig.add_scatter(x=anomalies['timestamp'], y=anomalies['counts'], mode='markers', 
                        marker=dict(size=12, color='red'), name="Anomaly")
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.subheader("🔍 Identified Issue Groups")
    cluster_df = pd.DataFrame(clusters)
    fig_pie = px.pie(cluster_df, values='count', names='cluster_id', title="Distribution of Issues")
    st.plotly_chart(fig_pie, use_container_width=True)

# 4. Detailed Patterns
st.subheader("📋 Pattern Breakdown")
for c in clusters:
    with st.expander(f"Group {c['cluster_id']} ({c['count']} reports)"):
        for sample in c['samples']:
            st.write(f"• *\"{sample}\"*")

# 5. Gemma Intelligence
st.divider()
st.subheader("🤖 Gemma Intelligence Briefing")
if st.button("Generate AI Insight"):
    with st.spinner("Consulting Gemma..."):
        insight = get_gemma_analysis(clusters, anomalies)
        st.markdown(insight)
else:
    st.info("Click the button above to generate the administrator briefing.")