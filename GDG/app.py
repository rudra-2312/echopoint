import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION & SETUP ---
st.set_page_config(page_title="ECHOPOINT | Feedback Intelligence", layout="wide")

REQUIRED_COLUMNS = ['timestamp', 'feedback_text']

def generate_synthetic_data(rows=100):
    """Generates a dataset with intentional patterns for demonstration."""
    np.random.seed(42)
    
    templates = [
        "The Wi-Fi in the library is extremely slow.",
        "Cannot connect to the campus Wi-Fi in the dorms.",
        "The student portal is lagging during registration.",
        "Assignment upload keeps failing on the portal.",
        "The laboratory equipment in Room 302 is broken.",
        "Classroom projectors are not working in the Science building.",
        "Administrative staff were unhelpful at the registrar.",
        "The cafeteria food quality has dropped significantly.",
        "Library study rooms are always fully booked.",
        "Internet connectivity issues in the cafeteria."
    ]
    
    data = {
        'timestamp': [datetime.now() - timedelta(days=np.random.randint(0, 30), hours=np.random.randint(0, 24)) for _ in range(rows)],
        'feedback_text': [np.random.choice(templates) for _ in range(rows)]
    }
    
    df = pd.DataFrame(data)
    
    # Inject an anomaly: A cluster of Wi-Fi complaints in the last 2 days
    anomaly_dates = [datetime.now() - timedelta(days=np.random.uniform(0, 2)) for _ in range(15)]
    anomaly_text = ["Wi-Fi is completely down!"] * 15
    df = pd.concat([df, pd.DataFrame({'timestamp': anomaly_dates, 'feedback_text': anomaly_text})], ignore_index=True)
    
    return df.sort_values('timestamp')

def validate_data(df):
    """Checks if the dataframe has required columns and valid data."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return False, f"Missing columns: {', '.join(missing)}"
    
    if df['feedback_text'].isnull().any():
        st.warning("Note: Missing feedback text detected and will be removed.")
        df = df.dropna(subset=['feedback_text'])
        
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        return False, f"Invalid timestamp format: {e}"
        
    return True, df

# --- UI LAYOUT ---

st.title("📡 ECHOPOINT")
st.subheader("Institutional Feedback Intelligence System")
st.divider()

# Sidebar: Data Ingestion
with st.sidebar:
    st.header("Data Ingestion")
    data_source = st.radio("Select Data Source:", ["Synthetic Data", "Upload CSV"])
    
    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Main Logic
if data_source == "Synthetic Data":
    st.info("Using high-fidelity synthetic data for demonstration.")
    raw_df = generate_synthetic_data(100)
else:
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.DataFrame()

if not raw_df.empty:
    is_valid, result = validate_data(raw_df.copy())
    
    if is_valid:
        df = result
        st.success(f"Successfully loaded {len(df)} feedback records.")
        
        # Data Preview Tab
        with st.expander("🔍 View Raw Data"):
            st.dataframe(df, use_container_width=True)
            
        # Placeholder for next steps
        st.divider()
        st.info("🚀 **Pipeline Status:** Data Ingested. Ready for Pattern Discovery.")
        
    else:
        st.error(f"Data Validation Error: {result}")
else:
    st.warning("Please upload a CSV file or select Synthetic Data to begin.")