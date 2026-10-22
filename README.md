📡 ECHOPOINT — Institutional Feedback Intelligence System
Transforming raw student feedback into actionable administrative intelligence.
🌟 Overview
Educational institutions are often overwhelmed by thousands of unstructured feedback entries. Most systems simply summarize text, leaving administrators to manually search for patterns and urgent issues.
ECHOPOINT is an intelligent feedback ecosystem. It doesn't just summarize; it analyzes. By combining deterministic keyword mapping with unsupervised machine learning and LLM-driven reasoning, ECHOPOINT identifies not just what is being said, but where it is happening, how severe it is, and what the underlying institutional connections are.
✨ Key Features
1. 📝 Dual-Mode Interface
Student Portal: A streamlined, friction-less interface for students to submit detailed complaints.
Admin Dashboard: A high-level command center for administrators to monitor campus health.
2. 🧠 Hybrid Intelligence Engine
Instead of relying on a single method, ECHOPOINT uses a sophisticated two-tier approach:
Deterministic Layer: High-accuracy keyword mapping for known institutional categories (Wi-Fi, Canteen, etc.).
Discovery Layer: Unsupervised K-Means Clustering (TF-IDF) that automatically identifies "Unknown" patterns, allowing the system to discover new, emerging problems without manual configuration.
3. 🌐 Knowledge Graph (Graphify-Inspired)
Inspired by graph-based code analysis, ECHOPOINT maps the relationship between Issue Clusters and Campus Entities (e.g., connecting "Connectivity" issues to the "Library" or "Dorms"). This visualizes the "web" of problems, helping administrators see how issues are geographically or contextually linked.
4. 🚨 Statistical Anomaly Detection
Using Z-Score analysis on time-series data, the system automatically flags unusual spikes in complaint volume, alerting administrators to sudden infrastructure failures or crises in real-time.
5. 🤖 Gemma-Powered Reasoning
ECHOPOINT integrates the Gemma LLM to act as an expert analyst. It consumes raw statistical findings and produces a professional Administrative Briefing, providing:
Concise explanations.
Evidence-based interpretations.
Actionable investigation steps.
🛠️ Tech Stack
Language: Python 3.13
Frontend/Dashboard: Streamlit
Data Science: Pandas, NumPy
Machine Learning: Scikit-learn (K-Means, TF-IDF)
Graph Theory: NetworkX
Visualization: Plotly
AI Reasoning: Google Gen AI SDK (Gemma)
🚀 Installation & Setup
1. Clone & Environment
code
Bash
# Navigate to project
cd ECHOPOINT

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
2. Install Dependencies
code
Bash
pip install -r requirements.txt
3. Configure API Key
Create a .env file in the root directory and add your Google API Key:
code
Text
GOOGLE_API_KEY=your_api_key_here
4. Prepare Data
Ensure your master data file is located at data/reportOct.csv. If no file exists, the system will allow students to create new data via the portal.
5. Launch
code
Bash
streamlit run app.py
📊 Workflow
Student Input 
→
→
 Monthly CSV Persistence 
→
→
 Hybrid Clustering/Keyword Analysis 
→
→
 Statistical Anomaly Detection 
→
→
 Knowledge Graph Generation 
→
→
 Gemma AI Briefing
👨‍💻 Authors
Rudra
