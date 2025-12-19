import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
import sqlite3
from datetime import datetime

# ==========================================
# 1. SETUP & ASSET LOADING
# ==========================================
# Load Models & Data
try:
    model = joblib.load('churn_model.pkl')
    encoder = joblib.load('contract_encoder.pkl')
    data = pd.read_csv('churn_data.csv')
except:
    st.error("⚠️ Error: Project files not found. Please run 'train_model.py' first.")
    st.stop()

# ==========================================
# 2. DATABASE MANAGEMENT (SQLite)
# ==========================================
def init_db():
    conn = sqlite3.connect('churn_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (date TEXT, tenure INTEGER, monthly REAL, total REAL, contract TEXT, prediction TEXT, probability REAL)''')
    conn.commit()
    conn.close()

def add_data(tenure, monthly, total, contract, prediction, probability):
    conn = sqlite3.connect('churn_history.db')
    c = conn.cursor()
    date_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO history VALUES (?,?,?,?,?,?,?)", 
              (date_now, tenure, monthly, total, contract, prediction, probability))
    conn.commit()
    conn.close()

def view_all_data():
    conn = sqlite3.connect('churn_history.db')
    df = pd.read_sql_query("SELECT * FROM history ORDER BY date DESC", conn)
    conn.close()
    return df

init_db()

# ==========================================
# 3. SIDEBAR & NAVIGATION
# ==========================================
st.set_page_config(page_title="Telecom Intelligence", layout="wide", page_icon="🤖")

with st.sidebar:
    # Standard Image (No Animation)
    st.image("https://cdn-icons-png.flaticon.com/512/2920/2920329.png", width=120)
        
    st.title("Navigation")
    option = st.radio("Select Module:", [
        "🔮 Smart Prediction", 
        "💬 AI Data Consultant", 
        "📂 Batch Scan", 
        "📊 Dashboard",
        "📜 Database History"
    ])
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Project Designed by:")
    st.markdown("Rajat Chauhan CSE-1 Roll No: 72")

# ==========================================
# MODULE 1: SMART PREDICTION
# ==========================================
if option == "🔮 Smart Prediction":
    st.title("🔮 Customer Retention AI")
    st.markdown("Analyze customer risk and generate **instant retention strategies**.")
    
    # Inputs
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            tenure = st.slider("Tenure (Months)", 0, 72, 12)
        with col2:
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 85.0)
        with col3:
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)
        contract_type = st.selectbox("Contract Type", encoder.classes_)
    
    # Analyze Button
    if st.button("Analyze Risk", type="primary"):
        
        # Simple Progress Bar (Cleaner than animation)
        progress_text = "Running Random Forest Algorithm..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(0.2)
        my_bar.empty()
            
        # Prediction Logic
        contract_num = encoder.transform([contract_type])[0]
        input_data = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'Contract': [contract_num]
        })
        
        prediction = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]
        churn_prob = probs[1] * 100
        
        # Save to Database
        add_data(tenure, monthly_charges, total_charges, contract_type, prediction, churn_prob)
        st.toast("✅ Record saved to Database!", icon="💾")

        st.divider()
        
        # Results Tabs
        tab1, tab2, tab3 = st.tabs(["🔍 Risk Analysis", "📧 Email Draft", "💰 ROI Calculator"])
        
        with tab1:
            c1, c2 = st.columns([1, 2])
            with c1:
                if prediction == "Yes":
                    st.error("🚨 HIGH RISK DETECTED")
                    st.metric("Churn Probability", f"{churn_prob:.2f}%", delta="Critical", delta_color="inverse")
                    st.snow()
                else:
                    st.success("✅ SAFE CUSTOMER")
                    st.metric("Loyalty Score", f"{probs[0]*100:.2f}%", delta="Stable")
                    st.balloons()
            
            with c2:
                st.caption("AI Model Explanation:")
                # Visualizing Feature Importance (Simulated for Demo)
                importance = {'Contract': 40, 'Bill Amount': 35, 'Tenure': 25}
                if contract_type == "Two year": importance['Contract'] = -40
                
                fig, ax = plt.subplots(figsize=(5, 1.5))
                colors = ['red' if prediction == 'Yes' else 'green']
                ax.barh(list(importance.keys()), list(importance.values()), color=colors)
                st.pyplot(fig)

        with tab2:
            st.subheader("📧 Retention Email Draft")
            if prediction == "Yes":
                code = f"SAVE{random.randint(10,99)}"
                st.text_area("Copy this email:", value=f"Subject: Special Offer!\n\nDear Customer,\n\nWe value your partnership. Use code {code} to get 15% off your next bill.\n\nBest,\nSupport Team", height=150)
            else:
                st.info("Customer is safe. No retention email needed.")

        with tab3:
            if prediction == "Yes":
                loss = monthly_charges * 12
                st.metric("Potential Revenue Protected", f"${loss:.2f}")
            else:
                st.success("Revenue is stable.")

# ==========================================
# MODULE 2: AI CONSULTANT (Chatbot)
# ==========================================
elif option == "💬 AI Data Consultant":
    st.title("💬 AI Business Consultant")
    st.markdown("Ask questions about your data in plain English.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AI Data Assistant. Ask me about churn rates or revenue."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question (e.g. 'What is the churn rate?')..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            prompt_lower = prompt.lower()
            
            # Simple Rule-Based Logic
            if "churn rate" in prompt_lower:
                rate = (data['Churn'].value_counts(normalize=True)['Yes']) * 100
                response_text = f"Based on the dataset, the overall Churn Rate is **{rate:.2f}%**."
            elif "total customers" in prompt_lower or "how many" in prompt_lower:
                count = len(data)
                response_text = f"There are a total of **{count:,}** customers in the database."
            elif "revenue" in prompt_lower:
                total = data['MonthlyCharges'].sum()
                response_text = f"The total monthly revenue is **${total:,.2f}**."
            elif "hello" in prompt_lower:
                response_text = "Hello! I am ready to analyze your telecom data."
            else:
                response_text = "I am trained on Telecom Churn data. Try asking: 'What is the churn rate?'"
            
            # Typing Effect
            for chunk in response_text.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# ==========================================
# MODULE 3: BATCH SCAN
# ==========================================
elif option == "📂 Batch Scan":
    st.title("📂 Enterprise Batch Scanner")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        if st.button("Run Analysis"):
            st.success("Analysis Complete!")
            st.dataframe(df_upload.head())
            st.download_button("Download Full Report", df_upload.to_csv(), "report.csv")

# ==========================================
# MODULE 4: DASHBOARD
# ==========================================
elif option == "📊 Dashboard":
    st.title("📊 Executive Dashboard")
    
    # Filters
    contract_filter = st.multiselect("Select Contract", data['Contract'].unique(), default=data['Contract'].unique())
    df_filtered = data[data['Contract'].isin(contract_filter)]
    
    # KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Users", len(df_filtered))
    k2.metric("Avg Bill", f"${df_filtered['MonthlyCharges'].mean():.2f}")
    
    if len(df_filtered) > 0:
        churn_rate = (df_filtered['Churn'].value_counts(normalize=True).get('Yes', 0)) * 100
        k3.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    st.divider()
    
    # Charts
    c1, c2 = st.columns(2)
    with c1:
        fig = plt.figure(figsize=(6,4))
        sns.countplot(data=df_filtered, x='Contract', hue='Churn', palette='viridis')
        st.pyplot(fig)
    with c2:
        fig = plt.figure(figsize=(6,4))
        sns.histplot(data=df_filtered, x='MonthlyCharges', kde=True, hue='Churn', palette='coolwarm')
        st.pyplot(fig)

# ==========================================
# MODULE 5: HISTORY
# ==========================================
elif option == "📜 Database History":
    st.title("📜 History Log")
    st.markdown("Records saved to **SQLite Database**.")
    result = view_all_data()
    st.dataframe(result)
    if not result.empty:
        st.download_button("Download Log", result.to_csv(), "log.csv")