import streamlit as st
import pandas as pd
import os
import time
import datetime
from corpus_loader import CorpusLoader
from retriever import Retriever
from agent import SupportAgent
import io

# Page Configuration
st.set_page_config(page_title="Support Triage AI", page_icon="🤖", layout="wide")

st.title("🤖 Support Triage Agent v2.0")
st.markdown("Automated ticket classification and response generation using RAG and Groq (Llama-3).")

# Sidebar Configuration
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
model_name = st.sidebar.selectbox("Model", ["llama-3.3-70b-versatile"])

# Initialize Components
@st.cache_resource
def init_rag():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    loader = CorpusLoader(data_dir)
    docs = loader.load_corpus()
    retriever = Retriever(docs)
    return retriever, docs

retriever, documents = init_rag()
st.sidebar.success(f"Indexed {len(documents)} support documents.")

# File Upload
uploaded_file = st.file_uploader("Upload Support Tickets (CSV)", type=["csv"])

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(df_input)} tickets.")
    
    if st.button("🚀 Process Tickets"):
        if not api_key:
            st.error("Please provide a Groq API Key in the sidebar.")
        else:
            agent = SupportAgent(api_key)
            results = []
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Table placeholder
            table_placeholder = st.empty()
            
            processed_data = []
            
            start_time = time.time()
            
            for index, row in df_input.iterrows():
                # Batch rate limiting: Sleep 60s every 10 tickets
                if index > 0 and index % 10 == 0:
                    status_text.warning(f"Rate Limit Safeguard: Processed {index} tickets. Sleeping for 60 seconds...")
                    time.sleep(60)
                
                ticket = row.to_dict()
                status_text.info(f"Processing ticket {index+1}/{len(df_input)}: {ticket.get('Subject', 'No Subject')[:50]}...")
                
                # Context Retrieval
                subject_q = str(ticket.get('Subject', '') or '')
                issue_q = str(ticket.get('Issue', '') or '')
                query = f"{subject_q} {issue_q}"
                context_results = retriever.search(query, top_k=3)
                confidence = retriever.get_confidence_score(context_results)
                
                # Agent Processing
                triage_result = agent.process_ticket(ticket, context_results, confidence)
                
                # Merge and append
                final_record = {**ticket, **triage_result}
                processed_data.append(final_record)
                
                # Update UI Table
                display_df = pd.DataFrame(processed_data)[['Issue', 'Subject', 'Company', 'status', 'product_area', 'request_type']]
                
                # Styling
                def style_status(val):
                    color = 'green' if val == 'replied' else 'red'
                    return f'color: {color}; font-weight: bold;'
                
                table_placeholder.dataframe(display_df.style.applymap(style_status, subset=['status']))
                
                # Update progress
                progress_bar.progress((index + 1) / len(df_input))
            
            end_time = time.time()
            status_text.success(f"Triaged {len(df_input)} tickets in {int(end_time - start_time)}s.")
            
            # Summary Statistics
            final_df = pd.DataFrame(processed_data)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Tickets", len(final_df))
            with col2:
                st.metric("Replied", len(final_df[final_df['status'] == 'replied']))
            with col3:
                st.metric("Escalated", len(final_df[final_df['status'] == 'escalated']))
            
            # Download Results
            csv_buffer = io.StringIO()
            final_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Triage Results (CSV)",
                data=csv_buffer.getvalue(),
                file_name="triage_results.csv",
                mime="text/csv"
            )

else:
    st.info("Please upload a CSV file with 'Issue', 'Subject', and 'Company' columns.")
    st.markdown("""
    ### Sample CSV Format:
    ```csv
    Issue,Subject,Company
    "My account is locked","Login Issue",HackerRank
    "How do I reset my password?","Password Reset",Visa
    ```
    """)
