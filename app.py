import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.rag import RAGPipeline

st.set_page_config(page_title="Support Ticket RAG Evaluator", layout="wide")

@st.cache_resource
def load_rag():
    try:
        return RAGPipeline()
    except ValueError as ve:
        st.error(str(ve))
        return None
    except Exception as e:
        st.error(f"Failed to load RAG pipeline. Did you run the ingestion and vectorstore scripts? Error: {e}")
        return None

def main():
    st.title("🎫 Support Ticket RAG Evaluator")
    
    rag = load_rag()
    if not rag:
        return
        
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Ask", "🔍 Evidence", "📊 Evaluation", "📁 Dataset"])
    
    with tab1:
        st.header("Ask Support")
        col1, col2 = st.columns([3, 1])
        with col2:
            top_k = st.slider("Top-K Tickets", min_value=1, max_value=5, value=3)
            threshold = st.slider("Distance Threshold", min_value=0.5, max_value=2.0, value=1.5, step=0.1)
        
        with col1:
            query = st.text_input("Enter your question:")
            if st.button("Generate Answer") and query:
                with st.spinner("Generating answer..."):
                    result = rag.generate_answer(query, top_k=top_k, distance_threshold=threshold)
                    
                    if result['status'] == 'refused':
                        st.warning(result['answer'])
                    elif result['status'] == 'error':
                        st.error(result['answer'])
                    else:
                        st.success(result['answer'])
                        
                    st.caption(f"Latency: {result['latency']:.2f}s | Status: {result['status']}")
                    
                    if result['citations']:
                        st.markdown("**Citations:**")
                        for cite in result['citations']:
                            st.markdown(f"- `{cite}`")
                            
                    st.session_state['last_result'] = result

    with tab2:
        st.header("Retrieved Evidence")
        if 'last_result' in st.session_state:
            res = st.session_state['last_result']
            if res['retrieved']:
                for idx, doc in enumerate(res['retrieved']):
                    with st.expander(f"Ticket: {doc['ticket_id']} (Distance: {doc['distance']:.3f})"):
                        st.markdown(f"**Category:** {doc['category']}")
                        st.text(doc['text'])
            else:
                st.info("No evidence retrieved.")
        else:
            st.info("Ask a question in the 'Ask' tab to see evidence.")
            
    with tab3:
        st.header("Evaluation Metrics")
        results_file = "eval/results.csv"
        if os.path.exists(results_file):
            df_eval = pd.read_csv(results_file)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Hit@3", f"{df_eval['hit_at_3'].mean()*100:.1f}%")
            c2.metric("Hit@5", f"{df_eval['hit_at_5'].mean()*100:.1f}%")
            c3.metric("Refusal Accuracy", f"{df_eval['refusal_correct'].mean()*100:.1f}%")
            c4.metric("Avg Latency", f"{df_eval['latency'].mean():.2f}s")
            
            st.dataframe(df_eval)
        else:
            st.warning("Evaluation results not found. Run `python src/evaluation.py` first.")
            
    with tab4:
        st.header("Dataset Overview")
        if os.path.exists("artifacts/metadata.parquet"):
            df_meta = pd.read_parquet("artifacts/metadata.parquet")
            st.markdown(f"**Total Tickets:** {len(df_meta)}")
            
            st.markdown("**Categories Distribution**")
            import plotly.express as px
            category_counts = df_meta['category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            fig = px.bar(category_counts, x='Category', y='Count', title="Tickets by Category")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("**Sample Records**")
            st.dataframe(df_meta.head(100))
        else:
            st.warning("Dataset not found. Run ingestion and vectorstore scripts.")

if __name__ == "__main__":
    main()
