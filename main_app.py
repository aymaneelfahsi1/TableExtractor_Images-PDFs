import streamlit as st
from PIL import Image
import sys
import os

# Set page config MUST be the first Streamlit command
st.set_page_config(
    page_title="AI Document & Table Extractor", 
    page_icon="📄", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application with navigation between Table and Document extraction."""
    
    # Sidebar navigation
    with st.sidebar:
        st.title("📄 AI Extractor")
        st.markdown("---")
        
        # Page selection
        page = st.selectbox(
            "🎯 Choose Application Mode",
            ["🏠 Home", "📊 Table Extraction", "📋 Document Extraction"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        if page == "🏠 Home":
            st.info("Welcome to AI Extractor!")
        elif page == "📊 Table Extraction":
            st.info("Extract tables from images and PDFs with hierarchical structure detection.")
        elif page == "📋 Document Extraction":
            st.info("Extract structured data from documents like ID cards, invoices, and forms using AI-powered OCR.")
        
        st.markdown("---")
        st.markdown("**Powered by Gemini AI**")
    
    # Route to appropriate page
    if page == "🏠 Home":
        import home_app
        home_app.main()
    elif page == "📊 Table Extraction":
        import table_app as tab_app
        tab_app.main()
    elif page == "📋 Document Extraction":
        import document_app as doc_app
        doc_app.main()

if __name__ == "__main__":
    main()