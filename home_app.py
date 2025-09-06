import streamlit as st

def main():
    """Main function for the application home page."""
    st.title("📚 AI Extractor User Guide")
    st.markdown("---")
    
    with st.expander("🔍 Getting Started", expanded=True):
        st.markdown("""
        1. **Select a mode** from the sidebar
        2. **Upload documents** or images
        3. **Configure settings** for your extraction
        4. **Review results** and download reports
        """)
    
    with st.expander("📊 Table Extraction Guide"):
        st.markdown("""
        **How to extract tables:**
        - Upload PDFs or images containing tables
        - The AI will detect and extract tabular data
        - Download results as Excel or CSV
        
        **Tips:**
        - Use clear, high-contrast documents
        - Avoid merged cells for best results
        """)
    
    with st.expander("📋 Document Extraction Guide"):
        st.markdown("""
        **How to extract document data:**
        1. Select document type (invoice, ID card, etc.)
        2. Upload single or multiple files
        3. Review extracted fields
        4. Download individual or batch reports
        
        **Batch processing:**
        - Process up to 20 documents at once
        - Download results while processing continues
        - ZIP download available for full batches
        """)
    
    with st.expander("⚙️ Advanced Features"):
        st.markdown("""
        **Custom Fields:**
        - Add specific fields you want extracted
        
        **Validation Rules:**
        - Set requirements for critical fields
        
        **Quality Control:**
        - Review confidence scores for each field
        """)
    
    with st.expander("❓ Troubleshooting"):
        st.markdown("""
        **Common issues:**
        - Blurry images: Use minimum 300 DPI scans
        - Complex layouts: Try single-page processing
        - Missing fields: Verify document type selection
        
        **Support:**
        - Contact support@aiextractor.com for assistance
        """)
    
    st.markdown("---")
    st.caption("© 2025 AI Extractor | Version 2.5")

if __name__ == "__main__":
    main()
