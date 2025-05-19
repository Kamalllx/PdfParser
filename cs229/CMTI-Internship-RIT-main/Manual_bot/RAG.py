import os
import streamlit as st
import tempfile
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq 
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(page_title="Manual Bot",  layout="wide")

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file"""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def process_documents(uploaded_files):
    """Process documents to create a vector store"""
    text = ""
    

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Extract text from all files
    for i, file in enumerate(uploaded_files):
        status_text.text(f"Processing {file.name}...")
        ext = os.path.splitext(file.name)[1].lower()
        if ext == ".pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                temp_path = tmp.name
            pdf_text = extract_text_from_pdf(temp_path)
            text += pdf_text
            os.unlink(temp_path)
        elif ext in [".txt", ".md"]:
            text += file.getvalue().decode("utf-8")
        else:
            st.error(f"Unsupported file format: {ext}")
        

        progress_bar.progress((i + 1) / len(uploaded_files))
    
    if not text.strip():
        st.error("No text found in uploaded documents.")
        return None

    # Split text into chunks
    status_text.text("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    if not chunks:
        st.error("Failed to split text into chunks.")
        return None
        
    st.info(f"Split documents into {len(chunks)} chunks")

    # Create vector store using FAISS 
    try:
        status_text.text("Creating vector embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vectorstore = FAISS.from_texts(chunks, embeddings)  
        status_text.text("Vector store created successfully!")
        progress_bar.progress(100)
        return vectorstore
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

def initialize_conversation(vectorstore):
    """Initialize the conversation chain"""
    # Verify API key is available
    if not os.getenv("GROQ_API_KEY"):
        st.error("Groq API key not found. Please enter it in the sidebar.")
        return None
    
    try:
        llm = ChatGroq(
            model_name="llama3-70b-8192",
            temperature=0.2,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            memory=memory,
            verbose=True
        )
        
        return conversation
    except Exception as e:
        st.error(f"Error initializing conversation: {e}")
        return None

def main():
    initialize_session_state()
    
    st.title("📚 Manual Bot")
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or MD files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True
        )
        if uploaded_files:
            process_button = st.button("Process Documents")
            if process_button:
                with st.spinner("Processing documents..."):
                    st.session_state.vectorstore = process_documents(uploaded_files)
                    if st.session_state.vectorstore:
                        st.session_state.conversation = initialize_conversation(st.session_state.vectorstore)
                        if st.session_state.conversation:
                            st.session_state.processed_files = [f.name for f in uploaded_files]
                            st.success("✅ Documents processed and ready for chat!")
        
        # API key input
        st.divider()
        api_key = st.text_input("Enter your Groq API key:", type="password")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
            if st.session_state.vectorstore and not st.session_state.conversation:
                with st.spinner("Initializing conversation..."):
                    st.session_state.conversation = initialize_conversation(st.session_state.vectorstore)
                    if st.session_state.conversation:
                        st.success("✅ Conversation initialized!")

        st.divider()
        st.markdown("### Instructions")
        st.markdown("""
        1. Upload documents
        2. Click "Process Documents"
        3. Start chatting with your documents
        """)

    # Main chat interface
    if st.session_state.conversation:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        user_input = st.chat_input("Ask a question about your documents:")
        if user_input:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.conversation.run(user_input)
                        st.markdown(response)
                        # Add assistant response to chat
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Error generating response: {e}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        st.info("👈 Upload documents and process them to start chatting.")

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sample Questions:")
            st.markdown("""
            Once your documents are processed, you can ask questions like:
            - What are the key sections in this document?
            - Can you summarize the main points?
            - What is the process described for [specific topic]?
            """)
        
        with col2:
            st.subheader("Troubleshooting:")
            st.markdown("""
            If you're not getting answers:
            1. Make sure your Groq API key is valid
            2. Check that documents were processed successfully
            3. Try shorter, more specific questions
            4. Ensure your documents contain readable text
            """)

if __name__ == "__main__":
    main()