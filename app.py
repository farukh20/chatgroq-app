import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Load API keys
groq_api_key = os.environ['GROQ_API_KEY']

st.title("ChatGroq with Llama 3")

# LLM and Prompt setup
llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.2-3b-preview")
prompt = ChatPromptTemplate.from_template(
    """Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    context: {context}
    question: {input}"""
)

# Ensure session state initialization
if "vectors" not in st.session_state:
    st.session_state.vectors = None  # Initialize as None

# Create 'temp_dir' if it does not exist
temp_dir = "temp_dir"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Function to create embeddings and vector store
def vector_embedding(uploaded_pdf):
    st.session_state.embeddings = OllamaEmbeddings(model="llama3.2")
    
    # Save uploaded PDF to a temporary location
    pdf_path = os.path.join(temp_dir, uploaded_pdf.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    
    # Load documents using PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Check the loaded document structure
    if not docs or not isinstance(docs, list):
        st.error("No documents were loaded. Please check the PDF file.")
        return

    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:10])

    # Create FAISS vector store
    st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)

# Upload button for PDF file
uploaded_pdf = st.file_uploader("Upload your PDF file", type="pdf")

# Button to create vector embeddings
if st.button("Create Document Embeddings") and uploaded_pdf is not None:
    vector_embedding(uploaded_pdf)
    st.write("Vector-store DB is ready")
elif uploaded_pdf is None:
    st.write("Please upload a PDF file before creating embeddings")

# Input prompt for user query
prompt1 = st.text_input("Enter your question from documents:")

if prompt1:
    if st.session_state.vectors is not None:  # Check if vectors are initialized
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write("Response time:", time.process_time() - start)
        st.write(response['answer'])

        # Display relevant document chunks in an expander
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                # Check for the correct structure in the doc
                if hasattr(doc, 'page_content'):
                    st.write(doc.page_content)  # Access text via attribute
                elif isinstance(doc, dict) and 'page_content' in doc:
                    st.write(doc['page_content'])  # Access text via dictionary key
                else:
                    st.write("Document format is not recognized")
                st.write("-----------------")
    else:
        st.write("Please click 'Create Document Embeddings' first to create the vector store.")

