import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

# --- Load and embed PDF ---
@st.cache_resource
def load_pdf_embeddings():
    loader = PyPDFLoader("./documents/llama2paper.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create Chroma DB
    db = Chroma.from_documents(chunks, embedding=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever

# --- RAG Query Function ---
def rag_query(query, retriever, llm):
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""You are an expert on Llama2. 
Use the following context to answer the user's question:

Context:
{context}

Question: {query}

Answer:"""

    response = llm.invoke(prompt)
    return response

# --- Streamlit UI ---
st.title("🦙 Llama2 Research Assistant")
st.write("Ask detailed questions about the Llama2 research paper.")

query = st.text_input("Enter your question:")
retriever = load_pdf_embeddings()
llm = OllamaLLM(model="llama2:7b")

if query:
    with st.spinner("Thinking..."):
        try:
            answer = rag_query(query, retriever, llm)
            st.success(answer)
        except Exception as e:
            st.error(f"Error: {e}")
