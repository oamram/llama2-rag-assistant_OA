# scripts/rag_llama2_manual_pdf.py

from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM  # updated import

# PDF reading library
import PyPDF2

# -------------------------
# 1. Load and extract PDF text
# -------------------------
pdf_path = "./documents/llama2paper.pdf"
text = ""
with open(pdf_path, "rb") as f:
    reader = PyPDF2.PdfReader(f)
    for page in reader.pages:
        text += page.extract_text()

# -------------------------
# 2. Split text into chunks
# -------------------------
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_text(text)

# -------------------------
# 3. Create embeddings
# -------------------------
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -------------------------
# 4. Load into Chroma
# -------------------------
db = Chroma.from_texts(chunks, embedding_function)
retriever = db.as_retriever(search_kwargs={"k": 1})

# -------------------------
# 5. Load Llama2 model
# -------------------------
llm = OllamaLLM(model="AIresearcher:latest")  # updated class

# -------------------------
# 6. Simple RAG query function
# -------------------------
def rag_query(query):
    # Use protected method with required run_manager argument
    docs = retriever._get_relevant_documents(query, run_manager=None)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {query}"
    
    # Use .generate() instead of calling the object
    result = llm.generate([prompt])
    
    # Extract the text from the first generation
    return result.generations[0][0].text

# -------------------------
# 7. Example query
# -------------------------
query = "How has Llama 2 improved model convergence speed during training?"
answer = rag_query(query)

print("Query:", query)
print("Answer:", answer)
