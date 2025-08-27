import os
from pathlib import Path
from dotenv import load_dotenv

# LangChain core
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# Text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document loaders
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Vector store
from langchain_community.vectorstores import FAISS

# LLMs / embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

# RAG chains
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GOOGLE_API_KEY:
    raise SystemExit("❌ GOOGLE_API_KEY not set in .env")
if not GROQ_API_KEY:
    raise SystemExit("❌ GROQ_API_KEY not set in .env")

# ───────────────────────────────
# CONFIG
# ───────────────────────────────
DOCS_PATH = Path("./my_docs")
INDEX_PATH = Path("./faiss_index")
REBUILD_INDEX = True
EMBED_MODEL = "models/embedding-001"
CHAT_MODEL = "llama-3.1-8b-instant"
TOP_K = 4
SEARCH_TYPE = "similarity"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# ───────────────────────────────
# STEP 1) Load documents
# ───────────────────────────────
def find_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    exts = {".txt", ".md", ".pdf", ".csv"}
    return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts]
import pandas as pd

def load_documents(paths: list[Path]) -> list[Document]:
    docs: list[Document] = []
    for p in paths:
        try:
            if p.suffix.lower() in {".txt", ".md"}:
                docs.extend(TextLoader(str(p), encoding="utf-8").load())
            elif p.suffix.lower() == ".pdf":
                docs.extend(PyPDFLoader(str(p)).load())
            elif p.suffix.lower() == ".csv":
                df = pd.read_csv(str(p))

                # Optional: limit rows for testing
                df = df.head(50)

                # Convert each row into a document
                for _, row in df.iterrows():
                    content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
                    metadata = {"source": str(p)}
                    docs.append(Document(page_content=content, metadata=metadata))
        except Exception as e:
            print(f"[WARN] Could not load {p}: {e}")
    return docs

# ───────────────────────────────
# STEP 2) Split documents
# ───────────────────────────────
def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)

# ───────────────────────────────
# STEP 3) Build / Load FAISS
# ───────────────────────────────
def build_or_load_faiss(chunks: list[Document], rebuild: bool) -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    if rebuild:
        print("🔁 Building FAISS index...")
        vs = FAISS.from_documents(chunks, embeddings)
        INDEX_PATH.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(INDEX_PATH))
        return vs
    print("📦 Loading FAISS index...")
    return FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)

# ───────────────────────────────
# STEP 4) Retriever
# ───────────────────────────────
def make_retriever(vectorstore: FAISS):
    return vectorstore.as_retriever(search_type=SEARCH_TYPE, search_kwargs={"k": TOP_K})

# ───────────────────────────────
# STEP 5) RAG Chain
# ───────────────────────────────
def make_rag_chain(retriever):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. ONLY answer from the given context. "
                   "If not in context, say 'I don't know'. Cite sources."),
        ("human", "Question:\n{input}\n\nContext:\n{context}")
    ])
    llm = ChatGroq(model=CHAT_MODEL, temperature=0.2)
    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, doc_chain)

# ───────────────────────────────
# STEP 6) Format Sources
# ───────────────────────────────
def format_sources(ctx: list[Document]) -> str:
    lines = []
    for d in ctx:
        src = d.metadata.get("source") or "unknown"
        lines.append(f"- {Path(src).name}")
    return "\n".join(lines)

# ───────────────────────────────
# STEP 7) Main
# ───────────────────────────────
def main():
    chunks: list[Document] = []

    if REBUILD_INDEX:
        print(f"📁 Scanning docs in {DOCS_PATH.resolve()}...")
        files = find_files(DOCS_PATH)
        if not files:
            raise SystemExit("❌ No valid documents found.")
        docs = load_documents(files)
        print(f"✂️ Splitting {len(docs)} documents into chunks...")
        chunks = split_documents(docs)

    vectorstore = build_or_load_faiss(chunks, rebuild=REBUILD_INDEX)
    retriever = make_retriever(vectorstore)
    rag_chain = make_rag_chain(retriever)

    while True:
        question = input("\n❓ Enter your question (or 'exit' to quit): ").strip()
        if question.lower() == "exit":
            print("👋 Goodbye!")
            break
        result = rag_chain.invoke({"input": question})
        print("\n🧠 Answer:\n", result.get("answer", "No answer."))
        ctx = result.get("context", [])
        if ctx:
            print("\n📚 Sources:\n", format_sources(ctx))
            
vectorstore = build_or_load_faiss([], rebuild=False) 
retriever = make_retriever(vectorstore)
rag_chain = make_rag_chain(retriever)

def ask_question(query: str) -> str:
    """Wrapper for Sheets integration"""
    try:
        result = rag_chain.invoke({"input": query})
        return result["answer"]
    except Exception as e:
        return f"⚠️ Error: {e}"


if __name__ == "__main__":
    main()
