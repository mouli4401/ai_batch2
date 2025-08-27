# import os
# from dotenv import load_dotenv
# from groq import Groq
# load_dotenv()

# client = Groq(
#     api_key=os.environ.get("GROQ_API_KEY"),
# )

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Explain the importance of fast language models",
#         }
#     ],
#     model="llama-3.3-70b-versatile",
# )

# print(chat_completion.choices[0].message.content)


#RAG
import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings  

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"D:\Infosys\ai_batch2\service_account.json"

load_dotenv()


# Google sheets integration




DOCS_PATH     = Path("./my_docs")          
INDEX_PATH    = Path("./faiss_index")    
REBUILD_INDEX = True                       
EMBED_MODEL   = "models/embedding-001"    
CHAT_MODEL = "llama-3.1-8b-instant"       
TOP_K         = 3                          
SEARCH_TYPE   = "similarity"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 120

QUESTION = "Summarize the documents in 3 lines."


# STEP 1) API KEY CHECK
if not os.environ.get("GROQ_API_KEY"):
    raise SystemExit(" GROQ_API_KEY not set. Run: export GROQ_API_KEY='your_key'")


# STEP 2) LOAD DOCUMENTS
def find_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    exts = {".txt", ".md", ".pdf"}
    return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def load_documents(paths: list[Path]) -> list[Document]:
    docs: list[Document] = []
    for p in paths:
        try:
            if p.suffix.lower() in {".txt", ".md"}:
                docs.extend(TextLoader(str(p), encoding="utf-8").load())
            elif p.suffix.lower() == ".pdf":
                docs.extend(PyPDFLoader(str(p)).load())
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")
    return docs


# STEP 3) SPLIT INTO CHUNKS

def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)



# STEP 4) FAISS INDEX

def build_or_load_faiss(chunks: list[Document], rebuild: bool) -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

    if rebuild:
        print(" Building FAISS index from documents...")
        vs = FAISS.from_documents(chunks, embeddings)
        INDEX_PATH.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(INDEX_PATH))
        print(f" Saved index to: {INDEX_PATH.resolve()}")
        return vs

    print(f" Loading FAISS index from: {INDEX_PATH.resolve()}")
    vs = FAISS.load_local(
        str(INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(" Loaded FAISS index.")
    return vs



# STEP 5) RETRIEVER

def make_retriever(vectorstore: FAISS):
    return vectorstore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={"k": TOP_K},
    )



# STEP 6) RAG CHAIN

def make_rag_chain(retriever):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a careful assistant. ONLY answer from the given context. "
         "If not in context, say 'I don't know'. Always cite sources."),
        ("human", "Question:\n{input}\n\nContext:\n{context}")
    ])

    # Use Groq LLM here
    llm = ChatGroq(model=CHAT_MODEL, temperature=0.2)

    doc_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, doc_chain)
    return rag_chain



# STEP 7) FORMAT SOURCES

def format_sources(ctx: list[Document]) -> str:
    lines = []
    for d in ctx:
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
        page = d.metadata.get("page")
        name = Path(src).name
        lines.append(f"- {name}" + (f" (page {page})" if page else ""))
    return "\n".join(lines)


# STEP 8) MAIN

def main():
    chunks: list[Document] = []

    if REBUILD_INDEX:
        print(f"📁 Scanning docs in {DOCS_PATH.resolve()}")
        files = find_files(DOCS_PATH)
        if not files:
            raise SystemExit(" No .txt/.md/.pdf files found.")
        docs = load_documents(files)
        print(f" Splitting {len(docs)} docs...")
        chunks = split_documents(docs)

    vectorstore = build_or_load_faiss(chunks, rebuild=REBUILD_INDEX)
    retriever = make_retriever(vectorstore)
    rag = make_rag_chain(retriever)

    # Interactive loop
    while True:
        question = input("\n Enter your question (or 'exit' to quit): ").strip()
        if question.lower() == "exit":
            print("👋 Goodbye!")
            break

        result = rag.invoke({"input": question})
        print("\n Answer:\n" + str(result["answer"]).strip())

        ctx = result.get("context", [])
        if ctx:
            print("\n Sources:")
            print(format_sources(ctx))



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





