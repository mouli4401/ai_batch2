# ai_batch2
This repository contains my work, projects, and learnings as part of the AI Springboard Internship


Step by Step Explanation of the RAG Pipeline Code
1. Import Libraries

The code starts by importing all the libraries needed for loading documents, splitting text, creating embeddings, building vector stores, and running the RAG chain.

os and Path are used to work with files and directories.

dotenv is used to load API keys from a .env file.

langchain_core.documents and langchain_core.prompts are used for document objects and prompt templates.

langchain_text_splitters is used to split large documents into smaller chunks.

langchain_community.document_loaders is used to load text, PDF, and CSV files.

langchain_community.vectorstores is used to create a FAISS vector index.

langchain_google_genai provides embeddings for your documents.

langchain_groq provides the generative AI model for answering questions.

langchain.chains.combine_documents and langchain.chains are used to create RAG chains.

pandas is used for handling CSV files.

2. Load Environment Variables

The code loads your API keys from a .env file:

GOOGLE_API_KEY is used for embeddings.

GROQ_API_KEY is used for the chat model.

If the keys are not set, the program will stop and show an error.

3. Configuration

This section defines constants that control how your RAG system works:

DOCS_PATH – Folder where your documents are stored.

INDEX_PATH – Path to save the FAISS vector index.

REBUILD_INDEX – Whether to create a new index or load an existing one.

EMBED_MODEL – The model used for creating embeddings.

CHAT_MODEL – The language model used to answer questions.

TOP_K – Number of most relevant documents retrieved for answering.

SEARCH_TYPE – Type of vector search (similarity).

CHUNK_SIZE and CHUNK_OVERLAP – Define chunk size and overlap for splitting documents.

4. Load Documents

The functions find_files and load_documents handle document loading:

find_files scans a folder recursively for .txt, .md, .pdf, and .csv files.

load_documents reads files based on type:

Text and Markdown files use TextLoader.

PDFs use PyPDFLoader.

CSVs are converted row by row into Document objects.

Each Document contains text and metadata such as file source.

5. Split Documents

Large documents are split into smaller chunks using RecursiveCharacterTextSplitter:

Each chunk is no larger than CHUNK_SIZE.

CHUNK_OVERLAP ensures some overlap between chunks to preserve context.

6. Build or Load FAISS Vector Store

FAISS is a library for fast vector similarity search:

build_or_load_faiss creates a new FAISS index if REBUILD_INDEX is True.

Otherwise, it loads an existing index from disk.

GoogleGenerativeAIEmbeddings converts document text into embeddings.

7. Create Retriever

The retriever helps find the most relevant chunks from the FAISS index:

make_retriever sets the number of top results to TOP_K.

The search type is similarity.

8. Build RAG Chain

A RAG (Retrieval-Augmented Generation) chain combines retrieval and generative AI:

make_rag_chain creates a system prompt instructing the AI to only answer using context.

It combines ChatGroq LLM with a document chain.

The retrieval chain fetches the top document chunks and passes them to the language model.

9. Format Sources

format_sources takes a list of documents and returns a readable list of source file names.

Useful for showing citations along with answers.

10. Main Program

The main function handles the workflow:

If REBUILD_INDEX is True:

Scan the document folder.

Load documents.

Split documents into chunks.

Build or load the FAISS vector index.

Create a retriever and a RAG chain.

Enter a loop to repeatedly ask questions.

For each question, fetch an answer from the RAG chain and print sources.

11. Helper Functions for Google Sheets

The ask_question function allows programmatic queries from external apps:

Calls the RAG chain and returns an answer.

Errors are caught and returned as a string.

12. Initialization Outside Main

The code initializes the FAISS vector store and RAG chain outside the main function:

This allows integration with other applications like Google Sheets or Flask.

Summary

Load documents from a folder (txt, md, pdf, csv).

Split large documents into chunks.

Convert text into embeddings using Google Generative AI embeddings.

Build or load a FAISS vector store for similarity search.

Create a retriever that finds top-k relevant chunks.

Combine retriever and ChatGroq LLM into a RAG chain.

Accept user questions and return answers with source files.

Optional: Use ask_question for programmatic queries like Google Sheets.
