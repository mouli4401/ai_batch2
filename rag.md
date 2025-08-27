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

The code loads your API keys from a .env file.

GOOGLE_API_KEY is used for embeddings.

GROQ_API_KEY is used for the chat model.

If the keys are not set, the program will stop and show an error.

3. Configuration

This section defines constants that control how your RAG system works:

DOCS_PATH is the folder where your documents are stored.

INDEX_PATH is where the FAISS vector index will be saved.

REBUILD_INDEX decides whether to create a new index or load an existing one.

EMBED_MODEL is the model used for creating embeddings.

CHAT_MODEL is the language model used to answer questions.

TOP_K is the number of most relevant documents retrieved for answering.

SEARCH_TYPE is the type of vector search (similarity in this case).

CHUNK_SIZE and CHUNK_OVERLAP define how large each document chunk is and how much overlap exists between chunks.

4. Load Documents

The function find_files scans a folder recursively to find all text, markdown, PDF, and CSV files.

The function load_documents reads the files:

For text and markdown files, it uses TextLoader.

For PDFs, it uses PyPDFLoader.

For CSV files, it reads the first 50 rows and converts each row into a Document object.

Each document contains the text and some metadata like the file source.

5. Split Documents

Large documents are split into smaller chunks using RecursiveCharacterTextSplitter.

This ensures that each chunk is no larger than CHUNK_SIZE.

CHUNK_OVERLAP defines how much of the previous chunk overlaps with the next to preserve context.

6. Build or Load FAISS Vector Store

FAISS is a library for fast vector similarity search.

build_or_load_faiss creates a new FAISS index if REBUILD_INDEX is True.

Otherwise, it loads an existing index from disk.

It uses GoogleGenerativeAIEmbeddings to convert document text into embeddings.

7. Create Retriever

The retriever is a helper that finds the most relevant chunks from the FAISS index for a given question.

make_retriever sets the number of top results to TOP_K.

The search type is similarity.

8. Build RAG Chain

A RAG chain is the combination of retrieval and generative AI.

make_rag_chain creates a system prompt that instructs the AI to only answer using context.

It combines a language model (ChatGroq) with a document chain.

The retrieval chain fetches the top document chunks and passes them to the language model.

9. Format Sources

The format_sources function takes a list of documents and returns a readable string showing the names of the source files.

This is useful to show citations along with answers.

10. Main Program

The main function handles the workflow:

If REBUILD_INDEX is True:

Scan the documents folder.

Load documents.

Split documents into chunks.

Build or load the FAISS vector index.

Create a retriever and a RAG chain.

Enter a loop to repeatedly ask questions.

For each question, it fetches an answer from the RAG chain and prints the sources.

11. Helper Functions for Google Sheets

The code also defines a wrapper ask_question for integration with Google Sheets.

It calls the RAG chain and returns an answer.

Errors are caught and returned as a string.

12. Initialization Outside Main

The last part initializes the FAISS vector store and RAG chain once, so you can use them outside the command-line interface.

This allows integration with other apps like Google Sheets or Flask.

Summary

Load documents from a folder (txt, md, pdf, csv).

Split large documents into chunks.

Convert text into embeddings using Google Generative AI embeddings.

Build or load a FAISS vector store for similarity search.

Create a retriever that finds top-k relevant chunks.

Combine retriever and ChatGroq LLM into a RAG chain.

Accept user questions and return answers with source files.

Optional: Use ask_question for programmatic queries like Google Sheets.
