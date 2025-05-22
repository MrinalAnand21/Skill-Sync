from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
import os

# Define paths for storing FAISS index and docstore
INDEX_PATH = './faiss_index'
DOCSTORE_PATH = './docstore.pkl'

# Create text splitter and embeddings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Function to save FAISS index
def save_index(vector_store):
    vector_store.save_local(INDEX_PATH)

# Function to load FAISS index with dangerous deserialization enabled
def load_index():
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# Check if the FAISS index already exists
if os.path.exists(INDEX_PATH):
    # Load the FAISS index and documents from disk
    vector_store = load_index()
else:
    # Create new FAISS index and document store
    index = faiss.IndexFlatL2(len(embeddings.embed_query("test_query")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    # Directory containing PDFs
    directory = './pdf_documents'
    texts = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory,filename)
        loader = PyMuPDFLoader(file_path)
        doc = loader.load()
        texts += text_splitter.split_documents(doc)

    # Generate UUIDs for documents
    uuids = [str(uuid4()) for _ in range(len(texts))]

    # Add documents to vector store and save the index
    vector_store.add_documents(documents=texts, id=uuids)
    save_index(vector_store)  # Save index to disk

# Function to perform similarity search using the loaded vector store
def get_related_content_in_pdf(query, k=1):
    results = vector_store.similarity_search(query, k)
    return results

# Example usage
# results = get_related_content_in_pdf("python programming", 1)
# for res in results:
#     print(f"* {res.page_content}")
