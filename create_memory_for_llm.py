from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# Step 1: Load PDF files
DATA_PATH = "data/"

def load_pdf_files(data):
  loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
  documents = loader.load()
  return documents
# Step 2: Load documents
documents = load_pdf_files(DATA_PATH)

# Step 3: Check document content
if documents:
  print("First page content:")
  print(documents[0].page_content[:500])
else:
  print("No PDF documents found. Check your data folder.")


# step 2: create chunks
def create_chunks(extracted_data):
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=500, chunk_overlap=50
  )
  text_chunks = text_splitter.split_documents(extracted_data)
  return text_chunks


text_chunks = create_chunks(extracted_data=documents)
# print("Length of text chunks: ", len(text_chunks))


# step 3: create vector embedding
def get_embedding_model():
  embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
  )
  return embedding_model
embedding_model = get_embedding_model()


# step 4: store embedding in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)