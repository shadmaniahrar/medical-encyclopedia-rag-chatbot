import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Load environment variables from .env
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found. Please set it in your .env file.")

# Step 1: Setup LLM with OpenRouter
OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"  # Replace with any valid OpenRouter model slug

def load_llm(model_name: str):
    llm = ChatOpenRouter(
        model=model_name,
        temperature=0.5,
        max_tokens=512,
        api_key=OPENROUTER_API_KEY,
    )
    return llm

# Step 2: Custom Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know; don't try to make up an answer.
Don't provide anything outside of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template: str):
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"],
    )

# Step 3: Load FAISS Vector Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Create RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(OPENROUTER_MODEL),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
)

# Step 5: Query from Terminal
if __name__ == "__main__":
    user_query = input("Write Query Here: ")
    response = qa_chain.invoke({"query": user_query})
    print("\nRESULT:\n", response["result"])
    print("\nSOURCE DOCUMENTS:\n", response["source_documents"])