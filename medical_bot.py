import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openrouter import ChatOpenRouter

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"
BANNER_IMAGE_PATH = "medical chatbot.png"  # Put a local image here, or skip if not added yet

SAMPLE_MEDICAL_ISSUES = {
    "Fever & Chills": "What are the common causes and general care steps mentioned for fever and chills?",
    "High Blood Pressure": "What information is provided about hypertension and blood pressure management?",
    "Diabetes Basics": "What key points are covered regarding diabetes symptoms and management?",
    "Common Cold": "What does the medical reference say about common cold symptoms and relief?",
}


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"],
    )


def load_llm(openrouter_model, openrouter_api_key):
    return ChatOpenRouter(
        model=openrouter_model,
        temperature=0.5,
        max_tokens=512,
        api_key=openrouter_api_key,
    )


def run_rag_query(query_text: str):
    CUSTOM_PROMPT_TEMPLATE = """
    Use the pieces of information provided in the context to answer user's question.
    If you dont know the answer, just say that you dont know, dont try to make up an answer.
    Dont provide anything out of the given context

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """

    OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

    if not OPENROUTER_API_KEY:
        st.error("OPENROUTER_API_KEY not found in .env file.")
        return

    vectorstore = get_vectorstore()
    if vectorstore is None:
        st.error("Failed to load the vector store")
        return

    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(
            openrouter_model=OPENROUTER_MODEL,
            openrouter_api_key=OPENROUTER_API_KEY,
        ),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
    )

    response = qa_chain.invoke({"query": query_text})
    result = response["result"]
    source_documents = response["source_documents"]

    # Two-line gap between the answer and Source Documents
    result_to_show = f"{result}<br><br>*Source Documents:*<br>{source_documents}"

    st.chat_message("assistant").markdown(result_to_show, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": result_to_show})


def main():
    st.set_page_config(page_title="Medical chatbot Assistant", page_icon="🩺", layout="wide")

    st.markdown(
    """
    <style>
        /* Pull the bottom chat input bar slightly upward */
        [data-testid="stChatInput"] {
            bottom: 36px;
        }
        /* Add bottom padding to the main container so messages aren't hidden behind it */
        .stMainBlockContainer {
            padding-bottom: 110px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

    # Sidebar with picture, info, and reset button
    with st.sidebar:
        if os.path.exists(BANNER_IMAGE_PATH):
            st.image(BANNER_IMAGE_PATH, caption="Medical Reference Assistant", use_container_width=True)
        else:
            st.info("Tip: Place a medical image at assets/medical_banner.png to display a sidebar picture.")

        st.markdown("### About Medical chatbot")
        st.caption(
            "Answers are retrieved from your indexed medical reference documents. "
            "This tool is for informational purposes only and is not a substitute for medical advice."
        )

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.title("🩺 Medical Chatbot")
    st.caption("Ask a medical question or pick a common topic below.")

    # Quick-select medical issue buttons
    st.markdown("*Common Medical Topics:*")
    cols = st.columns(len(SAMPLE_MEDICAL_ISSUES))
    selected_topic_query = None

    for idx, (label, sample_q) in enumerate(SAMPLE_MEDICAL_ISSUES.items()):
        if cols[idx].button(label, use_container_width=True):
            selected_topic_query = sample_q

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"], unsafe_allow_html=True)

    chat_input_query = st.chat_input("Ask your question here")
    active_query = chat_input_query or selected_topic_query

    if active_query:
        st.chat_message("user").markdown(active_query)
        st.session_state.messages.append({"role": "user", "content": active_query})

        try:
            with st.spinner("Searching medical knowledge base..."):
                run_rag_query(active_query)
        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()