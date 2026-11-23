## Live Demo: https://shadmaniahrar-medical-encyclopedia.onrender.com/

# Medical Encyclopedia RAG Chatbot

A domain-specific *Retrieval-Augmented Generation (RAG)* web application that answers health and medical reference questions grounded in the Gale Encyclopedia of Medicine. Built with *Python, LangChain, FAISS, HuggingFace Sentence-Transformers, Streamlit, and OpenRouter*, the system prioritizes factual grounding, source transparency, and hallucination control.

---

## Key Features

* *Context-Grounded Medical Q&A:* Retrieves relevant passages from a pre-indexed FAISS vector store before generating answers, keeping responses anchored strictly to the provided medical reference material.
* *Hallucination Control & Safe Fallbacks:* Instructs the LLM to rely solely on retrieved context and explicitly respond with "I don't know based on the provided medical context" when a query falls outside the indexed knowledge base.
* *Source Document Transparency:* Displays retrieved reference chunks alongside generated answers so users can verify where the medical information came from.
* *Interactive Streamlit Interface:* Features a custom-styled chat layout, sidebar branding with medical reference notes, one-click chat reset, and quick-topic buttons (Fever & Chills, High Blood Pressure, Diabetes Basics, Common Cold).
* *Cloud-Ready Architecture:* Decouples inference via OpenRouter APIs and caches embedding/vector-store initialization (@st.cache_resource) for lightweight deployment on cloud platforms such as Render.

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| *Frontend & UI* | Streamlit, Custom CSS |
| *RAG Orchestration* | LangChain (langchain-core, langchain-community, langchain-classic) |
| *Embeddings* | HuggingFace (sentence-transformers/all-MiniLM-L6-v2) |
| *Vector Database* | FAISS (faiss-cpu) |
| *LLM Inference* | OpenRouter API (langchain-openrouter) |
| *Document Parsing* | PyPDF (pypdf) |
| *Deployment* | GitHub, Render |

---

## Project Structure

```text
MEDICAL_CHATBOT/
├── data/                         # Source medical reference PDF (Gale Encyclopedia of Medicine)
├── vectorstore/                  # Persisted FAISS vector index
├── create_memory_for_llm.py      # PDF ingestion, text chunking, and FAISS index creation
├── connect_memory_with_llm.py    # CLI verification script for RAG retrieval & LLM responses
├── medical_bot.py                # Main Streamlit web application
├── medical chatbot.png           # Sidebar branding graphic
├── requirements.txt              # Python dependencies
├── .gitignore                    # Excludes .env and cache artifacts
└── README.md                     # Project documentation


1. Development & Version Control Apps
​Visual Studio Code (VS Code): Used to write, edit, and organize your Python scripts (medical_bot.py, create_memory_for_llm.py, connect_memory_with_llm.py), requirements.txt, .gitignore, and README.md.
​PowerShell (in Terminal): Used to run Git commands, manage files, and install Python packages.
​Web Browser: Used to preview the local Streamlit app (http://localhost:8501) and manage GitHub and Render dashboards.
​Git & GitHub: Used for local version control, repository hosting (shadmaniabrar/medical-encyclopedia-rag-chatbot), and connecting your code to deployment.
​
2. Cloud Hosting & LLM API Platform
​Render: Used as the cloud Web Service platform to host and deploy the Streamlit application.
​OpenRouter: Used as the LLM API gateway to call the language model without running heavy local model weights in production.
​
3. Python Libraries & Frameworks Inside the App
​Streamlit: Builds the interactive web chat UI, sidebar, quick-topic buttons, and session state.
​LangChain (langchain-core, langchain-community, langchain-classic, langchain-huggingface, langchain-openrouter): Connects document retrieval, prompt templates, and the LLM chain.
​FAISS (faiss-cpu): Stores and searches vector embeddings locally from the vectorstore folder.
​Sentence-Transformers / HuggingFace (sentence-transformers/all-MiniLM-L6-v2): Generates text embeddings for semantic similarity search.
​PyPDF (pypdf) & python-dotenv: Reads PDF reference material (such as Gale Encyclopedia of Medicine) and loads environment variables like OPENROUTER_API_KEY.

## How It Works
​1. Knowledge Ingestion (create_memory_for_llm.py): Loads the Gale Encyclopedia of Medicine PDF, splits text into overlapping semantic chunks, generates dense vector embeddings with all-MiniLM-L6-v2, and saves the index locally in vectorstore/.
​2. Semantic Retrieval (medical_bot.py): Converts each user question into an embedding and performs similarity search against the FAISS vector store to fetch the most relevant medical passages.
​3. Grounded Generation: Passes the retrieved passages and user query through a strict medical prompt template to OpenRouter, returning a concise answer paired with source citations.

## Local Setup & Installation
​1. Clone the Repository
git clone [https://github.com/shadmaniabrar/medical-encyclopedia-rag-chatbot.git](https://github.com/shadmaniabrar/medical-encyclopedia-rag-chatbot.git)
cd medical-encyclopedia-rag-chatbot

2. Install Dependencies
pip install -r requirements.txt

3. Configure Environment Variables
​Create a .env file in the project root and add your OpenRouter API key:
OPENROUTER_API_KEY=your_openrouter_api_key_here

4. (Optional) Rebuild the FAISS Vector Store
​If you update or add documents inside data/, regenerate the vector index
python create_memory_for_llm.py

5. Run the Streamlit App
streamlit run medical_bot.py

## Deployment (Render)
​1. Push the repository to GitHub (ensure .env is excluded via .gitignore and vectorstore/ is included).
​2. Create a new Web Service on Render connected to this repository.
​3. Configure the build and start commands:
​Build Command: pip install -r requirements.txt
​Start Command: streamlit run medical_bot.py --server.port $PORT --server.address 0.0.0.0
​4. Add OPENROUTER_API_KEY under Environment Variables and deploy.


​## Disclaimer
​This application is built for educational and informational reference purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider regarding medical conditions.
