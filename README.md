# University of Karachi Prospectus Assistant 

An intelligent Retrieval-Augmented Generation (RAG) virtual assistant designed to answer student queries interactively based on the official University of Karachi (UBIT) prospectus. 

This application processes the official university PDF, chunks the data, generates embeddings, and uses a Large Language Model (LLM) to provide highly accurate, context-aware answers to user questions via a clean chat interface.

##  Features
* Accurate RAG Architecture: Prevents LLM hallucinations by restricting answers strictly to the embedded prospectus data.
* Smart Document Processing: Utilizes PyMuPDF for clean text extraction and SentenceSplitter for semantic chunking.
* Streaming Responses: Real-time token streaming for a fast, responsive user experience.
* Persistent Storage: Automatically caches the vector index locally to drastically reduce loading times on subsequent runs.
* Interactive UI: Built with Gradio's ChatInterface for an intuitive, conversational user experience.
* Cloud Ready: Fully compatible with Hugging Face Spaces deployment.

##  Tech Stack
* Language: Python
* Framework: LlamaIndex
* LLM: Google Gemini 2.5 Flash
* Embedding Model: BAAI/bge-small-en-v1.5 (via Hugging Face)
* Frontend: Gradio

##  Project Structure
To run this project locally, ensure your directory looks like this:

├── data/
│   └── prospectus.pdf    <-- (Place the official PDF here)
├── app.py                <-- (Main application script)
├── requirements.txt
├── .env                  <-- (Your secret API keys, DO NOT commit this)
└── .gitignore
