import os
import sys
import gradio as gr
from pathlib import Path
from typing import List

from llama_index.llms.google_genai import GoogleGenAI

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.readers.file import PyMuPDFReader
#from llama_index.postprocessor.flashrank_rerank import FlashRankRerank
from llama_index.postprocessor.flashrank_rerank import FlashRankRerank

# RAG Setup
DATA_DIR = "./data"
PDF_FILE = "prospectus.pdf"
STORAGE_DIR = "./storage"

# Creating directories if they don't exist
Path(DATA_DIR).mkdir(exist_ok=True)
Path(STORAGE_DIR).mkdir(exist_ok=True)

pdf_path = Path(DATA_DIR) / PDF_FILE
if not pdf_path.exists():
    print(f"Error: PDF file '{PDF_FILE}' not found in the '{DATA_DIR}' directory.")
    sys.exit()

# Gemini API
#Settings.llm = GoogleGenAI(model="models/gemini-1.5-flash", api_key=os.environ.get("GOOGLE_API_KEY"))
Settings.llm = GoogleGenAI(model="gemini-2.5-flash", api_key=os.environ.get("GOOGLE_API_KEY"))

# Embedding
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cpu"
)

# Initializing RAG Engine
def get_query_engine():
    """Initializes and returns the RAG query engine."""
    
    print("Initializing RAG engine...")

    if not os.path.exists(STORAGE_DIR) or not os.listdir(STORAGE_DIR):
        print("Storage directory not found or is empty. Ingesting new document.")
        
        # Document Loading
        loader = PyMuPDFReader()
        documents = loader.load_data(file_path=pdf_path)
        
        # Node Parsing
        node_parser = SentenceWindowNodeParser(
            window_size=3,
            sentence_splitter=SentenceSplitter(chunk_size=512, chunk_overlap=20)
        )
        
        pipeline = IngestionPipeline(
            documents=documents,
            transformations=[
                node_parser,
                Settings.embed_model
            ]
        )
        
        # Vector Storing
        print("Ingesting document. This may take a while on a slower system...")
        try:
            nodes = pipeline.run()
            print(f"Successfully processed {len(nodes)} document chunks.")
            index = VectorStoreIndex(nodes)
            index.storage_context.persist(persist_dir=STORAGE_DIR)
            print("Index successfully built and persisted.")
        except Exception as e:
            print(f"An error occurred during ingestion: {e}")
            import traceback
            traceback.print_exc()
            sys.exit()

    else:
        print("Loading existing index from storage...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context=storage_context)
        print("Index loaded.")

    # Setting up the query engine with better parameters and a reranker
    reranker = FlashRankRerank(top_n=3)  # Initialize the reranker to return the top 5 most relevant chunks

    query_engine = index.as_query_engine(
        similarity_top_k=7,  # Retrieve a larger pool of potential chunks for the reranker to choose from
        streaming=True,
        node_postprocessors=[
            reranker
        ]
    )
    
    return query_engine

query_engine = get_query_engine()

# Gradio UI
def stream_response(message, history):
    """
    Function to handle user queries and stream the response back.
    """
    yield "Thinking..."

    try:
        # response = query_engine.query(message)
        # full_response = str(response)

        # partial = ""  # build up response progressively
        # for token in full_response.split():
        #     partial += token + " "
        #     yield partial
        streaming_response = query_engine.query(message)
        
        # The first yielded value ("Thinking...") will be overwritten by the generator below.
        # This yields tokens directly from the LLM's streaming response.
        for token in streaming_response.response_gen:
            yield token
    except Exception as e:
        error_message = f"An error occurred: {e}. Please check your LLM configuration or try a different query."
        print(error_message)
        yield error_message 

demo = gr.ChatInterface(
    fn=stream_response,
    title="University Of Karachi Prospectus Assistant",
    description="Ask questions about your prospectus PDF.",
    examples=[
        "What is the admission requirements?",
        "Tell me about University Of Karachi.",
        "Tell me about the eligibility to get enrolled in University Of Karachi."
    ],
    chatbot=gr.Chatbot(height=400),
    textbox=gr.Textbox(placeholder="Enter your question here..."),
)

#if __name__ == "__main__":
#    print("\nStarting Gradio UI...")
#    demo.launch(share=False)
if __name__ == "__main__":
    # Get the PORT assigned by Railway or default to 7860 for local testing
    PORT = int(os.environ.get('PORT', 7860)) 
    
    # Crucial for cloud deployment: listen on all network interfaces ("0.0.0.0")
    demo.launch(
        server_name="0.0.0.0", 
        server_port=PORT,
        share=False
    )
