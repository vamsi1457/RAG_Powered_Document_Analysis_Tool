import os
import shutil
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Updated import to remove the deprecation warning
from langchain_community.vectorstores import Chroma

# --- Configuration ---
# Make sure your PDF file is in the same directory as this script
PDF_FILE_PATH = "termsofusesVB.pdf"
PERSISTENT_DIRECTORY = "chroma_db_persistent"

def generate_vector_store():
    """
    Processes a PDF document and creates a persistent ChromaDB vector store.
    This script needs to be run only once to set up the knowledge base.
    """
    try:
        # 1. Get Google API Key from environment variables
        api_key = "AIzaSyCs5N1cThVZPqjSzexYDWRJ4AiERs_mTlc"
        if not api_key:
            raise ValueError("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
        print("API Key loaded successfully.")

        # 2. Check if the persistent directory already exists and remove it for a fresh start
        if os.path.exists(PERSISTENT_DIRECTORY):
            print(f"Directory '{PERSISTENT_DIRECTORY}' already exists. Removing it to create a new one.")
            shutil.rmtree(PERSISTENT_DIRECTORY)
        
        # 3. Load the PDF Document
        print(f"Loading document: {PDF_FILE_PATH}...")
        loader = PyPDFLoader(PDF_FILE_PATH)
        documents = loader.load()
        if not documents:
            raise ValueError("Could not load any content from the PDF. Please check the file path and content.")
        print(f"Document loaded. It has {len(documents)} page(s).")

        # 4. Split the Document into Manageable Chunks
        print("Splitting document into text chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        print(f"Document split into {len(chunks)} chunks.")

        # 5. Initialize the Google Embedding Model
        print("Initializing the embedding model...")
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=api_key)
        print("Embedding model initialized.")

        # 6. Create and Persist the Vector Store
        # This is the most important step. Chroma will create the database and
        # save its files to the 'persist_directory'.
        print(f"Creating and saving the vector store to '{PERSISTENT_DIRECTORY}'...")
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            persist_directory=PERSISTENT_DIRECTORY
        )
        
        print("\nSuccess!")
        print(f"The vector store has been created and saved in the '{PERSISTENT_DIRECTORY}' folder.")
        print("You can now run the 'app.py' to start your web application.")

    except Exception as e:
        print(f"\n An error occurred during the setup: {e}")

# --- Main Execution Block ---
if __name__ == '__main__':
    generate_vector_store()

