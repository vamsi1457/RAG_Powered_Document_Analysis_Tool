import os
from flask import Flask, render_template, request
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# --- App Initialization ---
app = Flask(__name__)

# --- Global variable to hold our AI pipeline ---
rag_chain = None
PIPELINE_READY = False

# --- Function to Initialize the RAG Pipeline ---
def initialize_rag_pipeline():
    """
    This function loads all the necessary components for the RAG pipeline.
    It's designed to run only once when the server starts.
    """
    global rag_chain, PIPELINE_READY
    try:
        print("--- Starting RAG Pipeline Initialization ---")

        # 1. Load Google API Key
        print("Step 1: Loading Google API Key...")
        api_key = "AIzaSyCs5N1cThVZPqjSzexYDWRJ4AiERs_mTlc"
        if not api_key:
            raise ValueError("🔴 CRITICAL: GOOGLE_API_KEY environment variable not found!")
        print("✅ API Key loaded successfully.")

        # 2. Load the PDF document
        print("Step 2: Loading PDF document...")
        pdf_path = "termsofusesVB.pdf"
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"🔴 CRITICAL: PDF file not found at '{pdf_path}'. Make sure it's in your GitHub repo.")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"✅ PDF loaded successfully. Total pages: {len(documents)}")

        # 3. Split the document into chunks
        print("Step 3: Splitting document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Document split into {len(chunks)} chunks.")

        # 4. Initialize embedding model
        print("Step 4: Initializing embedding model...")
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=api_key)
        print("✅ Embedding model initialized.")

        # 5. Create an in-memory vector store from the chunks
        print("Step 5: Creating in-memory vector store...")
        vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
        retriever = vector_store.as_retriever(search_kwargs={'k': 8})
        print("✅ Vector store created.")

        # 6. Initialize the LLM
        print("Step 6: Initializing the LLM...")
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
        print("✅ LLM initialized.")

        # 7. Create the RAG chain
        print("Step 7: Creating the RAG chain...")
        rag_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
        print("🎉 RAG pipeline is fully initialized and ready! 🎉")
        PIPELINE_READY = True

    except Exception as e:
        # This will print the exact error to your Render logs
        print(f"❌❌❌ An error occurred during pipeline initialization: {e} ❌❌❌")
        PIPELINE_READY = False

# --- Initialize the pipeline when the app starts ---
initialize_rag_pipeline()

# --- Flask Routes ---
@app.route('/')
def home():
    """Renders the main home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the user."""
    if not PIPELINE_READY or not rag_chain:
        error_message = "Error: The AI model is not ready. Please check the server logs for initialization errors."
        return render_template('index.html', prediction_text=error_message)

    user_question = request.form.get('question')
    if not user_question:
        return render_template('index.html')

    try:
        response = rag_chain.invoke({"query": user_question})
        answer = response.get("result", "Sorry, I couldn't find an answer in the document.")
    except Exception as e:
        # This will print prediction-time errors to your logs
        print(f"Error during prediction: {e}")
        answer = "An error occurred while trying to get an answer. Please check the server logs."

    return render_template('index.html', prediction_text=answer, user_question=user_question)

# --- Main Execution ---
if __name__ == '__main__':
    # For Render, it's better to let gunicorn handle the server
    # This block is mainly for local testing
    app.run(debug=False)


