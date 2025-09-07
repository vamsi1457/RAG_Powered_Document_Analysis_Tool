import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Define the path for the persistent vector store
PERSISTENT_DIRECTORY = "chroma_db_persistent"

class RAGPipeline:
    def __init__(self):
        """
        Initializes the RAG pipeline by loading all necessary components.
        """
        print("Initializing RAG Pipeline...")
        self.api_key = self._load_api_key()
        self.embeddings = self._initialize_embeddings()
        self.vector_store = self._load_vector_store()
        self.llm = self._initialize_llm()
        self.qa_chain = self._create_qa_chain()
        print("✅ RAG Pipeline Initialized Successfully.")

    def _load_api_key(self):
        api_key = "AIzaSyCs5N1cThVZPqjSzexYDWRJ4AiERs_mTlc"
        if not api_key:
            raise ValueError("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
        return api_key

    def _initialize_embeddings(self):
        return GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=self.api_key)

    def _load_vector_store(self):
        if not os.path.exists(PERSISTENT_DIRECTORY):
            raise FileNotFoundError(f"Persistent vector store not found at '{PERSISTENT_DIRECTORY}'. Please run the setup script first.")
        return Chroma(persist_directory=PERSISTENT_DIRECTORY, embedding_function=self.embeddings)

    def _initialize_llm(self):
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=self.api_key)

    def _create_qa_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={'k': 8})
        return RetrievalQA.from_chain_type(
            self.llm,
            retriever=retriever,
            return_source_documents=False
        )

    def get_answer(self, query: str) -> str:
        """
        Gets an answer for a given query using the RAG chain.
        """
        try:
            response = self.qa_chain({"query": query})
            return response.get("result", "Sorry, I couldn't find an answer.")
        except Exception as e:
            print(f"Error during prediction: {e}")
            return "An error occurred while trying to get an answer."

