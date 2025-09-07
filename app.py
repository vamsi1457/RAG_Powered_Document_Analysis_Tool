from flask import Flask, render_template, request
from rag_handler import RAGPipeline # Import our new RAG handler

# App Initialization
app = Flask(__name__)

#  Global RAG Pipeline Initialization
# This part runs only once when the server starts.
try:
    # This creates an instance of our RAGPipeline class from rag_handler.py
    rag_pipeline = RAGPipeline()
    RAG_PIPELINE_READY = True
except Exception as e:
    print(f"❌ Critical Error initializing RAG pipeline: {e}")
    rag_pipeline = None
    RAG_PIPELINE_READY = False

#  Flask Routes 

@app.route('/')
def home():
    """Renders the main home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the user."""
    # Check if the RAG pipeline failed to load
    if not RAG_PIPELINE_READY:
        return render_template('index.html', prediction_text="Error: The AI model is not ready. Please check the server logs.")

    # Get the user's question from the HTML form
    user_question = request.form.get('question')
    
    # If the user submitted an empty form, just show the home page
    if not user_question:
        return render_template('index.html')
    
    # Call the get_answer method from our rag_handler to get the result
    answer = rag_pipeline.get_answer(user_question)
    
    # Render the page again, passing the answer back to the UI
    return render_template('index.html', prediction_text=answer, user_question=user_question)

#  Main Execution
if __name__ == '__main__':
    # The app will run in debug mode for easier development
    app.run(debug=True)

