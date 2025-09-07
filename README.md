RAG-Powered Document Analysis Tool
This project is a full-stack web application that allows users to upload any PDF document and ask questions about its content. It uses a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, source-grounded answers, powered by Google's Gemini LLM.

The application processes the uploaded PDF in real-time, creating a temporary knowledge base to ensure that answers are always relevant to the provided document and that user data remains private.

(You can replace this with a real screenshot of your running application)

Key Features:-
Dynamic PDF Upload: Users can upload any PDF file directly through the web interface.

On-the-Fly RAG Pipeline: The RAG system is built dynamically for each uploaded document, ensuring data privacy.

Accurate, Source-Grounded Answers: Leverages Google's Gemini model to generate answers based only on the context from the document, minimizing hallucinations.

Modern & Responsive UI: A clean, dark-themed, and mobile-friendly interface built with Flask and standard web technologies.

Tech Stack:-
Backend: Python, Flask

LLM Framework: LangChain

LLM & Embeddings: Google Gemini (gemini-2.0-flash, models/embedding-001)

Vector Store: ChromaDB (In-memory)

Frontend: HTML, CSS, JavaScript
