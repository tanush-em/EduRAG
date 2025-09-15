import os
import json
import glob
from typing import List, Dict, Any
from pathlib import Path

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="NoteQuery Pro API", version="1.0.0")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and index
model = None
index = None
documents = []
index_path = Path("faiss_index")
notes_path = Path("../notes")

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    answer_type: str  # "2_mark" or "14_mark"

class AnswerResponse(BaseModel):
    direct_answer: str
    explanation: str
    summary: str

def load_notes() -> List[Dict[str, str]]:
    """Load all text files from the notes directory."""
    documents = []
    if not notes_path.exists():
        return documents
    
    for file_path in glob.glob(str(notes_path / "*.txt")):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                documents.append({
                    "content": content,
                    "filename": Path(file_path).name
                })
    
    return documents

def create_embeddings(texts: List[str]) -> np.ndarray:
    """Create embeddings for the given texts."""
    global model
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    embeddings = model.encode(texts)
    return embeddings

def build_faiss_index():
    """Build or rebuild the FAISS index from notes."""
    global index, documents
    
    print("Loading notes...")
    documents = load_notes()
    
    if not documents:
        print("No notes found in the notes directory.")
        return
    
    print(f"Found {len(documents)} notes")
    
    # Create embeddings
    texts = [doc["content"] for doc in documents]
    embeddings = create_embeddings(texts)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    # Save index
    index_path.mkdir(exist_ok=True)
    faiss.write_index(index, str(index_path / "index.faiss"))
    
    # Save document metadata
    with open(index_path / "documents.json", 'w') as f:
        json.dump(documents, f, indent=2)
    
    print(f"FAISS index built with {index.ntotal} documents")

def load_faiss_index():
    """Load existing FAISS index if available."""
    global index, documents
    
    index_file = index_path / "index.faiss"
    docs_file = index_path / "documents.json"
    
    if index_file.exists() and docs_file.exists():
        try:
            index = faiss.read_index(str(index_file))
            with open(docs_file, 'r') as f:
                documents = json.load(f)
            print(f"Loaded FAISS index with {index.ntotal} documents")
        except Exception as e:
            print(f"Error loading index: {e}")
            build_faiss_index()
    else:
        build_faiss_index()

def retrieve_relevant_docs(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve relevant documents for the query."""
    global model, index, documents
    
    if index is None or not documents:
        return []
    
    # Create query embedding
    query_embedding = create_embeddings([query])
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = index.search(query_embedding, k)
    
    relevant_docs = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(documents):
            relevant_docs.append({
                "content": documents[idx]["content"],
                "filename": documents[idx]["filename"],
                "score": float(score)
            })
    
    return relevant_docs

def get_llm_response(question: str, context: str, answer_type: str) -> Dict[str, str]:
    """Get response from Groq API with hardcoded format based on answer type."""
    
    if answer_type == "2_mark":
        # Hardcoded prompt for 2-mark answers
        system_prompt = """You are an educational assistant. Answer questions using the provided context from study notes. 
        For 2-mark answers, format your response EXACTLY as follows:
        
        1. Direct Answer: [Provide a clear, concise answer in 1-2 sentences]
        2. Explanation: [Provide at least 4 bullet points or lines with step-by-step reasoning and specific references to the notes]
        3. Summary: [Provide one key takeaway sentence]
        
        The explanation section MUST contain at least 4 bullet points or lines. Use the context provided to give accurate, educational responses."""
        
        user_prompt = f"""Context from notes:
{context}

Question: {question}

This is a 2-mark question. Please answer following the exact format specified above with at least 4 bullet points in the explanation section."""
        
    else:  # 14_mark
        # Hardcoded prompt for 14-mark answers
        system_prompt = """You are an educational assistant. Answer questions using the provided context from study notes. 
        For 14-mark answers, format your response EXACTLY as follows:
        
        1. Direct Answer: [Provide a clear, concise answer in 1-2 sentences]
        2. Explanation: [Provide 3-4 subheadings, each with 6-9 bullet points or lines with detailed reasoning and specific references to the notes]
        3. Summary: [Provide one key takeaway sentence]
        
        The explanation section MUST have 3-4 subheadings, and each subheading MUST contain 6-9 bullet points or lines. Use the context provided to give accurate, educational responses."""
        
        user_prompt = f"""Context from notes:
{context}

Question: {question}

This is a 14-mark question. Please answer following the exact format specified above with 3-4 subheadings in the explanation section, each containing 6-9 bullet points."""

    try:
        # Use Groq API
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        client = Groq(api_key=api_key)
        
        # Adjust max_tokens based on answer type
        max_tokens = 1000 if answer_type == "14_mark" else 500
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        
        answer_text = response.choices[0].message.content
        
    except Exception as e:
        print(f"Groq API error: {e}")
        # Fallback response if API error
        answer_text = f"""1. Direct Answer: I cannot provide a complete answer as the AI service encountered an error.

2. Explanation: The system attempted to process your question "{question}" but encountered an issue with the AI service. The retrieved context from notes was: {context[:200]}...

3. Summary: AI service error - please try again or check the system configuration."""

    # Parse the response into structured format
    try:
        lines = answer_text.strip().split('\n')
        direct_answer = ""
        explanation = ""
        summary = ""
        
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith("1. Direct Answer:"):
                direct_answer = line.replace("1. Direct Answer:", "").strip()
                current_section = "direct"
            elif line.startswith("2. Explanation:"):
                explanation = line.replace("2. Explanation:", "").strip()
                current_section = "explanation"
            elif line.startswith("3. Summary:"):
                summary = line.replace("3. Summary:", "").strip()
                current_section = "summary"
            elif line and current_section:
                if current_section == "direct":
                    direct_answer += " " + line
                elif current_section == "explanation":
                    # Preserve line breaks and formatting for explanation
                    if explanation and not explanation.endswith('\n'):
                        explanation += "\n"
                    explanation += line + "\n"
                elif current_section == "summary":
                    summary += " " + line
        
        # Clean up explanation formatting
        if explanation:
            explanation = explanation.strip()
        
        return {
            "direct_answer": direct_answer or "No direct answer provided.",
            "explanation": explanation or "No explanation provided.",
            "summary": summary or "No summary provided."
        }
        
    except Exception as e:
        print(f"Error parsing response: {e}")
        return {
            "direct_answer": "Error processing response.",
            "explanation": f"Failed to parse the AI response: {str(e)}",
            "summary": "Response parsing failed."
        }

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    print("Starting NoteQuery Pro API...")
    load_faiss_index()

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Answer a question using RAG."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Retrieve relevant documents
    relevant_docs = retrieve_relevant_docs(request.question, k=3)
    
    if not relevant_docs:
        return AnswerResponse(
            direct_answer="No relevant information found in the knowledge base.",
            explanation="The system could not find any relevant notes to answer your question. Please ensure notes are properly indexed.",
            summary="No relevant context available for this question."
        )
    
    # Combine context from relevant documents
    context = "\n\n".join([
        f"From {doc['filename']}: {doc['content']}" 
        for doc in relevant_docs
    ])
    
    # Get LLM response
    response = get_llm_response(request.question, context, request.answer_type)
    
    return AnswerResponse(**response)

@app.post("/reindex")
async def reindex_notes():
    """Rebuild the FAISS index from notes."""
    try:
        build_faiss_index()
        return {"message": f"Successfully reindexed {len(documents)} notes"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reindex: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "indexed_documents": len(documents) if documents else 0,
        "index_loaded": index is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5892)
