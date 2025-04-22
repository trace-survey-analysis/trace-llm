"""
LLM module for interacting with the Gemini model.
"""
import logging
from typing import List, Optional
import numpy as np
import google.generativeai as genai
import os
import config

try:
    # First try to import sentence-transformers if available
    from sentence_transformers import SentenceTransformer
    USE_SENTENCE_TRANSFORMER = True
except ImportError:
    # Fall back to scikit-learn TF-IDF as a simpler alternative
    from sklearn.feature_extraction.text import TfidfVectorizer
    USE_SENTENCE_TRANSFORMER = False

# Configure logging
logger = logging.getLogger(__name__)

# Lazy-loaded models
_embedding_model = None
_tfidf_vectorizer = None

def get_embedding_model():
    """Lazy-load the embedding model."""
    global _embedding_model, _tfidf_vectorizer
    
    if USE_SENTENCE_TRANSFORMER:
        if _embedding_model is None:
            try:
                _embedding_model = SentenceTransformer(config.EMBEDDING_CONFIG['model_name'])
                logger.info(f"Loaded SentenceTransformer model: {config.EMBEDDING_CONFIG['model_name']}")
            except Exception as e:
                logger.error(f"Error loading SentenceTransformer model: {e}")
                raise
        return _embedding_model
    else:
        # Use TF-IDF vectorizer as a fallback
        if _tfidf_vectorizer is None:
            try:
                _tfidf_vectorizer = TfidfVectorizer(max_features=config.EMBEDDING_CONFIG['embedding_dim'])
                # Initialize with a sample text
                _tfidf_vectorizer.fit(["This is a sample document to initialize the vectorizer"])
                logger.info(f"Initialized TF-IDF vectorizer with {config.EMBEDDING_CONFIG['embedding_dim']} features")
            except Exception as e:
                logger.error(f"Error initializing TF-IDF vectorizer: {e}")
                raise
        return _tfidf_vectorizer

def generate_embedding(text: str) -> Optional[List[float]]:
    """Generate embedding for the input text using the available model."""
    try:
        if USE_SENTENCE_TRANSFORMER:
            # Generate embedding using sentence-transformers
            model = get_embedding_model()
            embedding = model.encode(text)
            
            # Verify embedding dimension
            if len(embedding) != config.EMBEDDING_CONFIG['embedding_dim']:
                logger.warning(
                    f"Generated embedding dimension ({len(embedding)}) doesn't match "
                    f"configured dimension ({config.EMBEDDING_CONFIG['embedding_dim']})"
                )
            
            return embedding.tolist()
        else:
            # Generate embedding using TF-IDF
            vectorizer = get_embedding_model()
            
            # Transform the text
            embedding_sparse = vectorizer.transform([text])
            embedding_dense = embedding_sparse.toarray()[0]
            
            # Normalize the embedding to unit length for cosine similarity
            norm = np.linalg.norm(embedding_dense)
            if norm > 0:
                embedding_dense = embedding_dense / norm
            
            # Ensure the embedding has exactly the expected dimension
            if len(embedding_dense) < config.EMBEDDING_CONFIG['embedding_dim']:
                embedding_dense = np.pad(embedding_dense, 
                                       (0, config.EMBEDDING_CONFIG['embedding_dim'] - len(embedding_dense)))
            elif len(embedding_dense) > config.EMBEDDING_CONFIG['embedding_dim']:
                embedding_dense = embedding_dense[:config.EMBEDDING_CONFIG['embedding_dim']]
            
            return embedding_dense.tolist()
            
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

def generate_answer(question: str, contexts: List[str]) -> str:
    """Generate an answer using Gemini based on the question and retrieved contexts."""
    try:
        if not config.GEMINI_CONFIG['api_key']:
            return "Gemini API key not configured. Please set the GEMINI_API_KEY environment variable."
        
        # Prepare the prompt with retrieved contexts
        context_text = "\n\n".join([f"Context {i+1}:\n{context}" for i, context in enumerate(contexts)])
        
        prompt = f"""
        You are a helpful AI assistant tasked with answering questions about course evaluations at a university.
        
        CONTEXT:
        {context_text}
        
        QUESTION:
        {question}
        
        Instructions:
        1. Answer the question based only on the information provided in the context.
        2. If the context doesn't contain enough information to answer the question, state that you don't have sufficient information.
        3. Provide a concise and accurate answer in a conversational tone.
        4. Do not make up information that is not in the context.
        5. When discussing instructors or courses, be balanced and fair in your assessment.
        
        ANSWER:
        """
        
        return generate_answer_with_prompt(prompt)
        
    except Exception as e:
        logger.error(f"Error generating answer with Gemini: {e}")
        return f"Error generating answer: {str(e)}"

def generate_answer_with_prompt(prompt: str) -> str:
    """Generate an answer using Gemini with a custom prompt."""
    try:
        if not config.GEMINI_CONFIG['api_key']:
            return "Gemini API key not configured. Please set the GEMINI_API_KEY environment variable."
        
        # Call Gemini API with Gemini 2.0 Flash specific configuration
        model = genai.GenerativeModel(config.GEMINI_CONFIG['model'])
        
        # Configure generation parameters for Gemini
        generation_config = {
            "temperature": config.GEMINI_CONFIG['temperature'],
            "top_p": config.GEMINI_CONFIG['top_p'],
            "top_k": config.GEMINI_CONFIG['top_k'],
            "max_output_tokens": config.GEMINI_CONFIG['max_output_tokens']
        }
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Extract and return the answer
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Error generating answer with Gemini: {e}")
        return f"Error generating answer: {str(e)}"