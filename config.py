"""
Configuration module for the Gemini RAG Service.
"""
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Database configuration
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '5432'),
    'database': os.environ.get('DB_NAME', 'trace'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', ''),
}

# Vector embedding configuration
EMBEDDING_CONFIG = {
    'schema': os.environ.get('EMBEDDING_SCHEMA', 'vectors'),
    'source_schema': os.environ.get('SOURCE_SCHEMA', 'trace'),
    'tables': {
        'comment': os.environ.get('COMMENT_EMBEDDING_TABLE', 'comment_embeddings'),
        'rating': os.environ.get('RATING_EMBEDDING_TABLE', 'rating_embeddings'),
        'instructor': os.environ.get('INSTRUCTOR_EMBEDDING_TABLE', 'instructor_embeddings'),
        'course': os.environ.get('COURSE_EMBEDDING_TABLE', 'course_embeddings'),
    },
    'search_tables': os.environ.get('SEARCH_TABLES', 'comment,rating,instructor,course').split(','),
    'vector_column': os.environ.get('EMBEDDING_VECTOR_COLUMN', 'embedding'),
    'model_name': os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
    'embedding_dim': int(os.environ.get('EMBEDDING_DIM', '384')),  # Dimension for all-MiniLM-L6-v2
    'batch_size': int(os.environ.get('BATCH_SIZE', '32')),
    'max_retries': int(os.environ.get('MAX_RETRIES', '3')),
    'retry_delay': int(os.environ.get('RETRY_DELAY', '5')),  # seconds
}

# Gemini configuration
GEMINI_CONFIG = {
    'api_key': os.environ.get('GEMINI_API_KEY'),
    'model': os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash'),
    'temperature': float(os.environ.get('GEMINI_TEMPERATURE', '0.2')),
    'top_p': float(os.environ.get('GEMINI_TOP_P', '0.8')),
    'top_k': int(os.environ.get('GEMINI_TOP_K', '40')),
    'max_output_tokens': int(os.environ.get('GEMINI_MAX_OUTPUT_TOKENS', '1024')),
}

# RAG configuration
RAG_CONFIG = {
    'default_top_k': int(os.environ.get('RAG_DEFAULT_TOP_K', '10')),
    'default_similarity_threshold': float(os.environ.get('RAG_DEFAULT_SIMILARITY_THRESHOLD', '0.6')),
    # Prompts for different types of queries
    'instructor_prompt_template': os.environ.get('INSTRUCTOR_PROMPT_TEMPLATE', 
        """You are a helpful assistant analyzing teaching evaluations for college professors.
        Based ONLY on the following course evaluation data for {instructor_name}, provide a summary 
        of this instructor's teaching style, strengths, and areas for improvement.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        
        Give a concise, helpful response focused on the instructor's teaching abilities. Be conversational and natural.
        If the data doesn't provide enough information on a particular aspect, acknowledge this limitation.
        Provide specific examples from the data when available."""),
    
    'course_prompt_template': os.environ.get('COURSE_PROMPT_TEMPLATE',
        """You are a helpful assistant analyzing course evaluations for college courses.
        Based ONLY on the following course evaluation data for {course_id} ({course_name}), provide a summary 
        that addresses the specific question below.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        
        Give a concise, helpful response in a conversational tone. If the data shows information from multiple 
        semesters or instructors, summarize general trends while noting significant differences.
        If the data doesn't provide enough information on a particular aspect, acknowledge this limitation.
        Provide specific examples from the data when available."""),
    
    'general_prompt_template': os.environ.get('GENERAL_PROMPT_TEMPLATE',
        """You are a helpful assistant analyzing course evaluations for college courses.
        Based ONLY on the following course evaluation data, provide a response that addresses the specific question below.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        
        Give a concise, helpful response in a conversational tone. Focus only on what can be directly inferred from the data.
        If the data doesn't provide enough information, acknowledge this limitation rather than speculating.
        Provide specific examples from the data when available."""),
}

# Service configuration
SERVICE_CONFIG = {
    'port': int(os.environ.get('SERVICE_PORT', '8000')),
    'host': os.environ.get('SERVICE_HOST', '0.0.0.0'),
}

# Initialize Gemini if API key is available
if GEMINI_CONFIG['api_key']:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_CONFIG['api_key'])
    logger.info(f"Gemini configured with model: {GEMINI_CONFIG['model']}")
else:
    logger.warning("GEMINI_API_KEY not set - LLM functionality will be limited")