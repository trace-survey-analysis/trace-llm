"""
Main application module for the Gemini RAG Service.
"""
import logging
import uvicorn
from fastapi import FastAPI

import config
import src.controller as controller

# Configure logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Gemini RAG Service")

# Register routes
app.add_api_route("/health", controller.health_check, methods=["GET"])
app.add_api_route("/query", controller.query, methods=["POST"], response_model=controller.QueryResponse)

def main():
    """Main entry point of the application."""
    logger.info(f"Starting Gemini RAG Service on {config.SERVICE_CONFIG['host']}:{config.SERVICE_CONFIG['port']}")
    
    # Log configuration details
    logger.info(f"Embedding model: {config.EMBEDDING_CONFIG['model_name']}")
    logger.info(f"Gemini model: {config.GEMINI_CONFIG['model']}")
    logger.info(f"Database host: {config.DB_CONFIG['host']}")
    
    # Start the FastAPI application
    uvicorn.run(
        app, 
        host=config.SERVICE_CONFIG['host'], 
        port=config.SERVICE_CONFIG['port']
    )

if __name__ == "__main__":
    main()