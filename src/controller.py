"""
Controller module for the Gemini RAG Service API endpoints.
"""
import logging
import re
from typing import List, Dict, Any, Optional
from fastapi import HTTPException
from pydantic import BaseModel

import config
import src.llm as llm
import src.db_operations as db_operations

# Configure logging
logger = logging.getLogger(__name__)

# Request and response models
class QueryRequest(BaseModel):
    question: str
    top_k: int = config.RAG_CONFIG['default_top_k']
    similarity_threshold: float = config.RAG_CONFIG['default_similarity_threshold']

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []

async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

def detect_query_type(question: str) -> Dict[str, Optional[str]]:
    """
    Detect the type of query (instructor-specific, course-specific, or general).
    
    Args:
        question: The question text
        
    Returns:
        A dictionary with detected entity types and values
    """
    # Initialize result
    result = {
        'type': 'general',
        'instructor_name': None,
        'course_id': None,
        'course_name': None
    }
    
    # Check for instructor patterns
    instructor_patterns = [
        r"professor\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
        r"prof\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
        r"(Dr\.\s+[A-Z][a-z]+)",
        r"instructor\s+([A-Z][a-z]+)",
        r"how\s+(?:does|is|was)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s+(?:as a|as an|as|a|an|the)\s+(?:instructor|professor|teacher))?",
        r"([A-Z][a-z]+\s+[A-Z][a-z]+)'s\s+(?:teaching|class|course)"
    ]
    
    for pattern in instructor_patterns:
        match = re.search(pattern, question)
        if match:
            result['type'] = 'instructor'
            result['instructor_name'] = match.group(1)
            break
    
    # Check for course patterns
    course_patterns = [
    r"course\s+([A-Z]+\d+[A-Z]*)",
    r"class\s+([A-Z]+\d+[A-Z]*)",
    r"([A-Z]+\d+[A-Z]*)\s+(?:course|class)",
    r"(?:the\s+)?([A-Z]+\d+[A-Z]*)",  # Match just the course code
    r"(?:how|what)(?:\s+is|\s+was|\s+about)\s+(?:the\s+)?([A-Z]+\d+[A-Z]*(?:\s+\w+){0,4}?\s+(?:course|class))",
    r"(?:the\s+)?((?:[A-Z][a-z]+\s+){1,4}(?:course|class|department))"
    ]
    
    for pattern in course_patterns:
        match = re.search(pattern, question)
        if match:
            if result['type'] == 'instructor':
                result['type'] = 'instructor_course'
            else:
                result['type'] = 'course'
            
            course_text = match.group(1)
            # Check if it's a course code (like CS101) or a course name (like Computer Science)
            if re.match(r"[A-Z]+\d+[A-Z]*", course_text):
                result['course_id'] = course_text
            else:
                result['course_name'] = course_text
            break
    
    return result

def select_prompt_template(query_type: str, entities: Dict[str, Optional[str]], question: str, contexts: List[str]) -> str:
    """
    Select the appropriate prompt template based on the query type.
    
    Args:
        query_type: The type of query ('instructor', 'course', 'instructor_course', or 'general')
        entities: Dictionary with detected entities
        question: The original question
        contexts: The retrieved contexts
        
    Returns:
        A formatted prompt for the LLM
    """
    context_text = "\n\n".join(contexts)
    
    if query_type == 'instructor':
        prompt = config.RAG_CONFIG['instructor_prompt_template'].format(
            instructor_name=entities['instructor_name'],
            context=context_text,
            question=question
        )
    elif query_type == 'course':
        course_id = entities['course_id'] or "[Course ID unknown]"
        course_name = entities['course_name'] or "[Course name unknown]"
        prompt = f"""
        You are a helpful AI assistant analyzing course evaluations.
        Based ONLY on the context provided, summarize student opinions about the course {course_id} ({course_name}).
        
        Pay special attention to any comments or ratings specifically about this course.
        If the context mentions the course code '{course_id}' or course name '{course_name}', 
        prioritize that information in your answer.
        
        CONTEXT:
        {context_text}
        
        QUESTION:
        {question}
        
        Give a helpful response focusing specifically on course {course_id}. If there is no 
        specific information about this course in the context, clearly state this limitation.
        """
    elif query_type == 'instructor_course':
        course_id = entities['course_id'] or "[Course ID unknown]"
        course_name = entities['course_name'] or "[Course name unknown]"
        prompt = f"""You are a helpful assistant analyzing course evaluations for college courses.
        Based ONLY on the following course evaluation data for {course_id} ({course_name}) taught by {entities['instructor_name']}, 
        provide a summary that addresses the specific question below.
        
        CONTEXT:
        {context_text}
        
        QUESTION:
        {question}
        
        Give a concise, helpful response in a conversational tone. If the data shows information from multiple 
        semesters, summarize general trends while noting significant differences.
        If the data doesn't provide enough information on a particular aspect, acknowledge this limitation.
        Provide specific examples from the data when available."""
    else:  # general
        prompt = config.RAG_CONFIG['general_prompt_template'].format(
            context=context_text,
            question=question
        )
    
    return prompt

async def query(request: QueryRequest):
    """Process a query using RAG with Gemini."""
    question = request.question
    top_k = request.top_k
    threshold = request.similarity_threshold
    
    # Detect query type
    entities = detect_query_type(question)
    query_type = entities['type']
    logger.info(f"Detected query type: {query_type}")
    logger.info(f"Entities: {entities}")
    
    # Step 1: Generate embedding for the question
    embedding = llm.generate_embedding(question)
    
    if not embedding:
        raise HTTPException(status_code=500, detail="Failed to generate embedding")
    
    # Step 2: Retrieve relevant context from database - now with query_type and entities
    contexts, sources = db_operations.retrieve_context(
        query_embedding=embedding, 
        top_k=top_k, 
        similarity_threshold=threshold,
        query_type=query_type,  # Pass the query type
        entities=entities       # Pass the entities
    )
    
    if not contexts:
        return QueryResponse(
            answer="I couldn't find relevant information to answer your question.",
            sources=[]
        )
    
    # Step 3: Select appropriate prompt template
    prompt = select_prompt_template(query_type, entities, question, contexts)
    
    # Step 4: Generate answer using Gemini
    answer = llm.generate_answer_with_prompt(prompt)
    
    return QueryResponse(
        answer=answer,
        sources=sources
    )