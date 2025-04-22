"""
Database operations module for the Gemini RAG Service.
"""
import logging
from typing import List, Tuple, Dict, Any, Optional
from fastapi import HTTPException
import psycopg2
import psycopg2.extras

import config

# Configure logging
logger = logging.getLogger(__name__)

def get_db_connection():
    """Create a connection to the PostgreSQL database."""
    try:
        connection = psycopg2.connect(
            host=config.DB_CONFIG['host'],
            port=config.DB_CONFIG['port'],
            dbname=config.DB_CONFIG['database'],
            user=config.DB_CONFIG['user'],
            password=config.DB_CONFIG['password']
        )
        
        # Set search path to include vectors schema
        cursor = connection.cursor()
        cursor.execute("SET search_path TO public, vectors, trace;")
        cursor.close()
        
        return connection
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

def retrieve_context(
    query_embedding: List[float], 
    top_k: int, 
    similarity_threshold: float,
    query_type: str = 'general',
    entities: Dict[str, Optional[str]] = None
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Retrieve most similar contexts from the vector database.
    
    This function searches across multiple embedding tables (comments, ratings, instructors, courses)
    and joins with the original data tables to fetch the content.
    
    Args:
        query_embedding: The embedding vector of the query
        top_k: Maximum number of results to return
        similarity_threshold: Minimum similarity score to include
        query_type: Type of query ('instructor', 'course', 'instructor_course', 'general')
        entities: Dictionary with detected entities (instructor_name, course_id, course_name)
        
    Returns:
        Tuple of (contexts, sources) where contexts are the text contents
        and sources are metadata about each result
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Initialize results containers
        all_contexts = []
        all_sources = []
        
        # Set default entities if None
        if entities is None:
            entities = {
                'instructor_name': None,
                'course_id': None,
                'course_name': None
            }
        
        # Check if tables exist
        tables_to_check = ['vectors.comment_embeddings', 'vectors.rating_embeddings', 
                           'vectors.instructor_embeddings', 'vectors.course_embeddings']
        existing_tables = []
        
        for table in tables_to_check:
            try:
                cursor.execute(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'vectors' 
                        AND table_name = '{table.split('.')[1]}'
                    );
                """)
                if cursor.fetchone()[0]:
                    existing_tables.append(table.split('.')[1])
            except Exception as e:
                logger.warning(f"Error checking if table {table} exists: {e}")
        
        logger.info(f"Found existing tables: {existing_tables}")
        
        # Process comment embeddings
        if 'comment_embeddings' in existing_tables:
            try:
                # Query for comment embeddings
                query = """
                WITH similarity_results AS (
                    SELECT 
                        e.id as embedding_id,
                        e.comment_id,
                        1 - (e.embedding <=> %s::vector) as similarity
                    FROM vectors.comment_embeddings e
                    WHERE 1 - (e.embedding <=> %s::vector) > %s
                    ORDER BY similarity DESC
                    LIMIT %s
                )
                SELECT 
                    s.comment_id,
                    c.comment_text AS text_content,
                    c.category,
                    c.question_text,
                    s.similarity,
                    c.created_at,
                    co.course_id,
                    co.course_name,
                    co.subject,
                    co.semester,
                    co.year,
                    string_agg(DISTINCT i.name, ', ') AS instructor_names,
                    'comment' AS content_type
                FROM similarity_results s
                JOIN trace.comments c ON s.comment_id = c.id
                JOIN trace.courses co ON c.course_id = co.id
                LEFT JOIN trace.course_instructors ci ON co.id = ci.course_id
                LEFT JOIN trace.instructors i ON ci.instructor_id = i.id
                GROUP BY s.comment_id, c.comment_text, c.category, c.question_text, 
                         s.similarity, c.created_at, co.course_id, co.course_name, 
                         co.subject, co.semester, co.year
                ORDER BY s.similarity DESC
                """
                
                cursor.execute(query, (query_embedding, query_embedding, similarity_threshold, top_k * 2))  # Fetch more, filter later
                results = cursor.fetchall()
                logger.info(f"Found {len(results)} comment results")
                
                # Process comment results
                for row in results:
                    try:
                        # Format contextual information
                        semester_year = f"{row['semester']} {row['year']}" if row['semester'] and row['year'] else ""
                        instructor_info = f" (taught by {row['instructor_names']})" if row.get('instructor_names') else ""
                        
                        # Create context string
                        context = (f"Comment for {row['course_id']} ({row['course_name']}), {semester_year}{instructor_info}:\n"
                                  f"Question: {row['question_text']}\n"
                                  f"Response: {row['text_content']}")
                        
                        # Create metadata
                        metadata = {
                            'id': row['comment_id'],
                            'similarity': float(row['similarity']) if row['similarity'] is not None else 0.0,
                            'type': 'comment',
                            'category': row['category'],
                            'question': row['question_text'],
                            'course_id': row['course_id'],
                            'course_name': row['course_name'],
                            'subject': row['subject'],
                            'semester': row['semester'],
                            'year': row['year'],
                            'instructor_names': row['instructor_names'],
                            'created_at': row['created_at'].isoformat() if row['created_at'] else None
                        }
                        
                        all_contexts.append(context)
                        all_sources.append(metadata)
                    except Exception as e:
                        logger.error(f"Error processing comment result: {e}")
            except Exception as e:
                logger.error(f"Error querying comment embeddings: {e}")
        
        # Process rating embeddings
        if 'rating_embeddings' in existing_tables:
            try:
                # Query for rating embeddings
                query = """
                WITH similarity_results AS (
                    SELECT 
                        e.id as embedding_id,
                        e.rating_id,
                        1 - (e.embedding <=> %s::vector) as similarity
                    FROM vectors.rating_embeddings e
                    WHERE 1 - (e.embedding <=> %s::vector) > %s
                    ORDER BY similarity DESC
                    LIMIT %s
                )
                SELECT 
                    s.rating_id,
                    r.question_text AS text_content,
                    r.category,
                    r.course_mean,
                    r.dept_mean,
                    r.univ_mean,
                    s.similarity,
                    r.created_at,
                    co.course_id,
                    co.course_name,
                    co.subject,
                    co.semester,
                    co.year,
                    string_agg(DISTINCT i.name, ', ') AS instructor_names,
                    'rating' AS content_type
                FROM similarity_results s
                JOIN trace.ratings r ON s.rating_id = r.id
                JOIN trace.courses co ON r.course_id = co.id
                LEFT JOIN trace.course_instructors ci ON co.id = ci.course_id
                LEFT JOIN trace.instructors i ON ci.instructor_id = i.id
                GROUP BY s.rating_id, r.question_text, r.category, r.course_mean, 
                         r.dept_mean, r.univ_mean, s.similarity, r.created_at, 
                         co.course_id, co.course_name, co.subject, co.semester, co.year
                ORDER BY s.similarity DESC
                """
                
                cursor.execute(query, (query_embedding, query_embedding, similarity_threshold, top_k * 2))
                results = cursor.fetchall()
                logger.info(f"Found {len(results)} rating results")
                
                # Process rating results
                for row in results:
                    try:
                        # Format contextual information
                        semester_year = f"{row['semester']} {row['year']}" if row['semester'] and row['year'] else ""
                        instructor_info = f" (taught by {row['instructor_names']})" if row.get('instructor_names') else ""
                        course_score = f"{float(row['course_mean']):.2f}/5.00"
                        dept_score = f"{float(row['dept_mean']):.2f}/5.00"
                        univ_score = f"{float(row['univ_mean']):.2f}/5.00"
                        
                        # Create context string
                        context = (f"Rating for {row['course_id']} ({row['course_name']}), {semester_year}{instructor_info}:\n"
                                  f"Question: {row['text_content']}\n"
                                  f"Course score: {course_score} (Department: {dept_score}, University: {univ_score})")
                        
                        # Create metadata
                        metadata = {
                            'id': row['rating_id'],
                            'similarity': float(row['similarity']) if row['similarity'] is not None else 0.0,
                            'type': 'rating',
                            'category': row['category'],
                            'question': row['text_content'],
                            'course_id': row['course_id'],
                            'course_name': row['course_name'],
                            'subject': row['subject'],
                            'semester': row['semester'],
                            'year': row['year'],
                            'course_mean': float(row['course_mean']),
                            'dept_mean': float(row['dept_mean']),
                            'univ_mean': float(row['univ_mean']),
                            'instructor_names': row['instructor_names'],
                            'created_at': row['created_at'].isoformat() if row['created_at'] else None
                        }
                        
                        all_contexts.append(context)
                        all_sources.append(metadata)
                    except Exception as e:
                        logger.error(f"Error processing rating result: {e}")
            except Exception as e:
                logger.error(f"Error querying rating embeddings: {e}")
                
        # Process instructor embeddings if table exists
        if 'instructor_embeddings' in existing_tables:
            try:
                # Query for instructor embeddings
                query = """
                WITH similarity_results AS (
                    SELECT 
                        e.id as embedding_id,
                        e.instructor_id,
                        1 - (e.embedding <=> %s::vector) as similarity
                    FROM vectors.instructor_embeddings e
                    WHERE 1 - (e.embedding <=> %s::vector) > %s
                    ORDER BY similarity DESC
                    LIMIT %s
                )
                SELECT 
                    s.instructor_id,
                    i.name AS text_content,
                    'Instructor' AS category,
                    'Instructor Information' AS question_text,
                    s.similarity,
                    i.created_at,
                    array_agg(DISTINCT co.course_id) AS course_ids,
                    array_agg(DISTINCT co.course_name) AS course_names,
                    array_agg(DISTINCT co.subject) AS subjects,
                    array_agg(DISTINCT co.semester) AS semesters,
                    array_agg(DISTINCT co.year) AS years,
                    i.name AS instructor_name,
                    'instructor' AS content_type
                FROM similarity_results s
                JOIN trace.instructors i ON s.instructor_id = i.id
                LEFT JOIN trace.course_instructors ci ON i.id = ci.instructor_id
                LEFT JOIN trace.courses co ON ci.course_id = co.id
                GROUP BY s.instructor_id, i.name, s.similarity, i.created_at
                ORDER BY s.similarity DESC
                """
                
                cursor.execute(query, (query_embedding, query_embedding, similarity_threshold, top_k * 2))
                results = cursor.fetchall()
                logger.info(f"Found {len(results)} instructor results")
                
                # Process instructor results
                for row in results:
                    try:
                        # Format course information
                        courses = []
                        if row.get('course_ids') and len(row['course_ids']) > 0:
                            for i in range(len(row['course_ids'])):
                                course_id = row['course_ids'][i] if i < len(row['course_ids']) else None
                                course_name = row['course_names'][i] if i < len(row['course_names']) else None
                                if course_id and course_name:
                                    courses.append(f"{course_id} ({course_name})")
                        
                        courses_str = ", ".join(courses) if courses else "None"
                        
                        # Create context string
                        context = (f"Instructor: {row['text_content']}\n"
                                  f"Teaches courses: {courses_str}")
                        
                        # Create metadata
                        metadata = {
                            'id': row['instructor_id'],
                            'similarity': float(row['similarity']) if row['similarity'] is not None else 0.0,
                            'type': 'instructor',
                            'instructor_name': row['text_content'],
                            'course_ids': row['course_ids'],
                            'course_names': row['course_names'],
                            'subjects': row['subjects'],
                            'created_at': row['created_at'].isoformat() if row['created_at'] else None
                        }
                        
                        all_contexts.append(context)
                        all_sources.append(metadata)
                    except Exception as e:
                        logger.error(f"Error processing instructor result: {e}")
            except Exception as e:
                logger.error(f"Error querying instructor embeddings: {e}")
        
        # Process course embeddings if table exists
        if 'course_embeddings' in existing_tables:
            try:
                # Query for course embeddings
                query = """
                WITH similarity_results AS (
                    SELECT 
                        e.id as embedding_id,
                        e.course_id,
                        1 - (e.embedding <=> %s::vector) as similarity
                    FROM vectors.course_embeddings e
                    WHERE 1 - (e.embedding <=> %s::vector) > %s
                    ORDER BY similarity DESC
                    LIMIT %s
                )
                SELECT 
                    s.course_id,
                    co.course_name AS text_content,
                    co.subject AS category,
                    co.course_id AS question_text,
                    s.similarity,
                    co.created_at,
                    co.course_id AS course_code,
                    co.course_name,
                    co.subject,
                    co.semester,
                    co.year,
                    co.enrollment,
                    co.responses,
                    string_agg(DISTINCT i.name, ', ') AS instructor_names,
                    'course' AS content_type
                FROM similarity_results s
                JOIN trace.courses co ON s.course_id = co.id
                LEFT JOIN trace.course_instructors ci ON co.id = ci.course_id
                LEFT JOIN trace.instructors i ON ci.instructor_id = i.id
                GROUP BY s.course_id, co.course_name, co.subject, co.course_id, 
                         s.similarity, co.created_at, co.course_id, co.enrollment, 
                         co.responses, co.semester, co.year
                ORDER BY s.similarity DESC
                """
                
                cursor.execute(query, (query_embedding, query_embedding, similarity_threshold, top_k * 2))
                results = cursor.fetchall()
                logger.info(f"Found {len(results)} course results")
                
                # Process course results
                for row in results:
                    try:
                        # Format contextual information
                        semester_year = f"{row['semester']} {row['year']}" if row['semester'] and row['year'] else ""
                        instructor_info = f" (taught by {row['instructor_names']})" if row.get('instructor_names') else ""
                        enrollment_info = f"Enrollment: {row['enrollment']} students, {row['responses']} responses"
                        
                        # Create context string
                        context = (f"Course: {row['course_code']} - {row['text_content']} ({row['subject']}), {semester_year}{instructor_info}\n"
                                  f"{enrollment_info}")
                        
                        # Create metadata
                        metadata = {
                            'id': row['course_id'],
                            'similarity': float(row['similarity']) if row['similarity'] is not None else 0.0,
                            'type': 'course',
                            'course_id': row['course_code'],
                            'course_name': row['course_name'],
                            'subject': row['subject'],
                            'semester': row['semester'],
                            'year': row['year'],
                            'enrollment': row['enrollment'],
                            'responses': row['responses'],
                            'instructor_names': row['instructor_names'],
                            'created_at': row['created_at'].isoformat() if row['created_at'] else None
                        }
                        
                        all_contexts.append(context)
                        all_sources.append(metadata)
                    except Exception as e:
                        logger.error(f"Error processing course result: {e}")
            except Exception as e:
                logger.error(f"Error querying course embeddings: {e}")
        
        # Boost scores for exact matches based on query type
        if query_type == 'course' and entities.get('course_id'):
            for i in range(len(all_sources)):
                if all_sources[i]['type'] == 'course' and all_sources[i].get('course_id') == entities['course_id']:
                    # Boost exact course matches
                    all_sources[i]['similarity'] += 0.3
                    logger.info(f"Boosted course {entities['course_id']} by 0.3")
                elif all_sources[i].get('course_id') == entities['course_id']:
                    # Boost content related to the course
                    all_sources[i]['similarity'] += 0.2
                    logger.info(f"Boosted content for course {entities['course_id']} by 0.2")
        
        if query_type == 'instructor' and entities.get('instructor_name'):
            for i in range(len(all_sources)):
                if all_sources[i]['type'] == 'instructor' and all_sources[i].get('instructor_name') == entities['instructor_name']:
                    # Boost exact instructor matches
                    all_sources[i]['similarity'] += 0.3
                    logger.info(f"Boosted instructor {entities['instructor_name']} by 0.3")
                elif all_sources[i].get('instructor_names') and entities['instructor_name'] in all_sources[i]['instructor_names']:
                    # Boost content taught by the instructor
                    all_sources[i]['similarity'] += 0.2
                    logger.info(f"Boosted content for instructor {entities['instructor_name']} by 0.2")
        
        if query_type == 'instructor_course' and entities.get('instructor_name') and entities.get('course_id'):
            for i in range(len(all_sources)):
                if (all_sources[i].get('course_id') == entities['course_id'] and 
                    all_sources[i].get('instructor_names') and 
                    entities['instructor_name'] in all_sources[i]['instructor_names']):
                    # Boost content for this course and instructor combination
                    all_sources[i]['similarity'] += 0.4
                    logger.info(f"Boosted content for course {entities['course_id']} taught by {entities['instructor_name']} by 0.4")
        
        # Sort all results by similarity score
        combined_results = list(zip(all_contexts, all_sources))
        combined_results.sort(key=lambda x: x[1]['similarity'], reverse=True)
        
        # Take only the top k results
        combined_results = combined_results[:top_k]
        
        # Split back into contexts and sources
        contexts, sources = zip(*combined_results) if combined_results else ([], [])
        
        cursor.close()
        conn.close()
        
        return list(contexts), list(sources)
        
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return [], []
        
def get_content_by_id(content_type: str, content_id: int) -> Dict[str, Any]:
    """
    Retrieve a specific content item by its ID and type.
    
    Args:
        content_type: The type of content ('comment', 'rating', 'instructor', 'course')
        content_id: The ID of the content
        
    Returns:
        A dictionary with the content details
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Determine the source table based on content type
        if content_type == 'comment':
            source_table = "trace.comments"
            query = f"""
            SELECT c.*, co.course_id as course_code, co.course_name, co.semester, co.year,
                   string_agg(i.name, ', ') as instructor_names
            FROM {source_table} c
            JOIN trace.courses co ON c.course_id = co.id
            LEFT JOIN trace.course_instructors ci ON co.id = ci.course_id
            LEFT JOIN trace.instructors i ON ci.instructor_id = i.id
            WHERE c.id = %s
            GROUP BY c.id, co.id
            """
        elif content_type == 'rating':
            source_table = "trace.ratings"
            query = f"""
            SELECT r.*, co.course_id as course_code, co.course_name, co.semester, co.year,
                   string_agg(i.name, ', ') as instructor_names
            FROM {source_table} r
            JOIN trace.courses co ON r.course_id = co.id
            LEFT JOIN trace.course_instructors ci ON co.id = ci.course_id
            LEFT JOIN trace.instructors i ON ci.instructor_id = i.id
            WHERE r.id = %s
            GROUP BY r.id, co.id
            """
        elif content_type == 'instructor':
            source_table = "trace.instructors"
            query = f"""
            SELECT i.*,
                   array_agg(DISTINCT co.course_id) as course_codes,
                   array_agg(DISTINCT co.course_name) as course_names,
                   array_agg(DISTINCT co.semester) as semesters,
                   array_agg(DISTINCT co.year) as years
            FROM {source_table} i
            LEFT JOIN trace.course_instructors ci ON i.id = ci.instructor_id
            LEFT JOIN trace.courses co ON ci.course_id = co.id
            WHERE i.id = %s
            GROUP BY i.id
            """
        elif content_type == 'course':
            source_table = "trace.courses"
            query = f"""
            SELECT co.*,
                   string_agg(DISTINCT i.name, ', ') as instructor_names
            FROM {source_table} co
            LEFT JOIN trace.course_instructors ci ON co.id = ci.course_id
            LEFT JOIN trace.instructors i ON ci.instructor_id = i.id
            WHERE co.id = %s
            GROUP BY co.id
            """
        else:
            return {}
        
        cursor.execute(query, (content_id,))
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if result:
            # Convert to dict and ensure datetime objects are serialized
            content_dict = dict(result)
            for k, v in content_dict.items():
                if hasattr(v, 'isoformat'):
                    content_dict[k] = v.isoformat()
            return content_dict
        else:
            return {}
            
    except Exception as e:
        logger.error(f"Error retrieving content: {e}")
        return {}