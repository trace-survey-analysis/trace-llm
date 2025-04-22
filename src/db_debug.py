"""
Debug database connection and access to embeddings tables.
"""
import logging
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Database configuration
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '5432'),
    'database': os.environ.get('DB_NAME', 'postgres'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', 'postgres'),
}

def check_connection():
    """Test database connection and print configuration."""
    logger.info("Testing database connection with the following settings:")
    # Print sanitized config (hide password)
    safe_config = dict(DB_CONFIG)
    safe_config['password'] = '******' if safe_config['password'] else None
    for key, value in safe_config.items():
        logger.info(f"  {key}: {value}")
    
    try:
        # Connect to the database
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            dbname=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        logger.info("✅ Successfully connected to the database")
        
        # Check database version
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        logger.info(f"Database version: {version}")
        
        # Check current database
        cursor.execute("SELECT current_database();")
        database = cursor.fetchone()[0]
        logger.info(f"Current database: {database}")
        
        # Check current user
        cursor.execute("SELECT current_user;")
        user = cursor.fetchone()[0]
        logger.info(f"Connected as user: {user}")
        
        # Check search path
        cursor.execute("SHOW search_path;")
        search_path = cursor.fetchone()[0]
        logger.info(f"Search path: {search_path}")
        
        # Check if vectors schema exists
        cursor.execute("SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = 'vectors');")
        schema_exists = cursor.fetchone()[0]
        if schema_exists:
            logger.info("✅ Vectors schema exists")
        else:
            logger.error("❌ Vectors schema does not exist")
        
        # Explicitly set search path to include vectors
        cursor.execute("SET search_path TO public, vectors, trace;")
        logger.info("Updated search path to include vectors and trace schemas")
        
        # Check if embedding tables exist and count records
        tables_to_check = [
            "vectors.comment_embeddings",
            "vectors.rating_embeddings",
            "trace.comments",
            "trace.ratings"
        ]
        
        for table in tables_to_check:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table};")
                count = cursor.fetchone()[0]
                logger.info(f"✅ Table {table} exists with {count} records")
            except Exception as e:
                logger.error(f"❌ Error accessing table {table}: {e}")
        
        # Close connections
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Database connection error: {e}")

def main():
    """Main function to check database connection."""
    logger.info("Starting database connection check")
    check_connection()
    logger.info("Connection check completed")

if __name__ == "__main__":
    main()