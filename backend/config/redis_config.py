import os
import redis
from rq import Queue
from dotenv import load_dotenv

load_dotenv()

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)

# Create Redis connection
redis_conn = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    password=REDIS_PASSWORD,
    decode_responses=True
)

# Create queues for different task types
classification_queue = Queue('classification', connection=redis_conn, default_timeout=300)
csv_processing_queue = Queue('csv_processing', connection=redis_conn, default_timeout=1800)
batch_processing_queue = Queue('batch_processing', connection=redis_conn, default_timeout=600)

# Queue names
QUEUE_NAMES = {
    'CLASSIFICATION': 'classification',
    'CSV_PROCESSING': 'csv_processing', 
    'BATCH_PROCESSING': 'batch_processing'
} 