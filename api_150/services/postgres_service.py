

from psycopg2 import pool
import os
from dotenv import load_dotenv
import threading
from typing import Dict, List

load_dotenv()

_connection_pool = None
_pool_lock = threading.Lock()   

def get_pool():
    global _connection_pool
    if _connection_pool is None:
        with _pool_lock:
            if _connection_pool is None:  # double-checked locking
                _connection_pool = pool.SimpleConnectionPool(
                    1, 50,   # minconn, maxconn 
                    dbname=os.getenv("DB_NAME"),
                    user=os.getenv("DB_USER"),
                    password=os.getenv("DB_PASSWORD"),
                    host=os.getenv("DB_HOST"),
                    port=os.getenv("DB_PORT")
                )
    return _connection_pool

def get_connection():
    return get_pool().getconn()

def release_connection(conn):
    get_pool().putconn(conn)
    

    

