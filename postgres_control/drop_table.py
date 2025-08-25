import psycopg2

# DB config
DB_NAME = "keyframes_db"
DB_USER = "quannh"
DB_PASSWORD = "1"
DB_HOST = "192.168.20.156"
DB_PORT = 5432

conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cur = conn.cursor()

# Drop the table
cur.execute("DROP TABLE IF EXISTS keyframes CASCADE")
cur.execute("DROP TABLE IF EXISTS cluster CASCADE")

conn.commit()
cur.close()
conn.close()
