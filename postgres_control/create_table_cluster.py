import psycopg2
# DB config
DB_NAME = "keyframes_db"
DB_USER = "quannh"
DB_PASSWORD = "1"
DB_HOST = "192.168.20.150"
DB_PORT = 5432

conn = psycopg2.connect(
    dbname=DB_NAME,
    user="quannh",
    password="1",
    host=DB_HOST,
    port=5432
)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS cluster CASCADE")
cur.execute("""
CREATE TABLE IF NOT EXISTS cluster (
    id SERIAL PRIMARY KEY,
    video_id TEXT NOT NULL,
    frame_id INT NOT NULL,
    label INT NOT NULL,
    UNIQUE(video_id, frame_id)
)
""")

cur.execute("CREATE INDEX IF NOT EXISTS idx_video_id ON cluster(video_id)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_video_frame_id ON cluster(video_id, frame_id)")

conn.commit()

# cur.execute("SELECT COUNT(*) FROM cluster")
# record_count = cur.fetchone()[0]
# print(f"Number of records in 'cluster': {record_count}")

cur.close()
conn.close()
