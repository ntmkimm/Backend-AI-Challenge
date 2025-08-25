import psycopg2
# DB config
DB_NAME = "keyframes_db"
DB_USER = "quannh"
DB_PASSWORD = "1"
DB_HOST = "192.168.20.156"
DB_PORT = 5432

conn = psycopg2.connect(
    dbname=DB_NAME,
    user="quannh",
    password="1",
    host=DB_HOST,
    port=5432
)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS keyframes (
    id SERIAL PRIMARY KEY,
    video_id TEXT NOT NULL,
    frame_id INT NOT NULL,
    frame_name TEXT NOT NULL,
    frame_path TEXT NOT NULL,
    UNIQUE(video_id, frame_id)
)
""")

cur.execute("CREATE INDEX IF NOT EXISTS idx_video_id ON keyframes(video_id)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_video_frame_id ON keyframes(video_id, frame_id)")

conn.commit()
cur.close()
conn.close()
