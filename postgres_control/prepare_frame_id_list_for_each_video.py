import os
import psycopg2
import re

# DB config
DB_NAME = "keyframes_db"
DB_USER = "quannh"
DB_PASSWORD = "1"
DB_HOST = "192.168.20.156"
DB_PORT = 5432

# Dataset path
BASE_PATH = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/batch1"

def extract_frame_index(filename):
    match = re.search(r'keyframe_(\d+)\.webp', filename)
    return int(match.group(1)) if match else None

def insert_keyframes():
    inserted = 0
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        with conn:
            with conn.cursor() as cur:
                for video_folder in os.listdir(BASE_PATH):
                    video_path = os.path.join(BASE_PATH, video_folder, "keyframes")
                    if not os.path.isdir(video_path):
                        continue

                    for fname in os.listdir(video_path):
                        if not fname.endswith(".webp"):
                            continue

                        frame_index = extract_frame_index(fname)
                        if frame_index is None:
                            continue

                        file_path = os.path.join(video_path, fname)
                        relative_path = os.path.relpath(file_path, BASE_PATH)

                        try:
                            cur.execute("""
                                INSERT INTO keyframes (video_id, frame_id, frame_name, frame_path)
                                VALUES (%s, %s, %s, %s)
                                ON CONFLICT (video_id, frame_id) DO NOTHING;
                            """, (video_folder, frame_index, fname, relative_path))
                            inserted += 1
                        except Exception as e:
                            print(f"Error inserting {fname}: {e}")
    finally:
        conn.close()

    print(f"✅ Done inserting keyframes. Total inserted: {inserted}")

if __name__ == "__main__":
    insert_keyframes()
