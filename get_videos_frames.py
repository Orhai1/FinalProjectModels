import os
import boto3
import pandas as pd
from extract_frames.extract_frames_and_filter import extract_and_filter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
import os

load_dotenv()

# access environment variables
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION")

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_region
)

# Constants
BUCKET_NAME = "tiktok-project-storage"
S3_FOLDER = "videos/downloads/"
OUTPUT_DIR = "model_videos_frames/"
CSV_PATH = "model_videos_list.csv"

# Read video IDs
df = pd.read_csv(CSV_PATH, header=None)
video_ids = df.iloc[:, 0].tolist()

# AWS client setup (will be recreated in each process)
def get_s3_client():
    return boto3.client("s3")

# Process a single video
def process_video(video_id):
    s3_client = get_s3_client()
    key = f"{S3_FOLDER}{video_id}.mp4"
    local_path = os.path.join(OUTPUT_DIR, f"{video_id}.mp4")

    try:
        print(f"[{video_id}] Downloading from S3.")
        s3_client.download_file(BUCKET_NAME, key, local_path)

        print(f"[{video_id}] Extracting frames.")
        extract_and_filter(local_path)

    except Exception as e:
        print(f"[{video_id}] Error: {e}")
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)
            print(f"[{video_id}] Cleaned up downloaded file.")

    return video_id

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MAX_WORKERS = 4
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_video, vid) for vid in video_ids]
        for future in as_completed(futures):
            try:
                video_id = future.result()
                print(f"[{video_id}] Done.")
            except Exception as e:
                print(f"Error during processing: {e}")

    print("All videos processed.")
