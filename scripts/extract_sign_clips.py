import cv2
import pandas as pd
import os

# Paths
CSV_FILE = 'asllvd_signs_2024_06_27.csv'
VIDEOS_DIR = 'data'
OUTPUT_DIR = 'processed_videos'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load metadata
df = pd.read_csv(CSV_FILE)

# Normalize filenames: strip whitespace and ensure consistency
df['full video file'] = df['full video file'].astype(str).str.strip()

# Get all available video filenames in the folder, stripped and lowercased
available_videos = {f.strip() for f in os.listdir(VIDEOS_DIR)}

# Filter the DataFrame to include only rows with available videos
df = df[df['full video file'].isin(available_videos)]

print(f"✅ Found {len(df)} matching entries with existing video files.\n")

# Process each valid video
for i, row in df.iterrows():
    sign_name = str(row['main entry gloss label']).strip()
    video_file = str(row['full video file']).strip()
    start = int(row['start frame of video clip containing the sign (relative to full videos)'])
    end = int(row['end frame of video clip containing the sign (relative to full videos)'])

    input_path = os.path.join(VIDEOS_DIR, video_file)
    output_filename = f"{sign_name}_{i}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Couldn't open: {input_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if start <= current_frame <= end:
            out.write(frame)
        elif current_frame > end:
            break
        current_frame += 1

    cap.release()
    out.release()
    print(f"✅ Saved: {output_filename}")
