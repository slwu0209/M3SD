import os
import pandas as pd
from pydub import AudioSegment
from scenedetect import SceneManager, open_video, split_video_ffmpeg
from scenedetect.detectors import ContentDetector
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import logging
import shutil
import psutil  

video_folder = "/disk5/chime/mm/data/meeting/video"
audio_folder = "/disk5/chime/mm/data/meeting/audio"
output_folder = "/disk5/chime/mm/data/meeting/segment"
timestamp_data = []
MIN_SCENE_DURATION = 15  # Minimum scene duration (seconds)

log_file = '/disk5/chime/mm/data/meeting/video_processing.log'

# Delete existing log file before configuring logging
if os.path.exists(log_file):
    os.remove(log_file)
# Configure logging
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Get list of video files
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov', '.webm'))]

# Split audio
def split_audio(audio_path, scene_list, video_name, base_output_folder):
    audio = AudioSegment.from_wav(audio_path)
    local_timestamps = []

    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()

        # Split audio
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        output_audio_path = os.path.join(base_output_folder, f"{video_name}-Scene-{i+1:03d}.wav")

        audio_segment = audio[start_ms:end_ms]
        audio_segment.export(output_audio_path, format="wav")

        # Record timestamps
        local_timestamps.append([video_name, i+1, start_time, end_time])
    
    return local_timestamps

# Process video: detect scenes and split video and audio
def process_video(video_file):
    try:
        video_path = os.path.join(video_folder, video_file)
        base_name = os.path.splitext(video_file)[0]

        # Create output folders
        output = os.path.join(output_folder, base_name)
        output_video_folder = os.path.join(output_folder, base_name, "video")
        output_audio_folder = os.path.join(output_folder, base_name, "audio")
        os.makedirs(output_video_folder, exist_ok=True)
        os.makedirs(output_audio_folder, exist_ok=True)

        # Use SceneDetect for scene detection
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=30))
        video = open_video(video_path)
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        # Filter out scenes shorter than the minimum duration
        filtered_scene_list = []
        print(base_name)
        for scene in scene_list:
            scene_duration = scene[1].get_seconds() - scene[0].get_seconds()
            if scene_duration >= MIN_SCENE_DURATION:
                filtered_scene_list.append(scene)
        # If no scenes meet the criteria, skip the video
        if not filtered_scene_list:
            logging.info(f"Video '{video_file}' has no qualifying scenes, skipping.")
            shutil.rmtree(output)  # Delete the entire output folder
            return []

        # Split video
        split_video_ffmpeg(video_path, filtered_scene_list, output_video_folder)

        # Audio splitting
        audio_file = base_name + ".wav"  # Assume audio file has the same name as video
        audio_path = os.path.join(audio_folder, audio_file)
        if not os.path.exists(audio_path):
            logging.warning(f"Audio file '{audio_file}' does not exist, skipping audio splitting.")
            return []

        # Split audio
        return split_audio(audio_path, filtered_scene_list, base_name, output_audio_folder)
    
    except Exception as e:
        logging.error(f"Failed to process video '{video_file}': {e}")
        return []

# Parallel processing with multiple processes
def process_videos_in_parallel():
    global timestamp_data
    cpu_count = psutil.cpu_count(logical=False)  # Number of physical cores
    max_workers = cpu_count  # Set to number of physical cores

    logging.info(f"Using {max_workers} processes for video processing.")
    print(f"Using {max_workers} processes for video processing.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_video, video_file): video_file for video_file in video_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Videos"):
            video_file = futures[future]
            try:
                result = future.result()
                if result:
                    timestamp_data.extend(result)
            except Exception as e:
                logging.error(f"Failed to process video '{video_file}': {e}")

# Main function
def main():
    print("Starting video and audio processing...")
    logging.info("Starting video and audio processing...")

    # Process videos in parallel
    process_videos_in_parallel()

    if timestamp_data:
        # Create Excel file to store timestamps
        output_excel_path = os.path.join(output_folder, "timestamps.xlsx")
        df = pd.DataFrame(timestamp_data, columns=["Video Name", "Part", "Start Time", "End Time"])
        df.to_excel(output_excel_path, index=False)
        print(f"All video and audio processing completed, timestamps saved to {output_excel_path}")
        logging.info(f"All video and audio processing completed, timestamps saved to {output_excel_path}")
    else:
        print("No qualifying scenes were processed.")
        logging.info("No qualifying scenes were processed.")

if __name__ == "__main__":
    main()