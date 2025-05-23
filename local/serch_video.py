# -*- coding: utf-8 -*-
import yt_dlp
import os
import pandas as pd
from googleapiclient.discovery import build
from moviepy import VideoFileClip
from datetime import timedelta
import re
from tqdm import tqdm 
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import shutil
import subprocess


API_KEY = ' '  # Your YouTube API key
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'
SEARCH_KEYWORD = 'meeting'  # key word
MAX_RESULTS = 50 
VIDEO_OUTPUT_PATH = '/disk5/chime/mm/data/meeting/video'  # video output
AUDIO_OUTPUT_PATH = '/disk5/chime/mm/data/meeting/audio'    # audio output
EXCEL_FILE = '/disk5/chime/mm/data/meeting/videos_info.xlsx'  # metadata information
BACKUP_EXCEL_FILE = '/disk5/chime/mm/data/multi/videos_info_backup.xlsx' 
DURATION_THRESHOLD = 180  
SORT_ORDER = 'relevance'  # 'relevance', 'date', 'viewCount', 'rating'
PAGE_TOKEN_FILE = '/disk5/chime/mm/data/multi/last_page_token.txt'  # pageToken
MAX_WORKERS = 5   
NUMBERING_LENGTH = 5  # 00001


youtube = build(API_SERVICE_NAME, API_VERSION, developerKey=API_KEY)

download_lock = threading.Lock()
failure_lock = threading.Lock()
numbering_lock = threading.Lock()

# Global Number Counter
numbering_counter = 1

def backup_excel(excel_file, backup_file):

    if os.path.exists(excel_file):
        try:
            shutil.copy(excel_file, backup_file)
            print(f"The Excel file has been backed up to {backup_file}")
        except Exception as e:
            print(f"Backup Excel file failed: {e}")
            exit(1)
    else:
        print(f"Excel file does not exist: {excel_file}")
        exit(1)

def load_excel(excel_file):

    try:
        df = pd.read_excel(excel_file, engine='openpyxl')
        print(f"Successfully loaded Excel file: {excel_file}")
        return df
    except Exception as e:
        print(f"Failed to read Excel file: {e}")
        exit(1)

def initialize_numbering(df):
    """
    Initialize the number counter based on the maximum number in the existing Excel file.
    If there is no 'numbering' column, start from 1。
    """
    global numbering_counter
    if 'numbering' in df.columns:
        existing_numbers = df['numbering'].dropna().astype(int).tolist()
        if existing_numbers:
            numbering_counter = max(existing_numbers) + 1
        else:
            numbering_counter = 1
    else:
        numbering_counter = 1
    print(f"The initial number counter is set to: {numbering_counter}")

def get_next_number():
    """
    Get the next number, thread safe.
    """
    global numbering_counter
    with numbering_lock:
        current_number = numbering_counter
        numbering_counter += 1
    return f"{current_number:0{NUMBERING_LENGTH}d}"  # '00001'

def find_video_file(video_dir, video_id, extensions=['.mp4', '.mkv', '.webm']):
    """
    Search for files in the specified directory that contain video_id and have the specified extension.
    Return the complete path and extension of the file.
    """
    for ext in extensions:
        pattern = re.compile(rf'.*{re.escape(video_id)}.*{re.escape(ext)}$', re.IGNORECASE)
        for file in os.listdir(video_dir):
            if pattern.match(file):
                return os.path.join(video_dir, file), ext
    return None, None

def find_audio_file(audio_dir, video_id, extensions=['.wav']):
    """
    Search for files in the specified directory that contain video_id and have the specified extension.
    Return the complete path and extension of the file.
    """
    for ext in extensions:
        pattern = re.compile(rf'.*{re.escape(video_id)}.*{re.escape(ext)}$', re.IGNORECASE)
        for file in os.listdir(audio_dir):
            if pattern.match(file):
                return os.path.join(audio_dir, file), ext
    return None, None

def search_videos(keyword, max_results=50, order='relevance', page_token=None):
    """
    Search for videos on YouTube based on keywords, return a list containing video information and the next pageToken
    """
    request_params = {
        'q': keyword,
        'part': 'id,snippet',
        'maxResults': max_results,
        'type': 'video',
        'order': order  
    }
    if page_token:
        request_params['pageToken'] = page_token

    request = youtube.search().list(**request_params)
    response = request.execute()
    videos = []
    for item in response.get('items', []):
        video = {
            'videoId': item['id']['videoId'],
            'title': item['snippet']['title'],
            'description': item['snippet']['description'],
            'channelTitle': item['snippet']['channelTitle'],
            'publishedAt': item['snippet']['publishedAt']
        }
        videos.append(video)
    next_page_token = response.get('nextPageToken')
    return videos, next_page_token

def get_video_durations(video_ids):
    """
    Retrieve video duration based on video ID list
    """
    durations = {}
    # YouTube Data API, process up to 50 video IDs at a time
    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i:i+50]
        request = youtube.videos().list(
            part='contentDetails',
            id=','.join(batch_ids)
        )
        response = request.execute()
        for item in response.get('items', []):
            video_id = item['id']
            duration = item['contentDetails']['duration']
            durations[video_id] = parse_duration(duration)
    return durations

def parse_duration(duration):

    regex = re.compile(
        r'P(?:(?P<years>\d+)Y)?(?:(?P<months>\d+)M)?(?:(?P<weeks>\d+)W)?'
        r'(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?'
        r'(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?'
    )
    match = regex.match(duration)
    if not match:
        return 0
    parts = {name: int(value) if value else 0 for name, value in match.groupdict().items()}
    td = timedelta(
        days=parts.get('days', 0) + parts.get('weeks', 0) * 7,
        hours=parts.get('hours', 0),
        minutes=parts.get('minutes', 0),
        seconds=parts.get('seconds', 0)
    )
    total_seconds = int(td.total_seconds())
    return total_seconds

def load_existing_video_ids(excel_file):
    """
    Load an existing Excel file and retrieve the videoId list of downloaded videos
    """
    if os.path.exists(excel_file):
        existing_df = pd.read_excel(excel_file, engine='openpyxl')
        existing_video_ids = existing_df['videoId'].tolist()
        print(f"The number of existing videoIDs: {len(existing_video_ids)}")
        return existing_video_ids
    else:
        print("The Excel file does not exist, start creating a new one.")
        return []

def save_videos_to_excel(videos, durations, excel_file):
    """
    Save video information and duration to an Excel spreadsheet, and filter videos with a duration exceeding 3 minutes
    """
    for video in videos:
        video_id = video['videoId']
        duration_seconds = durations.get(video_id, 0)
        video['duration'] = str(timedelta(seconds=duration_seconds))
        video['duration_seconds'] = duration_seconds  

    filtered_videos = [video for video in videos if video['duration_seconds'] > DURATION_THRESHOLD]

    print(f"After filtering, a total of {len (filtered-videos)} videos with a duration exceeding 3 minutes were found.")

    if not filtered_videos:
        return 

    for video in filtered_videos:
        del video['duration_seconds']

    if os.path.exists(excel_file):
        existing_df = pd.read_excel(excel_file, engine='openpyxl')
        new_df = pd.DataFrame(filtered_videos)

        final_df = pd.concat([existing_df, new_df], ignore_index=True)

        final_df.drop_duplicates(subset=['videoId'], inplace=True)
    else:

        final_df = pd.DataFrame(filtered_videos)


    final_df.to_excel(excel_file, index=False)
    print(f"The video information has been saved to {excel_file}")

def convert_webm_to_mp4(input_path, output_path, ffmpeg_path='ffmpeg'):
    command = [
        ffmpeg_path,
        '-i', input_path,
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        output_path
    ]
    subprocess.run(command, check=True)

def download_video_yt_dlp(video_url, video_id, title, output_path=VIDEO_OUTPUT_PATH):
    """
    Use yt dlp to download videos from YouTube URL to the specified directory and return the downloaded file path
    """
    pattern = r'[\\/*?:"<>|]'
    safe_title = re.sub(pattern, '_', title)

    partial_file = os.path.join(output_path, f"{safe_title}-{video_id}.part")
    partial_file1 = os.path.join(output_path, f"{safe_title}-{video_id}.f4247.webm.part")

    cookies_path = '/disk5/chime/mm/www.youtube.com_cookies.txt'
    ffmpeg_path = '/usr/bin/ffmpeg'
    
    ydl_opts = {
        'outtmpl': os.path.join(output_path, f'{safe_title}-{video_id}.%(ext)s'),
        'format': 'bestvideo[height<=720][ext=webm]+bestaudio[ext=webm]',
        'quiet': False,  
        'no_warnings': True,
        'retries': 10, 
        'fragment_retries': 10, 
        'continuedl': True, 
        'ignoreerrors': False, 
        'cookies': cookies_path,
        'ffmpeg_location': ffmpeg_path,
        'merge_output_format': 'webm',
    }
    
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            filename = ydl.prepare_filename(info)
            # filename_mp4 = os.path.splitext(filename)[0] + '.mp4'
            # convert_webm_to_mp4(filename, filename_mp4) # 
            # print(filename_mp4)
        print(f"Video download completed: {video_url}")
        return filename  
    except Exception as e:
        print(f"Video download failed {video_url}: {e}")
        #   if os.path.exists(partial_file) or os.path.exists(partial_file1):
            try:
                if os.path.exists(partial_file):
                    os.remove(partial_file)
                    print(f"Removed residual downloaded files: {partial_file}")
                elif os.path.exists(partial_file1):
                    os.remove(partial_file1)
                    print(f"Removed residual downloaded files: {partial_file1}")
            except Exception as delete_error:
                print(f"Failed to delete some downloaded files {partial_file}: {delete_error}")
        return None

def extract_audio(video_path, output_path=AUDIO_OUTPUT_PATH):
    """
    Extract audio from video files and save it in WAV format
    """
    try:
        video = VideoFileClip(video_path)  
        if not os.path.exists(output_path):
            os.makedirs(output_path)
  
        audio_filename = re.sub(r'\.(mp4|mkv|webm)$', '.wav', os.path.basename(video_path))
        audio_path = os.path.join(output_path, audio_filename)

        video.audio.write_audiofile(audio_path, codec='pcm_s16le') 
        print(f"Audio extraction completed: {audio_path}")
    except Exception as e:
        print(f"Extracting audio from video failed {video_path}: {e}")
        raise 

def remove_videos_from_excel(failed_video_ids, df):
    """
    Delete download failed video information from Excel file
    """
    initial_count = len(df)
    df = df[~df['videoId'].isin(failed_video_ids)]
    final_count = len(df)

    if initial_count != final_count:
        try:
            #df.to_excel(excel_file, index=False)
            print(f"Deleted {initial_count - final_count}  failed video information from Excel.")
            return df

        except Exception as e:
            print(f"Writing to Excel file failed: {e}")
    else:
        print("The video information that needs to be deleted was not found.")

def load_last_page_token(file_path):
    """
    Load the last pageToken
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            token = f.read().strip()
            return token
    else:
        return None

def save_last_page_token(file_path, token):
    """
    Save the current pageToken
    """
    if token:
        with open(file_path, 'w') as f:
            f.write(token)
    else:

        if os.path.exists(file_path):
            os.remove(file_path)

def download_and_extract(video_info, downloaded_files, failed_video_ids, df):

    video_url = f"https://www.youtube.com/watch?v={video_info['videoId']}"
    video_id = video_info['videoId']
    title = video_info['title']
    filepath = download_video_yt_dlp(video_url, video_id, title)
    if filepath:
        extract_audio(filepath)
        with download_lock:
            downloaded_files.append(filepath)

        numbering_str = get_next_number()
        video_ext = os.path.splitext(filepath)[1]
        new_video_filename = f"{numbering_str}{video_ext}"
        new_video_path = os.path.join(VIDEO_OUTPUT_PATH, new_video_filename)
        try:
            os.rename(filepath, new_video_path)
            print(f"Video file renaming: {filepath} -> {new_video_path}")
            with download_lock:

                df.loc[df['videoId'] == video_id, 'numbering'] = numbering_str
        except Exception as e:
            print(f"Renaming video file failed {filepath}: {e}")
            with failure_lock:
                failed_video_ids.append(video_id)
            return

        base_name = os.path.splitext(os.path.basename(filepath))[0]
        audio_filename = f"{base_name}.wav"
        audio_path = os.path.join(AUDIO_OUTPUT_PATH, audio_filename)
        new_audio_filename = f"{numbering_str}.wav"
        new_audio_path = os.path.join(AUDIO_OUTPUT_PATH, new_audio_filename)
        try:
            os.rename(audio_path, new_audio_path)
            print(f"Audio file renaming: {audio_path} -> {new_audio_path}")
        except Exception as e:
            print(f"Renaming audio file failed {audio_path}: {e}")
            with failure_lock:
                failed_video_ids.append(video_id)
            return
    else:
        with failure_lock:
            failed_video_ids.append(video_id)

def main():
    # 1. Backup Excel files
    backup_excel(EXCEL_FILE, BACKUP_EXCEL_FILE)
    
    # 2. Load Excel file
    df = load_excel(EXCEL_FILE)
    
    # 3. Initialize the number counter
    initialize_numbering(df)    
    
    # 5. Load the existing videoId list
    existing_video_ids = load_existing_video_ids(EXCEL_FILE)

    # 6. Load the final pageToken
    last_page_token = load_last_page_token(PAGE_TOKEN_FILE)
    if last_page_token:
        print(f"Use the previous pageToken: {last_page_token}")
    else:
        print("Unable to find the previous pageToken, start a new search.")

    # 7. Search video
    print(f"Searching for videos with the keyword '{SEARCH_KEYWORLD}', sorted by:{SORT_ORDER}...")
    videos, next_page_token = search_videos(SEARCH_KEYWORD, max_results=MAX_RESULTS, order=SORT_ORDER, page_token=last_page_token)
    print(f"Found {len (videos)} videos in total.")

    if not videos:
        print("No relevant video found, program ends.")
        exit()

    # 8. Filter out existing videos
    new_videos = [video for video in videos if video['videoId'] not in existing_video_ids]
    print(f"Found {len (new-videos)} videos that have not been downloaded before.")

    if not new_videos:
        print("All found videos have been downloaded and the program has ended.")
        if next_page_token:
            save_last_page_token(PAGE_TOKEN_FILE, next_page_token)
        exit()

    # 9. Obtain video duration
    video_ids = [video['videoId'] for video in new_videos]
    durations = get_video_durations(video_ids)

    # 10. Save video information and duration to Excel, filter videos with a duration exceeding 3 minutes
    save_videos_to_excel(new_videos, durations, EXCEL_FILE)

    # 11. Reload Excel file to include newly added videos
    df = load_excel(EXCEL_FILE)

    if 'numbering' not in df.columns:
        df['numbering'] = None
    
    # 12. The filtered video list (completed in the save function)
    filtered_videos = [video for video in new_videos if durations.get(video['videoId'], 0) > DURATION_THRESHOLD]

    if not filtered_videos:
        print("After filtering, if there are no videos that meet the criteria, the program ends.")
        if next_page_token:
            save_last_page_token(PAGE_TOKEN_FILE, next_page_token)
        exit()

    # 13. Download videos and extract audio (using multithreading)
    failed_video_ids = []
    downloaded_files = [] 
    print("Start downloading videos and extracting audio ...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_video = {
            executor.submit(download_and_extract, video_info, downloaded_files, failed_video_ids, df): video_info
            for video_info in filtered_videos
        }


        for future in tqdm(as_completed(future_to_video), total=len(future_to_video), desc="process video"):
            video_info = future_to_video[future]
            try:
                future.result()
            except Exception as e:
                print(f"Processing video failed {video_info['videoId']}: {e}")
                with failure_lock:
                    failed_video_ids.append(video_info['videoId'])

    # 14. Delete video information that failed to download
    if failed_video_ids:
        print("Removing download failed video information from Excel...")
        df = remove_videos_from_excel(failed_video_ids, df)
    else:
        print("All videos have been successfully downloaded.")

    # 15. Save the updated Excel file
    df.to_excel(EXCEL_FILE, index=False)
    print(f"The updated Excel file has been saved to {EXCEL_FILE}")

    # 16. Save a new pageToken
    save_last_page_token(PAGE_TOKEN_FILE, next_page_token)
    if next_page_token:
        print(f"Save new pageToken: {next_page_token}")
    else:
        print("We have reached the end of the search results and the pageToken file has been deleted.")

    print("All operations completed.")

if __name__ == "__main__":
    main()
