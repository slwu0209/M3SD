import os
import csv

def check_offsets(csv_path):
    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Read header
        # Find the column indices for AV_offset and Confidence
        try:
            av_offset_idx = headers.index('AV_offset')
            confidence_idx = headers.index('Confidence')
        except ValueError:
            print(f"File {csv_path} is missing required columns.")
            return False
        
        for row in reader:
            try:
                av_offset = float(row[av_offset_idx])
                confidence = float(row[confidence_idx])
                if abs(av_offset) <= 5 and confidence >= 1:
                    return True
            except (ValueError, IndexError):
                continue
    return False
    
def get_audio_video_paths(txt_file, base_dir):
    audio_video_paths = set()  # Use a set to remove duplicates
    with open(txt_file, 'r') as f:
        for line in f:
            a, b = line.strip().split()  # Each line is 'A B'
            audio_folder = os.path.join(base_dir, a, 'audio')  # Construct audio folder path
            video_folder = os.path.join(base_dir, a, 'video')  # Construct video folder path
            audio_file = os.path.join(audio_folder, b + '.wav')  # Construct audio file path
            video_file = os.path.join(video_folder, b + '.mp4')  # Construct video file path
            audio_video_paths.add(audio_file)
            audio_video_paths.add(video_file)
    return audio_video_paths

# Delete audio and video files not listed in the txt file
def delete_unlisted_files(txt_file, base_dir):
    audio_video_paths = get_audio_video_paths(txt_file, base_dir)
    
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            audio_folder = os.path.join(folder_path, 'audio')
            video_folder = os.path.join(folder_path, 'video')
            
            # Delete audio files not listed in the txt file
            if os.path.exists(audio_folder):
                for audio_file in os.listdir(audio_folder):
                    audio_file_path = os.path.join(audio_folder, audio_file)
                    if audio_file_path not in audio_video_paths and audio_file.endswith('.wav'):
                        print(f"Deleting audio file: {audio_file_path}")
                        os.remove(audio_file_path)

            # Delete video files not listed in the txt file
            if os.path.exists(video_folder):
                for video_file in os.listdir(video_folder):
                    video_file_path = os.path.join(video_folder, video_file)
                    if video_file_path not in audio_video_paths and video_file.endswith('.mp4'):
                        print(f"Deleting video file: {video_file_path}")
                        os.remove(video_file_path)

def main():
    # Set main folder path
    main_folder = '/disk5/chime/mm/syncnet_python-master/output/meeting'  # Replace with your main folder path
    base_dir = '/disk5/chime/mm/data/meeting/segment'  # Scene detection and segmentation results

    result = []
    for A in os.listdir(main_folder):
        A_path = os.path.join(main_folder, A)
        if os.path.isdir(A_path):
            pywork_path = os.path.join(A_path, 'pywork')
            if os.path.isdir(pywork_path):
                for B in os.listdir(pywork_path):
                    B_path = os.path.join(pywork_path, B)
                    if os.path.isdir(B_path):
                        csv_file = os.path.join(B_path, 'offsets.csv')
                        if os.path.isfile(csv_file):
                            if check_offsets(csv_file):
                                result.append(f"{A}\t{B}\n")
                        else:
                            print(f"File {csv_file} does not exist.")
    
    # Write results to result.txt
    output_file = os.path.join(main_folder, '/disk5/chime/mm/syncnet_python-master/output/meeting_result.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(result)
    
    print(f"Results written to {output_file}")
    
    delete_unlisted_files(output_file, base_dir)
    for foldername, subfolders, filenames in os.walk(base_dir, topdown=False):  # Traverse from leaf nodes
        # Delete empty directories with no subfolders or files
        if not subfolders and not filenames:
            print(f"Deleting empty folder: {foldername}")
            os.rmdir(foldername)

if __name__ == "__main__":
    main()