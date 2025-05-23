import cv2
import json
import torch
import numpy as np
import time
import sys
import os
from pathlib import Path
import logging

# Add path to RetinaFace model
sys.path.append(os.path.abspath('/disk5/chime/mm/local/retinaface_pytorch'))
from retinaface_pytorch import retinaface_new as retinaface_model
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp

# Configure logging
def setup_logging(log_file_path):
    """
    Set up logging configuration
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='a', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_lip_coords_mediapipe(face_landmarks, bbox):
    """
    Extract lip coordinates from MediaPipe's 468 facial landmarks
    :param face_landmarks: MediaPipe facial landmarks
    :param bbox: Face bounding box [x1, y1, x2, y2]
    :return: Lip bounding box [x_min, y_min, x_max, y_max]
    """
    lip_indices = [
        61, 185, 40, 39, 37, 0,
        267, 269, 270, 409, 291,
        146, 91, 181, 84, 17, 314,
        405, 321, 375, 291, 
        57, 430, 164,
        287, 200, 210
    ]

    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    x_coords = []
    y_coords = []
    
    for idx in lip_indices:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * width + x1)
        y = int(landmark.y * height + y1)
        x_coords.append(x)
        y_coords.append(y)

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return [x_min, y_min, x_max, y_max]

def overlap(bbox1, bbox2):
    """
    Calculate the IoU (Intersection over Union) of two bounding boxes
    :return: IoU value
    """
    x1, y1, x2, y2 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    xi1 = max(x1, x1_2)
    yi1 = max(y1, y1_2)
    xi2 = min(x2, x2_2)
    yi2 = min(y2, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    if area1 + area2 - inter_area == 0:
        return 0.0  # Return 0 if no valid intersection
    iou = inter_area / float(area1 + area2 - inter_area) if (area1 + area2 - inter_area) > 0 else 0
    return iou

def is_face_in_frame(track_bbox, detected_faces):
    """
    Check if a face bounding box is present in the current frame's detection results
    :return: Whether a matching face bounding box is found
    """
    for face in detected_faces:
        # Check if there is an overlapping area, threshold adjustable
        if overlap(track_bbox, face) >= 0.8:
            return True
    return False

def process_video(input_video_path, output_json_path, retinaface, face_mesh):
    """
    Process a single video file and save results to JSON
    """
    if os.path.exists(output_json_path):
        logging.info(f"JSON file already exists, skipping: {output_json_path}")
        return
    # Initialize DeepSort
    deepsort = DeepSort(
        max_iou_distance=0.8,
        max_cosine_distance=0.4,
        max_age=1000,
        embedder='torchreid',
        n_init=1,
        embedder_gpu=True,
    )

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        logging.error(f"Unable to open video file: {input_video_path}")
        return

    frame_number = 0
    output_data = {}
    start_time = time.time()

    # Skip if more than 18 people are detected in 30 consecutive frames
    over_count = 0
    consecutive_threshold = 30  # 30 consecutive frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using RetinaFace
        faces = retinaface.detect_image(rgb_frame)

        if not any(isinstance(face, dict) for face in faces):
            output_data[str(frame_number)] = []
            frame_number += 1
            continue

        # Check the number of people detected in the current frame
        if len(faces) >= 18:
            over_count += 1
        else:
            over_count = 0  # Reset counter if fewer than 18 people are detected

        # Skip video if more than 18 people are detected in 30 consecutive frames
        if over_count >= consecutive_threshold:
            logging.info(f"More than 18 people detected in 30 consecutive frames, skipping video: {input_video_path}")
            cap.release()
            return

        bboxes = []
        confidences = []
        for face in faces:
            facial_area = face['box']
            score = face['confidence']
            bboxes.append(facial_area)
            confidences.append(score)

        # Prepare detection results for DeepSort [x1, y1, x2, y2, confidence]
        detections_deepsort = []
        for bbox, conf in zip(bboxes, confidences):
            x1, y1, x2, y2 = bbox
            width = x2 - x1  # Calculate width
            height = y2 - y1  # Calculate height
            class_id = 0  # Uniformly set as face class
            detection = ([x1, y1, width, height], conf, class_id)
            detections_deepsort.append(detection)

        # Update tracker
        tracks = deepsort.update_tracks(detections_deepsort, frame=frame)

        frame_data = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()

            x1, y1, x2, y2 = map(int, ltrb)

            # Skip track if its face bounding box is not found in detected faces
            if not is_face_in_frame([x1, y1, x2, y2], bboxes):
                continue

            # Extract face ROI for landmark detection
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                logging.warning(f"Empty ROI, Track ID: {track_id}, Video: {input_video_path}, Frame: {frame_number}")
                continue  # Skip if ROI is empty
            try:
                face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(face_roi_rgb)
                if not results.multi_face_landmarks:
                    # No landmarks detected, record face coordinates, lip coordinates as empty list
                    lip = []
                else:
                    # Assume each track corresponds to one face, take the first detected landmarks
                    face_landmarks = results.multi_face_landmarks[0]

                    # Extract lip coordinates
                    lip_coords = extract_lip_coords_mediapipe(face_landmarks, [x1, y1, x2, y2])
                    if lip_coords is not None:
                        # Lip coordinates in [x1, y1, x2, y2] format
                        lip = [lip_coords[0], lip_coords[1], lip_coords[2], lip_coords[3]]
                    else:
                        lip = []

                face_dict = {
                    "id": track_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "lip": lip
                }
                frame_data.append(face_dict)
            except Exception as e:
                logging.error(f"Landmark detection failed, Track ID: {track_id}, Error: {e}")

        # Add current frame data to output dictionary
        output_data[str(frame_number)] = frame_data
        frame_number += 1

    # Release video capture
    cap.release()

    # Save output data to JSON file
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.error(f"Failed to save JSON: {output_json_path}, Error: {e}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Completed processing video: {input_video_path}, Time taken: {elapsed_time:.2f} seconds")

def main():
    # Define main input and output directories
    root_input_dir = "/disk5/chime/mm/data/meeting/segment/"  # Replace with your main input directory path
    root_output_dir = "/disk5/chime/mm/data/meeting/roi_result/"  # Replace with your main output directory path

    # Define log file path
    log_file_path = "/disk5/chime/mm/data/meeting/meeting_process_0-500.log"  # Replace with your desired log file path
    setup_logging(log_file_path)
    logging.info("Program started")

    logging.getLogger('deep_sort_realtime.deepsort_tracker').setLevel(logging.WARNING)
    logging.getLogger('deep_sort_realtime').setLevel(logging.WARNING)
    logging.getLogger('torchreid').setLevel(logging.WARNING)
    logging.getLogger('retinaface_pytorch').setLevel(logging.ERROR)
    logging.getLogger('mediapipe').setLevel(logging.ERROR)

    # Instantiate RetinaFace model (assuming this step is time-consuming, instantiate once)
    try:
        retinaface = retinaface_model.Retinaface()  # Instantiate
        logging.info("RetinaFace model instantiated successfully")
    except Exception as e:
        logging.error(f"Failed to instantiate RetinaFace model: {e}")
        sys.exit(1)

    # Iterate through all subfolders (A) in the main input directory
    for a_dir in os.listdir(root_input_dir):
        a_path = os.path.join(root_input_dir, a_dir)
        if not os.path.isdir(a_path):
            continue  # Skip non-directory files

        if not (0 < int(a_dir) <= 500):  # Calculation in groups
            continue

        logging.info(f"Starting to process folder: {a_dir}")

        video_dir = os.path.join(a_path, "video")
        if not os.path.isdir(video_dir):
            continue

        # Create corresponding output directory
        output_a_dir = os.path.join(root_output_dir, a_dir)
        os.makedirs(output_a_dir, exist_ok=True)

        # Iterate through all mp4 files in the video directory
        for video_file in os.listdir(video_dir):
            if not video_file.lower().endswith('.mp4'):
                logging.warning(f"Skipping non-mp4 file: {video_file}")
                continue  # Skip non-mp4 files

            input_video_path = os.path.join(video_dir, video_file)
            video_name = Path(video_file).stem  # Get filename (without extension)
            output_json_path = os.path.join(output_a_dir, f"{video_name}.json")

            logging.info(f"Starting to process video: {input_video_path}")
            process_video(input_video_path, output_json_path, retinaface, face_mesh)

        logging.info(f"Folder processing completed: {a_dir}")

    logging.info("All video processing completed.")

if __name__ == "__main__":
    main()