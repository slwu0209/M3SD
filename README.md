# M3SD
M3SD: Multi-modal, Multi-scenario and Multi-language Speaker Diarization Dataset

# Quick start

- **Step 1: Get audio and video**

You can download our dataset (https://huggingface.co/spaces/OldDragon/m3sd) or crawl audio and video from YouTube according to the code (local/search_video.py)

You first need to get your own YouTube API key (https://console.developers.google.com/apis/api/youtube.googleapis.com) and fill it in the code.
```
python local/search_video.py  # (Please change your file path in the script)
```
- **Step 2: Scene detection and segmentation**

In order to ensure the effect of subsequent face detection and tracking, it is necessary to perform scene detection and segmentation.
```
python local/segment.py   # (Please change your file path in the script)
```
- **Step 3: Audio-visual synchronization detection**

Ensure audio and video synchronization to improve overall data quality.
```
bash syncnet_python-master/download_model.sh
python syncnet_python-master/run.py # (Please change your file path in the script)
python local/synv_result.py  # (Please change your file path in the script)
```
- **Step 4: Face detection and tracking, lip ROI extraction**

In order to better perform audio-visual speaker diarization, we need to get high-quality lip ROI, so we need to perform face detection and tracking as well as lip ROI extraction. We use Retinaface (https://github.com/bubbliiiing/retinaface-pytorch) for face detection, Deepsort (https://github.com/levan92/deep_sort_realtime/tree/master) for face trajectory tracking, and Mediapipe (https://github.com/google-ai-edge/mediapipe) for lip ROI extraction.

First, download the model files required by Retinaface according to retinaface_pytorch/README, then install Deepsort (pip3 install deep-sort-realtime), Mediapipe (pip install mediapipe), and then build deep-person-reid (cd deep-person-reid, python setup.py develop).
```
python local/face_detection_tracking.py  # (Please change your file path in the script)
```
