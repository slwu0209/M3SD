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

In order to ensure the effect of subsequent face detection and tracking, it is necessary to perform scene detection and segmentation
```
python local/segment.py   # (Please change your file path in the scriptt)
```
