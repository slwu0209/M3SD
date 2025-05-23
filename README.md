# M3SD
M3SD: Multi-modal, Multi-scenario and Multi-language Speaker Diarization Dataset

# Quick start

- **Step 1: Get audio and video**

You can download our dataset (https://huggingface.co/spaces/OldDragon/m3sd) or crawl audio and video from YouTube according to the code (local/search_video.py)
You first need to get your own YouTube API key (https://console.developers.google.com/apis/api/youtube.googleapis.com) and fill it in the code.
```
python local/search_video.py
```
- **Data prepare**
```
bash data_prepare.sh   # (Please change your file path in the script. Note that WPE is not necessary for training set)
```
