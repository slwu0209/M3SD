# M3SD
M3SD: Multi-modal, Multi-scenario and Multi-language Speaker Diarization Dataset
# Direct use

Our dataset can be downloaded at https://huggingface.co/spaces/OldDragon/m3sd.

The dataset contains 770+ hours of conversations, covering multiple scenarios such as online and offline meetings, home communications, outdoor conversations, interviews, movie clips, news broadcasts, and multiple languages ​​including English and Chinese. The data comes from YouTube and is pseudo-labeled through a variety of speaker diarization systems. 

We will provide audio files, annotation files, and video metadata including uid. You can also download videos from YouTube based on video meta information for multimodal research. It is worth noting that since the speaker diarization labels of the data are not manually annotated, they are not guaranteed to be completely accurate. They can be used for model pre-training, etc.

# Self-build

- **Step 1: Get audio and video**

Crawl audio and video from YouTube.

You first need to get your own [YouTube API key](https://console.developers.google.com/apis/api/youtube.googleapis.com) and fill it in the code.
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

In order to better perform audio-visual speaker diarization, we need to get high-quality lip ROI, so we need to perform face detection and tracking as well as lip ROI extraction. We use [Retinaface](https://github.com/bubbliiiing/retinaface-pytorch) for face detection, [Deepsort](https://github.com/levan92/deep_sort_realtime/tree/master) for face trajectory tracking, and [Mediapipe](https://github.com/google-ai-edge/mediapipe) for lip ROI extraction.

First, download the model files required by Retinaface according to retinaface_pytorch/README, then install Deepsort (pip3 install deep-sort-realtime), Mediapipe (pip install mediapipe), and then build deep-person-reid (cd deep-person-reid, python setup.py develop).
```
python local/face_detection_tracking.py  # (Please change your file path in the script)
```
- **Step 5: Pseudo label generation**

The last step is to use the pre-trained speaker diarization system to generate pseudo labels for the data. To ensure the quality of pseudo labels, we use the audio-only speaker diarization and audio-visual speaker diarization systems to generate results, and then perform voting fusion.

The audio-only speaker diarization uses the [3D-Speaker](https://github.com/modelscope/3D-Speaker) system, and the audio-visual speaker diarization uses the [End-to-end audio-visual speaker diarization](https://github.com/mispchallenge/misp2022_baseline/tree/main/track1_AVSD) system, and finally uses [Dover-Lap](https://github.com/desh2608/dover-lap) for fusion.
Researchers can use these open source systems for pseudo label generation, or use more and better systems to get more accurate results.

- **Optional Steps**

After the first step, you can choose to perform audio and video quality detection to eliminate low-quality audio and video. You can choose [DNSMOS](https://github.com/microsoft/DNS-Challenge) for audio quality detection and [MD-VQA](https://github.com/kunyou99/MD-VQA_cvpr2023?tabreadme-ov-file) for video quality detection.

## Citation

If you find this code useful in your research, please consider to cite the following papers:

```bibtex
@article{wu2025m3sd,
  title={M3SD: Multi-modal, Multi-scenario and Multi-language Speaker Diarization Dataset},
  author={Wu, Shilong},
  journal={arXiv preprint arXiv:2506.14427},
  year={2025}
}
```
