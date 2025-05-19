import os
import subprocess
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import GPUtil
import threading
import time
from queue import Queue

# Set root folder path
root_folder = '/disk5/chime/mm/data/meeting/segment/'

# Set output directory path
output_folder = '/disk5/chime/mm/syncnet_python-master/output/meeting'

# Configure log file path
log_file = '/disk5/chime/mm/syncnet_python-master/process_log_meeting.log'

# Set GPU environment variable
def set_gpu_env(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

def get_available_gpu(preferred_gpus, threshold_mem=0.01):
    available_gpu = None
    for gpu_id in preferred_gpus:
        try:
            gpu = GPUtil.getGPUs()[gpu_id]
            if gpu.memoryUtil < threshold_mem:
                available_gpu = gpu_id
                break
        except IndexError:
            logging.error(f"GPU {gpu_id} does not exist.")
    return available_gpu

def process_video(video_info, gpu_id, subfolder, progress_dict, lock):
    # Initialize logging in subprocess
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    video_path, video_file = video_info
    video_name = os.path.splitext(video_file)[0]
    reference_folder = os.path.join(output_folder, subfolder)

    # Set GPU
    set_gpu_env(gpu_id)

    try:
        # Run run_pipeline.py
        pipeline_command = [
            "python", "run_pipeline.py", 
            "--videofile", video_path,
            "--reference", video_name,
            "--data_dir", reference_folder
        ]
        # Capture stdout and stderr via PIPE
        result = subprocess.run(pipeline_command, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
    except subprocess.CalledProcessError as e:
        # Capture and log detailed error
        logging.error(f"Error running run_pipeline.py for video: {video_path}, GPU: {gpu_id}, Error: {e}")
        return

    try:
        # Run run_syncnet.py
        syncnet_command = [
            "python", "run_syncnet.py", 
            "--videofile", video_path,
            "--reference", video_name,
            "--data_dir", reference_folder
        ]
        result = subprocess.run(syncnet_command, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
    except subprocess.CalledProcessError as e:
        # Capture and log detailed error
        logging.error(f"Error running run_syncnet.py for video: {video_path}, GPU: {gpu_id}, Error: {e}")
        return

    # Update progress dictionary
    with lock:
        progress_dict[subfolder] += 1

    return True

def process_subfolder(args):
    subfolder, gpu_id, progress_dict, lock = args
    subfolder_path = os.path.join(root_folder, subfolder)
    video_folder = os.path.join(subfolder_path, 'video')

    # Initialize logging in subprocess
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Skip if the subfolder already exists in the output directory
    reference_folder = os.path.join(output_folder, subfolder)
    if os.path.exists(reference_folder):
        logging.info(f"Skipping {subfolder}, already processed.")
        return subfolder  # Return subfolder name

    if os.path.isdir(video_folder):
        video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

        if not video_files:
            logging.warning(f"No mp4 video files found in subfolder {subfolder}.")
            return subfolder
        try:
            os.makedirs(reference_folder, exist_ok=True)
        except Exception as e:
            logging.error(f"Error creating output folder for subfolder: {subfolder}, Error: {e}")
            return subfolder

        video_info = [(os.path.join(video_folder, f), f) for f in video_files]

        # Get available GPU
        preferred_gpus = [gpu_id] + [x for x in [0, 1, 2, 3] if x != gpu_id]
        gpu_id = get_available_gpu(preferred_gpus=preferred_gpus)
        if gpu_id is None:
            logging.error(f"No available GPU to process subfolder {subfolder}")
            return subfolder

        logging.info(f"Starting to process {subfolder} using GPU {gpu_id}")

        # Initialize progress_dict
        with lock:
            progress_dict[subfolder] = 0

        # Use thread pool to process video files, avoiding nested process pools
        from concurrent.futures import ThreadPoolExecutor, as_completed

        processed_count = 0

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(process_video, video, gpu_id, subfolder, progress_dict, lock)
                for video in video_info
            ]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        processed_count += 1
                except Exception as e:
                    logging.error(f"Error processing video in subfolder: {subfolder}, Error: {e}")

        logging.info(f"Completed processing {subfolder}, processed {processed_count} videos")
        return subfolder  # Return subfolder name
    else:
        logging.warning(f"No video folder found in {subfolder_path}.")
        return subfolder

def update_progress_bars(progress_dict, pbar_dict, pbar_lock, lock, total_dict, stop_event):
    while not stop_event.is_set():
        with pbar_lock:
            with lock:
                for subfolder, pbar in pbar_dict.items():
                    current = progress_dict.get(subfolder, 0)
                    pbar.n = current
                    pbar.refresh()
        time.sleep(0.5)
    # Final update
    with pbar_lock:
        with lock:
            for subfolder, pbar in pbar_dict.items():
                current = progress_dict.get(subfolder, 0)
                if current < total_dict[subfolder]:
                    pbar.update(total_dict[subfolder] - pbar.n)
                pbar.close()

def process_subfolders():
    # Initialize logging in main process
    if os.path.exists(log_file):
        os.remove(log_file)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Get all subfolders
    subfolders = [s for s in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, s))]

    manager = multiprocessing.Manager()
    progress_dict = manager.dict()
    lock = manager.Lock()

    # Filter subfolders to process
    subfolders_to_process = []
    total_dict = {}
    for subfolder in subfolders:
        reference_folder = os.path.join(output_folder, subfolder)
        if os.path.exists(reference_folder):
            logging.info(f"Skipping {subfolder}, already processed.")
            continue  # Skip processed subfolders
        subfolder_path = os.path.join(root_folder, subfolder)
        video_folder = os.path.join(subfolder_path, 'video')
        if os.path.isdir(video_folder):
            video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
            if video_files:
                subfolders_to_process.append(subfolder)
                total_dict[subfolder] = len(video_files)
                progress_dict[subfolder] = 0
            else:
                logging.warning(f"No mp4 video files found in subfolder {subfolder}.")
        else:
            logging.warning(f"No video folder found in {subfolder_path}.")

    if not subfolders_to_process:
        logging.info("No subfolders need processing.")
        return

    # GPU assignment
    gpu_assignment = [0, 1, 2, 3]  # GPU IDs

    # Prepare subfolders to process and their assigned GPUs
    assigned_subfolders = []
    for i, subfolder in enumerate(subfolders_to_process):
        gpu_id = gpu_assignment[i % len(gpu_assignment)]
        assigned_subfolders.append((subfolder, gpu_id))

    # Create a queue to manage subfolders to process
    subfolder_queue = Queue()
    for subfolder, gpu_id in assigned_subfolders:
        subfolder_queue.put((subfolder, gpu_id))

    # Create progress bar dictionary to track currently processing subfolders
    pbar_dict = {}

    # Create lock to protect pbar_dict
    pbar_lock = threading.Lock()

    # Create stop event
    stop_event = threading.Event()

    # Start progress bar update thread
    updater_thread = threading.Thread(target=update_progress_bars, args=(progress_dict, pbar_dict, pbar_lock, lock, total_dict, stop_event))
    updater_thread.start()

    # Use ProcessPoolExecutor for parallel subfolder processing
    with ProcessPoolExecutor(max_workers=len(gpu_assignment)) as executor:
        # Submit initial tasks equal to the number of GPUs
        futures = {}
        for gpu_id in gpu_assignment:
            if not subfolder_queue.empty():
                subfolder, assigned_gpu_id = subfolder_queue.get()
                # Create progress bar
                with pbar_lock:
                    pbar = tqdm(
                        total=total_dict[subfolder],
                        desc=f"Processing {subfolder}",
                        position=gpu_id,
                        leave=True
                    )
                    pbar_dict[subfolder] = pbar
                # Submit task
                future = executor.submit(process_subfolder, (subfolder, assigned_gpu_id, progress_dict, lock))
                futures[future] = gpu_id

        while futures:
            # Get the next completed future
            done = next(as_completed(futures))
            gpu_id = futures.pop(done)
            try:
                subfolder_done = done.result()
            except Exception as e:
                logging.error(f"Task failed: GPU {gpu_id}, Error: {e}")
                subfolder_done = None

            if subfolder_done and subfolder_done in pbar_dict:
                # Ensure progress bar is updated to total before closing
                with pbar_lock:
                    pbar = pbar_dict[subfolder_done]
                    # Calculate remaining updates
                    remaining = total_dict[subfolder_done] - pbar.n
                    if remaining > 0:
                        pbar.update(remaining)
                    pbar.close()
                    del pbar_dict[subfolder_done]

            # Assign new subfolder to this GPU
            if not subfolder_queue.empty():
                subfolder_new, assigned_gpu_id_new = subfolder_queue.get()
                # Create new progress bar
                with pbar_lock:
                    pbar_new = tqdm(
                        total=total_dict[subfolder_new],
                        desc=f"Processing {subfolder_new}",
                        position=gpu_id,
                        leave=True
                    )
                    pbar_dict[subfolder_new] = pbar_new
                # Submit new task
                future_new = executor.submit(process_subfolder, (subfolder_new, assigned_gpu_id_new, progress_dict, lock))
                futures[future_new] = gpu_id

    # Processing complete, stop update thread
    stop_event.set()
    updater_thread.join()

    # Ensure all progress bars are closed
    with pbar_lock:
        for pbar in pbar_dict.values():
            pbar.close()

if __name__ == "__main__":
    process_subfolders()