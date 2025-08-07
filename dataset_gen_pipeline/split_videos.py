import os
import json
import cv2
import numpy as np
import argparse
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_files(root_dir):
    """Extract all video and JSON files from the given directory recursively."""
    video_extensions = ('.mp4', '.avi', '.mkv')
    video_files = []
    json_files = []
    
    try:
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                file_path = os.path.join(dirpath, file)
                if file.lower().endswith(video_extensions):
                    video_files.append(file_path)
                elif file.lower().endswith('.json'):
                    json_files.append(file_path)
        return video_files, json_files
    except Exception as e:
        logger.error(f"Error extracting files from {root_dir}: {e}")
        return [], []

def read_json(json_file):
    """Read and parse a JSON file."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error reading JSON file {json_file}: {e}")
        return []

def write_json_segments(output_dir, json_data):
    """Write or append JSON segments to a file."""
    json_path = os.path.join(output_dir, "vid_segments.json")
    try:
        if not os.path.exists(json_path):
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
        else:
            with open(json_path, 'r+', encoding='utf-8') as f:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    raise ValueError("JSON file content must be a list")
                
                # Append new data, avoiding duplicates
                for new_item in json_data:
                    if new_item not in existing_data:
                        existing_data.append(new_item)
                
                f.seek(0)
                json.dump(existing_data, f, indent=4, ensure_ascii=False)
                f.truncate()
    except Exception as e:
        logger.error(f"Error writing JSON segments to {json_path}: {e}")

def segment_video(video_path, segments, output_dir):
    """
    Segment a video based on given timestamps.
    
    Args:
        video_path (str): Path to the input video file.
        segments (list): List of tuples [(start, end), ...] with timestamps in seconds.
        output_dir (str): Directory to save segmented videos.
    """
    try:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if all segments already exist
        all_segments_exist = all(
            os.path.exists(os.path.join(output_dir, f"{video_name}_segment_{idx + 1}.mp4"))
            for idx, _ in enumerate(segments)
        )
        if all_segments_exist:
            logger.info(f"All segments for {video_name} already exist, skipping.")
            return

        with VideoFileClip(video_path) as video:
            video_duration = video.duration
            
            for idx, (start, end) in enumerate(segments):
                segment_output_path = os.path.join(output_dir, f"{video_name}_segment_{idx + 1}.mp4")
                
                if os.path.exists(segment_output_path):
                    logger.info(f"Segment {idx + 1} for {video_name} already exists, skipping.")
                    continue
                
                # Validate timestamps
                end = min(end, video_duration)
                if start < 0 or end <= start:
                    logger.warning(f"Invalid segment: start={start}, end={end}, skipping.")
                    continue
                
                # Extract and save segment
                segment = video.subclip(start, end)
                try:
                    segment.write_videofile(segment_output_path, codec='libx264', logger=None)
                except Exception as e:
                    logger.warning(f"Error processing segment {idx + 1}: {e}. Retrying without audio...")
                    try:
                        segment.write_videofile(segment_output_path, codec='libx264', audio=False, logger=None)
                    except Exception as e2:
                        logger.error(f"Failed to process segment {idx + 1} without audio: {e2}")
    except Exception as e:
        logger.error(f"Error segmenting video {video_path}: {e}")

def sample_frames(segment_path, segment_idx, output_dir, num_frames):
    """
    Sample frames uniformly from a video segment and save as images.
    
    Args:
        segment_path (str): Path to the video segment.
        segment_idx (int): Index of the segment.
        output_dir (str): Directory to save sampled frames.
        num_frames (int): Number of frames to sample.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(segment_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video segment {segment_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        # Adjust num_frames based on duration
        if duration < 1:
            num_frames = 1
        elif duration < num_frames:
            num_frames = max(1, round(duration))
        elif duration > num_frames * 5:
            num_frames = max(1, int(duration // 5))
        
        frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Failed to read frame {frame_idx} from {segment_path}")
                continue
            
            # Resize frame if necessary
            h, w = frame.shape[:2]
            max_dim = max(h, w)
            if max_dim > 512:
                scale = 512 / max_dim
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            
            # Save frame
            frame_output_path = os.path.join(output_dir, f"keyframe_{segment_idx}_{i + 1}.jpg")
            cv2.imwrite(frame_output_path, frame)
        
        cap.release()
    except Exception as e:
        logger.error(f"Error sampling frames from {segment_path}: {e}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Segment videos and sample keyframes based on JSON timestamps.")
    parser.add_argument('--input_dir', type=str, default='Dataset1', 
                        help='Root directory containing videos and JSON files')
    parser.add_argument('--output_dir', type=str, default='output', 
                        help='Directory to save segmented videos and keyframes')
    parser.add_argument('--num_frames', type=int, default=20, 
                        help='Number of frames to sample from each video segment (adjusted based on duration)')
    return parser.parse_args()

def main():
    """Main function to process videos and JSON files."""
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    num_frames = args.num_frames
    
    # Validate input directory
    if not os.path.isdir(input_dir):
        logger.error(f"Input directory {input_dir} does not exist")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract files
    video_files, json_files = extract_files(input_dir)
    if not json_files:
        logger.error("No JSON files found in the input directory")
        return
    if not video_files:
        logger.error("No video files found in the input directory")
        return
    
    # Process each JSON file
    for json_file in json_files:
        datas = read_json(json_file)
        if not datas:
            continue
        
        write_json_segments(output_dir, datas)
        
        for data in tqdm(datas, desc="Segmenting videos"):
            video_name = data.get("name")
            time_stamps = data.get("time_stamps")
            
            if not video_name or not time_stamps:
                logger.warning(f"Invalid data in JSON: {json_file}")
                continue
            
            # Find matching video file
            video_path = next((v for v in video_files if os.path.basename(v).split('.')[0] == video_name), None)
            if not video_path:
                logger.warning(f"No video file found for {video_name}")
                continue
            
            # Define output directories
            output_dir_segments = os.path.join(output_dir, "segments", video_name)
            output_dir_frames = os.path.join(output_dir, "keyframes", video_name)
            
            # Segment video
            logger.info(f"Processing video {video_name}")
            segment_video(video_path, time_stamps, output_dir_segments)
            
            # Sample frames from segments
            if os.path.exists(output_dir_segments):
                for segment_file in os.listdir(output_dir_segments):
                    if not segment_file.endswith('.mp4'):
                        continue
                    segment_idx = os.path.splitext(segment_file)[0].split('_')[-1]
                    segment_path = os.path.join(output_dir_segments, segment_file)
                    sample_frames(segment_path, segment_idx, output_dir_frames, num_frames)

if __name__ == "__main__":
    main()