#!/usr/bin/env python3
"""
Enhanced Batch Video Processing Script for TransNetV2
Processes all MP4 files in a specified folder using TransNetV2 for scene detection
and extracts individual shots as separate video files.
"""

import os
import sys
import glob
import argparse
from pathlib import Path
import time
import shutil
import subprocess
from typing import List, Optional, Tuple
import numpy as np
from TransNetV2.inference.transnetv2 import TransNetV2  # Adjust the import based on your project structure

def check_ffmpeg():
    """Check if FFmpeg is available in the system."""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_video_info(video_path: str) -> dict:
    """Get video information using FFprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        import json
        output = result.stdout.decode('utf-8') if isinstance(result.stdout, bytes) else result.stdout
        return json.loads(output)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return {}

def get_video_fps(video_path: str) -> float:
    """Get video frame rate."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate', '-of', 'csv=s=x:p=0',
            video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        output = result.stdout.decode('utf-8') if isinstance(result.stdout, bytes) else result.stdout
        fps_str = output.strip()
        if '/' in fps_str:
            num, den = fps_str.split('/')
            return float(num) / float(den)
        return float(fps_str)
    except:
        return 30.0  # Default fallback

def extract_shot(video_path: str, start_frame: int, end_frame: int, fps: float, 
                output_path: str, copy_streams: bool = True) -> bool:
    """
    Extract a shot from a video using FFmpeg.
    
    Args:
        video_path: Path to the input video
        start_frame: Starting frame number
        end_frame: Ending frame number
        fps: Video frame rate
        output_path: Path for the output video
        copy_streams: Whether to copy streams (faster) or re-encode
    
    Returns:
        True if extraction was successful
    """
    try:
        # Convert frame numbers to timestamps
        start_time = start_frame / fps
        duration = (end_frame - start_frame) / fps
        
        print(f"[DEBUG] Extracting from {start_time:.3f}s for {duration:.3f}s")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if copy_streams:
            # Fast extraction by copying streams (no re-encoding)
            cmd = [
                'ffmpeg', '-y', '-ss', str(start_time), '-i', video_path,
                '-t', str(duration), '-c', 'copy', '-avoid_negative_ts', 'make_zero',
                output_path
            ]
        else:
            # Re-encode (slower but more accurate)
            cmd = [
                'ffmpeg', '-y', '-ss', str(start_time), '-i', video_path,
                '-t', str(duration), '-c:v', 'libx264', '-c:a', 'aac',
                '-preset', 'fast', output_path
            ]
        
        print(f"[DEBUG] FFmpeg command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            # Decode stderr for error message
            stderr_output = result.stderr.decode('utf-8') if isinstance(result.stderr, bytes) else result.stderr
            print(f"[ERROR] FFmpeg failed with return code {result.returncode}")
            print(f"[ERROR] FFmpeg stderr: {stderr_output}")
            return False
        
        # Check if file was created and has content
        if not os.path.exists(output_path):
            print(f"[ERROR] Output file was not created: {output_path}")
            return False
        
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            print(f"[ERROR] Output file is empty: {output_path}")
            return False
        
        print(f"[DEBUG] Successfully created file: {output_path} ({file_size} bytes)")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to extract shot: {str(e)}")
        return False

def find_video_files(folder_path: str, extensions: List[str] = None) -> List[str]:
    """
    Find all video files in the specified folder.
    
    Args:
        folder_path: Path to the folder containing videos
        extensions: List of video file extensions to look for
    
    Returns:
        List of video file paths
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    
    video_files = []
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder_path}")
    
    for ext in extensions:
        pattern = f"*{ext}"
        files = list(folder_path.glob(pattern))
        files.extend(list(folder_path.glob(pattern.upper())))  # Include uppercase extensions
        video_files.extend(files)
    
    return sorted([str(f) for f in video_files])

def get_output_paths(video_path: str, output_folder: Optional[str] = None) -> dict:
    """
    Get the output file paths for a video, either next to the original or in output folder.
    
    Args:
        video_path: Path to the video file
        output_folder: Optional output folder path
    
    Returns:
        Dictionary with output file paths
    """
    video_path = Path(video_path)
    
    if output_folder is None:
        # Save next to original video
        base_path = str(video_path.with_suffix(''))
        shots_folder = video_path.parent / f"{video_path.stem}_shots"
    else:
        # Save in output folder, maintaining directory structure
        output_folder = Path(output_folder)
        relative_path = video_path.name
        base_path = str(output_folder / Path(relative_path).with_suffix(''))
        shots_folder = output_folder / f"{video_path.stem}_shots"
    
    return {
        'predictions': base_path + ".predictions.txt",
        'scenes': base_path + ".scenes.txt",
        'visualization': base_path + ".vis.png",
        'shots_folder': str(shots_folder)
    }

def is_already_processed(video_path: str, output_folder: Optional[str] = None, 
                        check_scenes: bool = True, check_predictions: bool = True,
                        check_shots: bool = False) -> bool:
    """
    Check if a video has already been processed by looking for output files.
    
    Args:
        video_path: Path to the video file
        output_folder: Optional output folder path
        check_scenes: Whether to check for .scenes.txt file
        check_predictions: Whether to check for .predictions.txt file
        check_shots: Whether to check for shots folder
    
    Returns:
        True if already processed
    """
    output_paths = get_output_paths(video_path, output_folder)
    
    scenes_exists = os.path.exists(output_paths['scenes']) if check_scenes else False
    predictions_exists = os.path.exists(output_paths['predictions']) if check_predictions else False
    shots_exists = os.path.exists(output_paths['shots_folder']) and \
                  len(os.listdir(output_paths['shots_folder'])) > 0 if check_shots else False
    
    return scenes_exists or predictions_exists or shots_exists

def extract_all_shots(video_path: str, scenes: np.ndarray, output_folder: str, 
                     fps: float, min_shot_duration: float = 0.5,
                     max_shots: Optional[int] = None, copy_streams: bool = True) -> Tuple[int, int]:
    """
    Extract all shots from a video based on scene boundaries.
    
    Args:
        video_path: Path to the input video
        scenes: Array of scene boundaries (start, end frames)
        output_folder: Folder to save shot videos
        fps: Video frame rate
        min_shot_duration: Minimum shot duration in seconds
        max_shots: Maximum number of shots to extract (None for all)
        copy_streams: Whether to copy streams for faster extraction
    
    Returns:
        Tuple of (successful_extractions, total_shots)
    """
    video_path = Path(video_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"[DEBUG] Found {len(scenes)} scenes to process")
    print(f"[DEBUG] Output folder: {output_folder}")
    print(f"[DEBUG] Video FPS: {fps}")
    print(f"[DEBUG] Min shot duration: {min_shot_duration}s")
    
    successful = 0
    total_shots = len(scenes)
    
    if total_shots == 0:
        print("[WARNING] No scenes found in video!")
        return 0, 0
    
    # Limit number of shots if specified
    if max_shots is not None:
        scenes = scenes[:max_shots]
        total_shots = len(scenes)
        print(f"[DEBUG] Limited to {total_shots} shots")
    
    for i, (start_frame, end_frame) in enumerate(scenes):
        print(f"[DEBUG] Processing scene {i+1}: frames {start_frame} to {end_frame}")
        
        # Check minimum duration
        duration = (end_frame - start_frame) / fps
        print(f"[DEBUG] Shot duration: {duration:.2f}s")
        
        if duration < min_shot_duration:
            print(f"[SKIP] Shot {i+1}: Too short ({duration:.2f}s < {min_shot_duration}s)")
            continue
        
        # Generate output filename
        shot_filename = f"{video_path.stem}_shot_{i+1:04d}_{start_frame:06d}-{end_frame:06d}.mp4"
        shot_path = output_folder / shot_filename
        
        print(f"[DEBUG] Output path: {shot_path}")
        
        # Skip if already exists
        if shot_path.exists():
            print(f"[SKIP] Shot {i+1}: Already exists")
            successful += 1
            continue
        
        print(f"[EXTRACT] Shot {i+1}/{total_shots}: frames {start_frame}-{end_frame} ({duration:.2f}s)")
        
        success = extract_shot(
            str(video_path), start_frame, end_frame, fps, str(shot_path), copy_streams
        )
        
        if success:
            if os.path.exists(str(shot_path)) and os.path.getsize(str(shot_path)) > 0:
                successful += 1
                file_size = os.path.getsize(str(shot_path)) / (1024 * 1024)  # MB
                print(f"[SUCCESS] Saved {shot_filename} ({file_size:.1f} MB)")
            else:
                print(f"[ERROR] Shot file created but is empty or missing: {shot_path}")
        else:
            print(f"[ERROR] Failed to extract shot {i+1}")
    
    print(f"[DEBUG] Extraction complete: {successful}/{total_shots} shots successful")
    return successful, total_shots

def process_video_safely(model, video_path: str, output_folder: Optional[str] = None, 
                        visualize: bool = False, skip_existing: bool = True,
                        extract_shots: bool = False, min_shot_duration: float = 0.5,
                        max_shots: Optional[int] = None, copy_streams: bool = True) -> bool:
    """
    Process a single video file with error handling and optional shot extraction.
    
    Args:
        model: TransNetV2 model instance
        video_path: Path to the video file
        output_folder: Optional output folder path
        visualize: Whether to create visualization
        skip_existing: Whether to skip already processed videos
        extract_shots: Whether to extract individual shots
        min_shot_duration: Minimum shot duration in seconds
        max_shots: Maximum number of shots to extract
        copy_streams: Whether to copy streams for faster extraction
    
    Returns:
        True if processing was successful, False otherwise
    """
    try:
        # Check if already processed
        if skip_existing and is_already_processed(video_path, output_folder, 
                                                 check_shots=extract_shots):
            print(f"[SKIP] {video_path} - Already processed")
            return True
        
        # Create output folder if specified
        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
        
        # Get output paths
        output_paths = get_output_paths(video_path, output_folder)
        
        print(f"\n[PROCESSING] {video_path}")
        if output_folder:
            print(f"[OUTPUT] Saving to: {output_folder}")
        start_time = time.time()
        
        # Process the video
        video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(video_path)
        
        # Save predictions
        predictions = np.stack([single_frame_predictions, all_frame_predictions], 1)
        np.savetxt(output_paths['predictions'], predictions, fmt="%.6f")
        print(f"[SAVED] {output_paths['predictions']}")
        
        # Get scenes
        scenes = model.predictions_to_scenes(single_frame_predictions)
        
        # Save scenes
        np.savetxt(output_paths['scenes'], scenes, fmt="%d")
        print(f"[SAVED] {output_paths['scenes']}")
        
        # Extract shots if requested
        if extract_shots and len(scenes) > 0:
            print(f"[SHOTS] Extracting {len(scenes)} shots...")
            
            # Get video FPS
            fps = get_video_fps(video_path)
            print(f"[INFO] Video FPS: {fps:.2f}")
            
            successful_shots, total_shots = extract_all_shots(
                video_path, scenes, output_paths['shots_folder'], fps,
                min_shot_duration, max_shots, copy_streams
            )
            
            print(f"[SHOTS] Successfully extracted {successful_shots}/{total_shots} shots")
            print(f"[SHOTS] Saved to: {output_paths['shots_folder']}")
        
        # Create visualization if requested
        if visualize:
            vis_path = output_paths['visualization']
            if not os.path.exists(vis_path):
                pil_image = model.visualize_predictions(
                    video_frames, predictions=(single_frame_predictions, all_frame_predictions)
                )
                pil_image.save(vis_path)
                print(f"[SAVED] {vis_path}")
            else:
                print(f"[SKIP] {vis_path} - Already exists")
        
        elapsed_time = time.time() - start_time
        print(f"[SUCCESS] Processed {video_path} in {elapsed_time:.2f} seconds")
        print(f"[INFO] Found {len(scenes)} scenes")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to process {video_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch process videos using TransNetV2 with shot extraction")
    parser.add_argument("folder", type=str, help="Path to folder containing MP4 videos")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output folder for generated files (default: save next to original videos)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to TransNet V2 weights, tries to infer the location if not specified")
    parser.add_argument("--visualize", action="store_true",
                        help="Save a png file with prediction visualization for each video")
    parser.add_argument("--extensions", type=str, nargs="+", 
                        default=[".mp4"], 
                        help="Video file extensions to process (default: .mp4)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip videos that have already been processed")
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing",
                        help="Process all videos, even if already processed")
    parser.add_argument("--recursive", action="store_true",
                        help="Search for videos recursively in subfolders")
    
    # Shot extraction arguments
    parser.add_argument("--extract-shots", action="store_true",
                        help="Extract individual shots as separate video files")
    parser.add_argument("--min-shot-duration", type=float, default=0.5,
                        help="Minimum shot duration in seconds (default: 0.5)")
    parser.add_argument("--max-shots", type=int, default=None,
                        help="Maximum number of shots to extract per video (default: all)")
    parser.add_argument("--re-encode", action="store_true",
                        help="Re-encode shots instead of copying streams (slower but more accurate)")
    
    args = parser.parse_args()
    
    try:
        # Check FFmpeg availability if shot extraction is requested
        if args.extract_shots and not check_ffmpeg():
            print("[ERROR] FFmpeg is required for shot extraction but not found in PATH")
            print("Please install FFmpeg or disable shot extraction")
            sys.exit(1)
        
        # Initialize the model
        print("[INIT] Loading TransNetV2 model...")
        model = TransNetV2(args.weights)
        print("[INIT] Model loaded successfully")
        
        # Find video files
        print(f"[SEARCH] Looking for videos in: {args.folder}")
        if args.recursive:
            # Recursive search
            video_files = []
            for ext in args.extensions:
                pattern = os.path.join(args.folder, "**", f"*{ext}")
                video_files.extend(glob.glob(pattern, recursive=True))
                # Also search for uppercase extensions
                pattern = os.path.join(args.folder, "**", f"*{ext.upper()}")
                video_files.extend(glob.glob(pattern, recursive=True))
            video_files = sorted(list(set(video_files)))  # Remove duplicates and sort
        else:
            video_files = find_video_files(args.folder, args.extensions)
        
        if not video_files:
            print(f"[WARNING] No video files found in {args.folder}")
            return
        
        print(f"[FOUND] {len(video_files)} video file(s)")
        
        # Setup output folder if specified
        if args.output:
            output_folder = Path(args.output)
            output_folder.mkdir(parents=True, exist_ok=True)
            print(f"[OUTPUT] Results will be saved to: {output_folder}")
        else:
            print("[OUTPUT] Results will be saved next to original videos")
        
        if args.extract_shots:
            print(f"[SHOTS] Shot extraction enabled")
            print(f"[SHOTS] Minimum duration: {args.min_shot_duration}s")
            if args.max_shots:
                print(f"[SHOTS] Maximum shots per video: {args.max_shots}")
            print(f"[SHOTS] Mode: {'Re-encode' if args.re_encode else 'Copy streams (fast)'}")
        
        # Process each video
        successful = 0
        failed = 0
        skipped = 0
        total_shots_extracted = 0
        
        for i, video_path in enumerate(video_files, 1):
            print(f"\n{'='*60}")
            print(f"Processing video {i}/{len(video_files)}")
            
            if args.skip_existing and is_already_processed(video_path, args.output, 
                                                          check_shots=args.extract_shots):
                print(f"[SKIP] {video_path} - Already processed")
                skipped += 1
                continue
            
            success = process_video_safely(
                model, 
                video_path, 
                output_folder=args.output,
                visualize=args.visualize,
                skip_existing=args.skip_existing,
                extract_shots=args.extract_shots,
                min_shot_duration=args.min_shot_duration,
                max_shots=args.max_shots,
                copy_streams=not args.re_encode
            )
            
            if success:
                successful += 1
                # Count extracted shots if enabled
                if args.extract_shots:
                    output_paths = get_output_paths(video_path, args.output)
                    shots_folder = Path(output_paths['shots_folder'])
                    if shots_folder.exists():
                        shot_count = len([f for f in shots_folder.glob("*.mp4")])
                        total_shots_extracted += shot_count
            else:
                failed += 1
        
        # Print summary
        print(f"\n{'='*60}")
        print("[SUMMARY]")
        print(f"Total videos found: {len(video_files)}")
        print(f"Successfully processed: {successful}")
        print(f"Failed: {failed}")
        print(f"Skipped: {skipped}")
        if args.extract_shots:
            print(f"Total shots extracted: {total_shots_extracted}")
        print("Batch processing completed!")
        
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"[FATAL ERROR] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()