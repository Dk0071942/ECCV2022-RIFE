import os
import sys
import subprocess
import shutil
import tempfile
import argparse
import glob
import cv2
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import tqdm
import time
import uuid
import re

# --- Helper Functions (Adapted from ECCV2022-RIFE/video_transition.py and modified) ---

DEFAULT_FFMPEG_TIMEOUT = 180  # 3 minutes for most ffmpeg operations
DEFAULT_FFPROBE_TIMEOUT = 60 # 1 minute for ffprobe
DEFAULT_RIFE_TIMEOUT = 600    # 10 minutes for RIFE interpolation

def _print_step(message):
    print(f"\n>>> {message}")

def _print_info(message):
    print(f"    INFO: {message}")

def _print_error(message):
    print(f"    ERROR: {message}")

def _print_warning(message):
    print(f"    WARNING: {message}")

def _sanitize_filename(filename):
    """
    Sanitize filename to avoid Windows path issues.
    Removes or replaces problematic characters and limits length.
    """
    # Replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)  # Windows forbidden chars
    sanitized = re.sub(r'[,\s\.]+', '_', sanitized)     # Commas, spaces, periods
    sanitized = re.sub(r'[äöüßÄÖÜ]', lambda m: {'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss', 'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue'}[m.group()], sanitized)  # German chars
    sanitized = re.sub(r'_+', '_', sanitized)           # Multiple underscores to single
    sanitized = sanitized.strip('_')                    # Remove leading/trailing underscores
    
    # Limit length (reserve space for UUID and extensions)
    if len(sanitized) > 50:
        sanitized = sanitized[:50]
    
    return sanitized

def _safe_rmtree(path, max_retries=3, delay=1.0):
    """
    Safely remove a directory tree with retries for Windows file locking issues.
    """
    if not os.path.exists(path):
        return True
        
    for attempt in range(max_retries):
        try:
            shutil.rmtree(path)
            return True
        except PermissionError as e:
            if attempt < max_retries - 1:
                _print_warning(f"Permission error removing {path} (attempt {attempt + 1}/{max_retries}). Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                _print_error(f"Failed to remove {path} after {max_retries} attempts: {e}")
                return False
        except Exception as e:
            _print_error(f"Unexpected error removing {path}: {e}")
            return False
    return False

def _extract_frames_util(video_path, output_dir, frame_prefix="frame_", quality=1, specific_frame_number=None):
    """
    Extracts frames from a video using FFmpeg.
    Can extract all frames, a specific frame, or just the first frame.
    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save extracted frames.
        frame_prefix (str): Prefix for saved frame filenames.
        quality (int): FFmpeg q:v quality (1-31, lower is better).
        specific_frame_number (int, optional): 1-based frame number to extract.
                                              If None, extracts all.
                                              If 'first', extracts only the first.
    Returns:
        list: Sorted list of paths to extracted frames.
        None: If extraction fails.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure consistent color space and pixel format
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-q:v', str(quality),
        '-vf', 'format=yuv420p,colorspace=all=bt709:iall=bt709:fast=1',  # Force consistent color space
        '-color_primaries', 'bt709',
        '-color_trc', 'bt709',
        '-colorspace', 'bt709'
    ]

    if specific_frame_number == 'first':
        cmd.extend(['-vframes', '1', os.path.join(output_dir, f"{frame_prefix}000001.png")])
    elif isinstance(specific_frame_number, int) and specific_frame_number > 0:
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vf', f"select='eq(n,{specific_frame_number-1})',format=yuv420p,colorspace=all=bt709:iall=bt709:fast=1",
            '-color_primaries', 'bt709',
            '-color_trc', 'bt709',
            '-colorspace', 'bt709',
            '-vsync', 'vfr',
            '-frames:v', '1',
            os.path.join(output_dir, f"{frame_prefix}{specific_frame_number:06d}.png")
        ]
    else: # Extract all frames
        cmd.append(os.path.join(output_dir, f"{frame_prefix}%06d.png"))

    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=DEFAULT_FFMPEG_TIMEOUT)
        extracted_files = sorted(glob.glob(os.path.join(output_dir, f"{frame_prefix}*.png")))
        if not extracted_files:
            _print_error(f"No frames extracted. FFmpeg command was: {' '.join(cmd)}")
            if process.stdout: _print_info(f"FFmpeg STDOUT:\n{process.stdout[:500]}")
            if process.stderr: _print_error(f"FFmpeg STDERR:\n{process.stderr[:1000]}")
            return None
        return extracted_files
    except subprocess.CalledProcessError as e:
        _print_error(f"Failed to extract frames from {os.path.basename(video_path)}.")
        _print_error(f"FFmpeg command: {' '.join(e.cmd)}")
        _print_error(f"FFmpeg stderr:\n{e.stderr}")
        return None
    except subprocess.TimeoutExpired:
        _print_error(f"FFmpeg timed out while extracting frames from {os.path.basename(video_path)} after {DEFAULT_FFMPEG_TIMEOUT} seconds.")
        _print_error(f"Command was: {' '.join(cmd)}")
        return None
    except Exception as e_gen:
        _print_error(f"An unexpected error occurred during frame extraction: {e_gen}")
        return None

def _get_video_fps(video_path):
    """Get the FPS of a video file."""
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', video_path
    ]
    try:
        output = subprocess.check_output(cmd, timeout=DEFAULT_FFPROBE_TIMEOUT).decode().strip()
        if '/' in output:
            num, den = map(int, output.split('/'))
            return num / den if den != 0 else 30.0 # Default on error
        return float(output)
    except Exception as e:
        _print_warning(f"Could not get FPS for {video_path}. Defaulting to 30. Error: {e}")
        return 30.0
    except subprocess.TimeoutExpired:
        _print_warning(f"ffprobe timed out while getting FPS for {video_path} after {DEFAULT_FFPROBE_TIMEOUT} seconds. Defaulting to 30.")
        return 30.0

def _get_image_dimensions(image_path):
    """Reads an image and returns its (height, width). Returns None on failure."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            _print_error(f"Could not read image for dimension check: {image_path}")
            return None
        return img.shape[:2] # Returns (height, width)
    except Exception as e:
        _print_error(f"Error reading image dimensions for {image_path}: {e}")
        return None

def _run_rife_interpolation(rife_inference_script_path, frame_a_path, frame_b_path, rife_exp,
                           rife_parent_work_dir, unique_job_name):
    """
    Runs RIFE interpolation using the modified inference_img.py.
    Prepares a specific input job directory for inference_img.py and retrieves its output.
    Args:
        rife_inference_script_path (str): Absolute path to the modified 'inference_img.py'.
        frame_a_path (str): Path to the first input frame (e.g., source_last.png).
        frame_b_path (str): Path to the second input frame (e.g., neutral_first.png).
        rife_exp (int): RIFE exponent for interpolation.
        rife_parent_work_dir (str): A parent temporary directory where RIFE's input job dir will be created.
        unique_job_name (str): A unique name for this interpolation job (used for subdir name).
    Returns:
        str: Path to the directory containing the final, sequentially named interpolated frames 
             (frame_000001.png ...), or None on failure.
    """
    import time
    import uuid
    
    # Add a small random delay to stagger job starts and reduce race conditions
    time.sleep(0.1 + (hash(unique_job_name) % 1000) / 10000.0)  # 0.1-0.2 second stagger
    
    # Sanitize the job name to avoid Windows path issues
    sanitized_job_name = _sanitize_filename(unique_job_name)
    
    # Create a more unique job directory to avoid conflicts
    unique_suffix = str(uuid.uuid4())[:8]
    rife_job_input_dir = os.path.join(rife_parent_work_dir, f"rife_job_{sanitized_job_name}_{unique_suffix}")
    expected_rife_output_dir = os.path.join(rife_job_input_dir, 'output')

    if os.path.exists(rife_job_input_dir):
        _safe_rmtree(rife_job_input_dir)
    os.makedirs(rife_job_input_dir)

    # Verify input frames exist and are readable before copying
    if not os.path.isfile(frame_a_path):
        _print_error(f"Frame A does not exist: {frame_a_path}")
        return None
    if not os.path.isfile(frame_b_path):
        _print_error(f"Frame B does not exist: {frame_b_path}")
        return None

    start_png_path = os.path.join(rife_job_input_dir, 'start.png')
    end_png_path = os.path.join(rife_job_input_dir, 'end.png')

    try:
        _print_info(f"Copying frame A: {frame_a_path} -> {start_png_path}")
        shutil.copy(frame_a_path, start_png_path)
        
        _print_info(f"Copying frame B: {frame_b_path} -> {end_png_path}")
        shutil.copy(frame_b_path, end_png_path)
        
        # Verify files were copied successfully
        if not os.path.isfile(start_png_path):
            _print_error(f"Failed to copy start.png to {start_png_path}")
            return None
        if not os.path.isfile(end_png_path):
            _print_error(f"Failed to copy end.png to {end_png_path}")
            return None
            
        start_size = os.path.getsize(start_png_path)
        end_size = os.path.getsize(end_png_path)
        _print_info(f"Successfully copied frames: start.png ({start_size} bytes), end.png ({end_size} bytes)")
        
    except Exception as e:
        _print_error(f"Failed to copy start/end frames for RIFE job {sanitized_job_name}: {e}")
        if os.path.exists(rife_job_input_dir): _safe_rmtree(rife_job_input_dir)
        return None

    cmd = [
        sys.executable, rife_inference_script_path,
        '--base_process_dir', rife_parent_work_dir, 
        '--exp_value', str(rife_exp),
        '--ext', 'png'
    ]
    _print_info(f"Running RIFE for job '{sanitized_job_name}': {' '.join(cmd)}")
    
    rife_script_dir = os.path.dirname(rife_inference_script_path)
    process = None # Define process here to ensure it's in scope for finally
    job_specific_name_for_logging = os.path.basename(rife_job_input_dir) # More specific than unique_job_name

    try:
        process = subprocess.Popen(cmd, cwd=rife_script_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
        
        _print_info(f"RIFE job '{job_specific_name_for_logging}' started (PID: {process.pid}). Waiting for output...")

        rife_timed_out = False
        stderr_output = []

        # Read stdout and stderr in a non-blocking way (or at least, line by line)
        while True:
            try:
                # Try to read a line from stdout with a short timeout
                # This makes the loop responsive if the process is slow to output
                line = process.stdout.readline()
                if line:
                    line = line.strip()
                    if line.startswith("PROGRESS:"):
                        _print_info(f"RIFE Update ({job_specific_name_for_logging}): {line}")
                    elif line.startswith("JOB_START:"):
                        _print_info(f"RIFE Log ({job_specific_name_for_logging}): {line}")
                    elif line.startswith("JOB_END:"):
                        _print_info(f"RIFE Log ({job_specific_name_for_logging}): {line}")
                    # else: # Optionally print other RIFE stdout lines
                    #    _print_info(f"RIFE STDOUT ({job_specific_name_for_logging}): {line}")
                
                # Check if process has ended
                if process.poll() is not None:
                    break # Process has finished

            except Exception as e: # Catch any exception during readline, could be due to process ending
                 _print_warning(f"Exception while reading RIFE stdout for {job_specific_name_for_logging}: {e}")
                 break

        # Wait for the process to complete fully and get return code and remaining stderr
        try:
            # The main timeout for the whole RIFE process
            process.wait(timeout=DEFAULT_RIFE_TIMEOUT) 
        except subprocess.TimeoutExpired:
            rife_timed_out = True
            _print_error(f"RIFE interpolation script timed out for job '{job_specific_name_for_logging}' after {DEFAULT_RIFE_TIMEOUT} seconds.")
            _print_error(f"Command was: {' '.join(cmd)}")
            process.kill() # Ensure the process is killed
            # Try to grab any final output
            _, final_err = process.communicate()
            stderr_output.append(final_err if final_err else "")

        # Collect all stderr after process completion or timeout
        if not rife_timed_out:
            for err_line in process.stderr:
                 stderr_output.append(err_line.strip())

        full_stderr = "\n".join(filter(None, stderr_output))

        if rife_timed_out:
            if full_stderr: _print_error(f"RIFE STDERR ({job_specific_name_for_logging}) on timeout:\n{full_stderr}")
            if os.path.exists(rife_job_input_dir): _safe_rmtree(rife_job_input_dir)
            return None

        if process.returncode != 0:
            _print_error(f"RIFE interpolation script failed for job '{job_specific_name_for_logging}' with exit code {process.returncode}.")
            _print_error(f"Command: {' '.join(cmd)}")
            if full_stderr: _print_error(f"RIFE STDERR ({job_specific_name_for_logging}):\n{full_stderr}")
            if os.path.exists(rife_job_input_dir): _safe_rmtree(rife_job_input_dir)
            return None
        
        _print_info(f"RIFE process completed successfully for job '{job_specific_name_for_logging}'.")
        if full_stderr: _print_warning(f"RIFE STDERR ({job_specific_name_for_logging}) (though process succeeded):\n{full_stderr}")

        # Add a small delay before checking output to ensure file operations are complete
        time.sleep(0.5)

        if not os.path.isdir(expected_rife_output_dir) or not os.listdir(expected_rife_output_dir):
            _print_error(f"RIFE interpolation for '{job_specific_name_for_logging}' seemed to succeed, but output dir '{expected_rife_output_dir}' is missing or empty.")
            if os.path.exists(rife_job_input_dir): _safe_rmtree(rife_job_input_dir)
            return None

        _print_info(f"RIFE output frames found in: {expected_rife_output_dir} for job '{job_specific_name_for_logging}'.")
        return expected_rife_output_dir

    except Exception as ex:
        _print_error(f"An unexpected error occurred running RIFE job '{job_specific_name_for_logging}': {ex}")
        if process and process.poll() is None: # If process is still running, kill it
            try:
                process.kill()
                process.wait(timeout=5) # Wait a bit for kill to take effect
            except Exception as e_kill:
                _print_warning(f"Failed to kill RIFE process for {job_specific_name_for_logging} after error: {e_kill}")
        if os.path.exists(rife_job_input_dir): _safe_rmtree(rife_job_input_dir)
        return None
    finally:
        # Ensure process is cleaned up if it was started and still running for some reason
        # (e.g. an exception not caught by the specific TimeoutExpired or CalledProcessError)
        if process and process.poll() is None:
            _print_warning(f"RIFE process for {job_specific_name_for_logging} (PID: {process.pid}) still running in finally block. Attempting to terminate.")
            try:
                process.terminate() # Try graceful termination first
                process.wait(timeout=10) # Wait for terminate
            except subprocess.TimeoutExpired:
                _print_warning(f"RIFE process for {job_specific_name_for_logging} did not terminate gracefully. Killing.")
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _print_error(f"Failed to confirm kill for RIFE process {job_specific_name_for_logging}.")
            except Exception as e_term:
                 _print_warning(f"Error during final RIFE process cleanup for {job_specific_name_for_logging}: {e_term}")

def _assemble_video_from_segments(segment_frame_dirs_and_specs, output_video_path, fps):
    """
    Assembles a video from multiple segments of frames. Output is SILENT.
    Args:
        segment_frame_dirs_and_specs (list of tuples):
            Each tuple: (frames_source_dir, include_spec, frame_prefix_in_source_dir)
            include_spec: "all", "all_except_first", "all_except_last", "all_except_first_and_last"
        output_video_path (str): Path for the final assembled video.
        fps (float): Frames per second for the output video.
    Returns:
        bool: True if successful, False otherwise.
    """
    with tempfile.TemporaryDirectory(prefix="video_assembly_") as assembly_frames_dir:
        _print_info(f"Assembling final SILENT video in temp dir: {assembly_frames_dir}")
        master_frame_idx = 1
        all_frames_copied_for_assembly = []

        for frames_source_dir, include_spec, frame_prefix in segment_frame_dirs_and_specs:
            if not os.path.isdir(frames_source_dir):
                _print_warning(f"Source directory for segment not found: {frames_source_dir}. Skipping segment.")
                continue

            # Ensure frame_prefix is not None and construct glob pattern carefully
            glob_pattern = os.path.join(frames_source_dir, f"{frame_prefix}*.png" if frame_prefix else "*.png")
            source_frame_files = sorted(glob.glob(glob_pattern))
            
            if not source_frame_files:
                _print_warning(f"No frames found in {frames_source_dir} with glob pattern '{glob_pattern}'. Skipping segment.")
                continue

            frames_to_copy_this_segment = []
            if include_spec == "all":
                frames_to_copy_this_segment = source_frame_files
            elif include_spec == "all_except_first":
                frames_to_copy_this_segment = source_frame_files[1:]
            elif include_spec == "all_except_last":
                frames_to_copy_this_segment = source_frame_files[:-1]
            elif include_spec == "all_except_first_and_last":
                frames_to_copy_this_segment = source_frame_files[1:-1]
            else:
                _print_warning(f"Unknown include_spec '{include_spec}'. Defaulting to 'all' for {frames_source_dir}")
                frames_to_copy_this_segment = source_frame_files
            
            if not frames_to_copy_this_segment:
                _print_info(f"Segment from {frames_source_dir} resulted in zero frames to copy with spec '{include_spec}'.")
                continue

            for src_frame_path in frames_to_copy_this_segment:
                dst_frame_name = f"frame_{master_frame_idx:06d}.png"
                dst_frame_path = os.path.join(assembly_frames_dir, dst_frame_name)
                shutil.copy(src_frame_path, dst_frame_path)
                all_frames_copied_for_assembly.append(dst_frame_path)
                master_frame_idx += 1
        
        if not all_frames_copied_for_assembly:
            _print_error("No frames were collected for the final video assembly.")
            return False

        _print_info(f"Collected {len(all_frames_copied_for_assembly)} frames for final video. Compiling...")
        
        # Updated FFmpeg command with explicit color space settings
        cmd_compile = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(assembly_frames_dir, 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-vf', 'format=yuv420p,colorspace=all=bt709:iall=bt709:fast=1',
            '-color_primaries', 'bt709',
            '-color_trc', 'bt709',
            '-colorspace', 'bt709',
            '-an',  # Ensure no audio
            output_video_path
        ]
        try:
            subprocess.run(cmd_compile, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=DEFAULT_FFMPEG_TIMEOUT)
            _print_info(f"Silent video successfully compiled: {output_video_path}")
            return os.path.exists(output_video_path)

        except subprocess.CalledProcessError as e:
            _print_error(f"Failed during FFmpeg video compilation.")
            _print_error(f"Command: {' '.join(e.cmd)}")
            if e.stdout: _print_error(f"FFmpeg STDOUT:\n{e.stdout}")
            if e.stderr: _print_error(f"FFmpeg STDERR:\n{e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            _print_error(f"FFmpeg timed out during video compilation for {output_video_path} after {DEFAULT_FFMPEG_TIMEOUT} seconds.")
            _print_error(f"Command was: {' '.join(cmd_compile)}")
            return False
        except Exception as ex_final:
             _print_error(f"Unexpected error during final video compilation: {ex_final}")
             return False
    return False

def _get_video_dimensions(video_path):
    """Get the dimensions (width, height) of a video file."""
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'csv=p=0:s=x', video_path
    ]
    try:
        output = subprocess.check_output(cmd, timeout=DEFAULT_FFPROBE_TIMEOUT).decode().strip()
        if 'x' in output:
            width, height = map(int, output.split('x'))
            return (width, height)
        return None
    except Exception as e:
        _print_warning(f"Could not get dimensions for {video_path}. Error: {e}")
        return None
    except subprocess.TimeoutExpired:
        _print_warning(f"ffprobe timed out while getting dimensions for {video_path} after {DEFAULT_FFPROBE_TIMEOUT} seconds.")
        return None

def _check_dimensions_compatibility(source_clips, neutral_video):
    """
    Check if all videos have compatible dimensions.
    Returns a list of incompatible videos.
    """
    neutral_dims = _get_video_dimensions(neutral_video)
    if not neutral_dims:
        _print_error(f"Could not determine dimensions of neutral video: {neutral_video}")
        return source_clips  # Assume all incompatible if we can't check

    incompatible_clips = []
    for clip in source_clips:
        clip_dims = _get_video_dimensions(clip)
        if not clip_dims or clip_dims != neutral_dims:
            incompatible_clips.append((clip, clip_dims))
    
    return incompatible_clips

def process_clip_with_neutral_loop(
    source_clip_path,
    # Path to the single, extracted neutral frame
    neutral_first_frame_path, 
    rife_inference_script_path,
    rife_exp,
    final_output_video_path,
    global_temp_dir,
    output_fps):

    _print_step(f"Processing Source Clip: {os.path.basename(source_clip_path)}")

    source_clip_fps = _get_video_fps(source_clip_path)
    _print_info(f"Source Clip FPS: {source_clip_fps:.2f}. Output will be {output_fps:.2f} FPS.")

    # Validate the passed neutral frame path
    if not os.path.isfile(neutral_first_frame_path):
        _print_error(f"Extracted neutral frame image not found: {neutral_first_frame_path}")
        return False

    # --- Prepare Source Clip Frames --- 
    _print_info(f"Preparing Source Clip frames for {os.path.basename(source_clip_path)}...")
    source_clip_basename = os.path.splitext(os.path.basename(source_clip_path))[0]
    source_clip_frames_dir = os.path.join(global_temp_dir, f"source_frames_{source_clip_basename}")
    source_clip_all_frames_list = _extract_frames_util(source_clip_path, source_clip_frames_dir, "sc_")
    if not source_clip_all_frames_list: 
        _print_error("Failed to extract source clip frames.")
        # Clean up frame dir if created but empty/failed
        if os.path.isdir(source_clip_frames_dir): _safe_rmtree(source_clip_frames_dir)
        return False
    source_first_frame_path = source_clip_all_frames_list[0]
    source_last_frame_path = source_clip_all_frames_list[-1]

    # --- Check Frame Dimensions --- 
    _print_info("Checking frame dimensions...")
    neutral_dims = _get_image_dimensions(neutral_first_frame_path)
    source_first_dims = _get_image_dimensions(source_first_frame_path)
    source_last_dims = _get_image_dimensions(source_last_frame_path)

    if not neutral_dims or not source_first_dims or not source_last_dims:
        _print_error("Could not read dimensions for all required frames. Skipping clip.")
        return False
        
    mismatch = False
    if neutral_dims != source_first_dims:
        _print_error(f"Dimension mismatch: Neutral frame ({neutral_dims[1]}x{neutral_dims[0]}) vs Source First frame ({source_first_dims[1]}x{source_first_dims[0]}).")
        mismatch = True
    if neutral_dims != source_last_dims:
        # Only print if different from the first frame mismatch message
        if source_first_dims != source_last_dims:
             _print_error(f"Dimension mismatch: Neutral frame ({neutral_dims[1]}x{neutral_dims[0]}) vs Source Last frame ({source_last_dims[1]}x{source_last_dims[0]}).")
        elif not mismatch: # Print only if first frame matched but last didn't
             _print_error(f"Dimension mismatch: Neutral frame ({neutral_dims[1]}x{neutral_dims[0]}) vs Source Last frame ({source_last_dims[1]}x{source_last_dims[0]}) (First frame matched).")
        mismatch = True

    if mismatch:
        _print_error("Source clip dimensions do not match the neutral frame. Skipping interpolation for this clip to avoid errors/artifacts.")
        return False
    else:
        _print_info(f"Frame dimensions match: {neutral_dims[1]}x{neutral_dims[0]}. Proceeding with interpolation.")
    # --- End Dimension Check --- 

    # --- Interpolation: SourceLast -> NeutralFirstFrame --- 
    _print_info(f"Interpolating: Source Clip Last -> Neutral Frame ({os.path.basename(neutral_first_frame_path)})")
    interp_s_to_n_job_name = f"S_{source_clip_basename}_to_N"
    interp_s_to_n_dir = _run_rife_interpolation(
        rife_inference_script_path, 
        source_last_frame_path, 
        neutral_first_frame_path, # Use the path to the extracted neutral frame
        rife_exp,
        global_temp_dir, 
        interp_s_to_n_job_name
    )
    if not interp_s_to_n_dir: 
        _print_error(f"Failed SourceLast->NeutralFrame interpolation for {os.path.basename(source_clip_path)}.")
        return False

    # --- Interpolation: NeutralFirstFrame -> SourceFirst ---
    _print_info(f"Interpolating: Neutral Frame ({os.path.basename(neutral_first_frame_path)}) -> Source Clip First")
    interp_n_to_s_job_name = f"N_to_S_{source_clip_basename}"
    interp_n_to_s_dir = _run_rife_interpolation(
        rife_inference_script_path, 
        neutral_first_frame_path, # Use the path to the extracted neutral frame
        source_first_frame_path, 
        rife_exp,
        global_temp_dir, 
        interp_n_to_s_job_name
    )
    if not interp_n_to_s_dir: 
        _print_error(f"Failed NeutralFrame->SourceFirst interpolation for {os.path.basename(source_clip_path)}.")
        return False

    # --- Assemble Single Output Video: Neutral->Source Interp + Source + Source->Neutral Interp --- 
    _print_info("Assembling Final Looped Video")
    output_video_name = f"{source_clip_basename}_looped_via_neutral.mp4"
    output_video_path = final_output_video_path
    
    segments_for_final_video = [
        # Starts with neutral frame, interpolates to source_first, EXCLUDES source_first copy
        (interp_n_to_s_dir, "all_except_last", "frame_"), 
        # Full source clip (includes source_first and source_last)
        (source_clip_frames_dir, "all", "sc_"),            
        # Interpolates from source_last to neutral, EXCLUDES source_last copy, ends with neutral frame
        (interp_s_to_n_dir, "all_except_first", "frame_") 
    ]
    
    if not _assemble_video_from_segments(segments_for_final_video, output_video_path, output_fps):
        _print_error(f"Failed to assemble final looped video: {output_video_name}")
        return False # Indicate failure for this clip
    else:
        _print_info(f"Successfully assembled: {output_video_name}")
        
        # Clean up temporary directories now that video assembly is complete
        _print_info("Cleaning up temporary directories...")
        
        # Clean up source frames directory
        if source_clip_frames_dir and os.path.exists(source_clip_frames_dir):
            try:
                _safe_rmtree(source_clip_frames_dir)
                _print_info(f"Cleaned up source frames directory: {os.path.basename(source_clip_frames_dir)}")
            except Exception as e:
                _print_warning(f"Failed to cleanup source frames directory {source_clip_frames_dir}: {e}")
        
        # Extract job directory paths from the interpolation output directories
        if interp_s_to_n_dir and os.path.exists(interp_s_to_n_dir):
            s_to_n_job_dir = os.path.dirname(interp_s_to_n_dir)  # Remove '/output' to get job dir
            if os.path.exists(s_to_n_job_dir):
                try:
                    _safe_rmtree(s_to_n_job_dir)
                    _print_info(f"Cleaned up Source->Neutral job directory: {os.path.basename(s_to_n_job_dir)}")
                except Exception as e:
                    _print_warning(f"Failed to cleanup {s_to_n_job_dir}: {e}")
        
        if interp_n_to_s_dir and os.path.exists(interp_n_to_s_dir):
            n_to_s_job_dir = os.path.dirname(interp_n_to_s_dir)  # Remove '/output' to get job dir
            if os.path.exists(n_to_s_job_dir):
                try:
                    _safe_rmtree(n_to_s_job_dir)
                    _print_info(f"Cleaned up Neutral->Source job directory: {os.path.basename(n_to_s_job_dir)}")
                except Exception as e:
                    _print_warning(f"Failed to cleanup {n_to_s_job_dir}: {e}")
        
        return True # Indicate success for this clip

def process_connect_two_videos(
    video1_path,
    video2_path,
    rife_inference_script_path,
    rife_exp,
    output_video_path,
    global_temp_dir,
    output_fps,
    transfer_audio=True,
    enable_preprocessing=False,
    rife_scale=1.0,
    quality_mode="balanced"):
    """
    Connect two videos with RIFE interpolation between them.
    Args:
        video1_path (str): Path to the first video
        video2_path (str): Path to the second video
        rife_inference_script_path (str): Path to RIFE inference script
        rife_exp (int): RIFE exponent for interpolation
        output_video_path (str): Path for the final output video
        global_temp_dir (str): Temporary directory for processing
        output_fps (float): Output video frame rate
        transfer_audio (bool): Whether to transfer audio from first video
        enable_preprocessing (bool): Enable frame preprocessing to reduce artifacts
        rife_scale (float): DEPRECATED - scale parameter not supported by RIFE
        quality_mode (str): Quality mode - "fast", "balanced", or "high_quality"
    Returns:
        bool: True if successful, False otherwise
    """
    _print_step(f"Connecting videos: {os.path.basename(video1_path)} -> {os.path.basename(video2_path)}")
    _print_info(f"Quality mode: {quality_mode}, Preprocessing: {enable_preprocessing}")
    
    # Note: rife_scale parameter is ignored as it's not supported by the actual RIFE implementation
    if rife_scale != 1.0:
        _print_warning(f"Scale parameter ({rife_scale}) is not supported by the RIFE implementation and will be ignored.")

    # Get video properties
    video1_fps = _get_video_fps(video1_path)
    video2_fps = _get_video_fps(video2_path)
    _print_info(f"Video 1 FPS: {video1_fps:.2f}, Video 2 FPS: {video2_fps:.2f}. Output will be {output_fps:.2f} FPS.")

    # Check dimension compatibility
    video1_dims = _get_video_dimensions(video1_path)
    video2_dims = _get_video_dimensions(video2_path)
    
    if not video1_dims or not video2_dims:
        _print_error("Could not determine video dimensions.")
        return False
        
    if video1_dims != video2_dims:
        _print_error(f"Video dimension mismatch: Video1 ({video1_dims[0]}x{video1_dims[1]}) vs Video2 ({video2_dims[0]}x{video2_dims[1]})")
        _print_error("Videos must have the same dimensions for proper connection.")
        return False
    
    _print_info(f"Video dimensions match: {video1_dims[0]}x{video1_dims[1]}. Proceeding with connection.")

    # Adjust extraction quality based on quality mode
    extraction_quality = 1  # Default
    if quality_mode == "high_quality":
        extraction_quality = 1  # Highest quality
    elif quality_mode == "balanced":
        extraction_quality = 2
    elif quality_mode == "fast":
        extraction_quality = 3

    # --- Extract frames from both videos ---
    _print_info("Extracting frames from Video 1...")
    video1_basename = os.path.splitext(os.path.basename(video1_path))[0]
    video1_frames_dir = os.path.join(global_temp_dir, f"video1_frames_{video1_basename}")
    video1_frames_list = _extract_frames_util(video1_path, video1_frames_dir, "v1_", quality=extraction_quality)
    if not video1_frames_list:
        _print_error("Failed to extract frames from Video 1.")
        return False

    _print_info("Extracting frames from Video 2...")
    video2_basename = os.path.splitext(os.path.basename(video2_path))[0]
    video2_frames_dir = os.path.join(global_temp_dir, f"video2_frames_{video2_basename}")
    video2_frames_list = _extract_frames_util(video2_path, video2_frames_dir, "v2_", quality=extraction_quality)
    if not video2_frames_list:
        _print_error("Failed to extract frames from Video 2.")
        return False

    # Get the connecting frames
    video1_last_frame = video1_frames_list[-1]
    video2_first_frame = video2_frames_list[0]
    
    # Apply preprocessing if enabled
    if enable_preprocessing:
        _print_info("Applying preprocessing to reduce artifacts...")
        video1_last_frame = _preprocess_frame_for_interpolation(
            video1_last_frame, global_temp_dir, "v1_last_processed.png", 
            enable_advanced_denoise=(quality_mode == "high_quality")
        )
        video2_first_frame = _preprocess_frame_for_interpolation(
            video2_first_frame, global_temp_dir, "v2_first_processed.png",
            enable_advanced_denoise=(quality_mode == "high_quality")
        )
    
    _print_info(f"Connecting from Video1 last frame to Video2 first frame...")

    # --- RIFE Interpolation between the connecting frames ---
    interp_job_name = f"connect_{video1_basename}_to_{video2_basename}"
    interp_frames_dir = _run_rife_interpolation(
        rife_inference_script_path,
        video1_last_frame,
        video2_first_frame,
        rife_exp,
        global_temp_dir,
        interp_job_name
    )
    if not interp_frames_dir:
        _print_error("Failed to generate interpolation frames between videos.")
        return False

    # --- Assemble the final connected video ---
    _print_info("Assembling final connected video...")
    
    segments_for_final_video = [
        # All frames from video 1 (including the last frame)
        (video1_frames_dir, "all", "v1_"),
        # Interpolated frames (excluding first frame to avoid duplication with video1's last frame,
        # and excluding last frame to avoid duplication with video2's first frame)
        (interp_frames_dir, "all_except_first_and_last", "frame_"),
        # All frames from video 2 (including the first frame)
        (video2_frames_dir, "all", "v2_")
    ]
    
    # Create a temporary silent video first
    temp_silent_output = os.path.join(global_temp_dir, "temp_connected_silent.mp4")
    
    if not _assemble_video_from_segments(segments_for_final_video, temp_silent_output, output_fps):
        _print_error("Failed to assemble connected video.")
        return False

    # Handle audio transfer if requested
    if transfer_audio:
        _print_info("Transferring audio from first video...")
        try:
            # Use ffmpeg to combine the silent video with audio from the first video
            cmd_audio = [
                'ffmpeg', '-y',
                '-i', temp_silent_output,  # Silent video
                '-i', video1_path,         # Source for audio
                '-c:v', 'copy',            # Copy video stream
                '-c:a', 'aac',             # Re-encode audio to ensure compatibility
                '-b:a', '160k',            # Audio bitrate
                '-map', '0:v:0',           # Video from first input
                '-map', '1:a:0',           # Audio from second input
                '-shortest',               # End when shortest stream ends
                output_video_path
            ]
            
            subprocess.run(cmd_audio, check=True, capture_output=True, text=True, 
                         encoding='utf-8', errors='ignore', timeout=DEFAULT_FFMPEG_TIMEOUT)
            _print_info("Successfully transferred audio from first video.")
            
        except subprocess.CalledProcessError as e:
            _print_warning("Failed to transfer audio. Saving video without audio.")
            _print_warning(f"Audio transfer error: {e.stderr}")
            # Fall back to silent video
            shutil.copy(temp_silent_output, output_video_path)
        except subprocess.TimeoutExpired:
            _print_warning("Audio transfer timed out. Saving video without audio.")
            shutil.copy(temp_silent_output, output_video_path)
    else:
        # Just copy the silent video as final output
        shutil.copy(temp_silent_output, output_video_path)

    if os.path.exists(output_video_path):
        _print_info(f"Successfully created connected video: {output_video_path}")
        return True
    else:
        _print_error("Failed to create final output video.")
        return False

def _advanced_denoise_frame(frame_path, temp_dir, output_name):
    """
    Apply advanced denoising specifically for reducing interpolation artifacts.
    Uses Non-local Means Denoising and additional filtering.
    """
    try:
        import cv2
        import numpy as np
        
        frame = cv2.imread(frame_path)
        if frame is None:
            _print_warning(f"Could not read frame for denoising: {frame_path}")
            return frame_path
        
        # Convert to float for better processing
        frame_float = frame.astype(np.float32) / 255.0
        
        # Apply Non-local Means Denoising
        # This is particularly good for preserving texture while removing noise
        denoised = cv2.fastNlMeansDenoisingColored(frame, None, 6, 6, 7, 21)
        
        # Apply edge-preserving filter
        # This helps maintain sharp edges while smoothing areas
        filtered = cv2.edgePreservingFilter(denoised, flags=2, sigma_s=50, sigma_r=0.4)
        
        # Subtle Gaussian blur to further reduce high-frequency artifacts
        blurred = cv2.GaussianBlur(filtered, (3, 3), 0.5)
        
        # Blend the filtered result with the original to maintain detail
        result = cv2.addWeighted(filtered, 0.7, blurred, 0.3, 0)
        
        # Save denoised frame
        denoised_path = os.path.join(temp_dir, output_name)
        cv2.imwrite(denoised_path, result)
        
        _print_info(f"Advanced denoised frame saved: {output_name}")
        return denoised_path
        
    except Exception as e:
        _print_warning(f"Advanced denoising failed: {e}. Using original frame.")
        return frame_path

def _preprocess_frame_for_interpolation(frame_path, temp_dir, output_name, enable_advanced_denoise=False):
    """
    Preprocess a frame to reduce interpolation artifacts.
    Applies noise reduction, sharpening, and color normalization.
    """
    try:
        import cv2
        import numpy as np
        
        # Read the frame
        frame = cv2.imread(frame_path)
        if frame is None:
            _print_warning(f"Could not read frame for preprocessing: {frame_path}")
            return frame_path
        
        # Apply advanced denoising first if requested
        if enable_advanced_denoise:
            frame_path_denoised = _advanced_denoise_frame(frame_path, temp_dir, f"denoised_{output_name}")
            frame = cv2.imread(frame_path_denoised)
        
        # Apply bilateral filter for noise reduction while preserving edges
        filtered = cv2.bilateralFilter(frame, 9, 75, 75)
        
        # Subtle sharpening to enhance details
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(filtered, -1, kernel * 0.1)
        
        # Blend original and sharpened
        result = cv2.addWeighted(filtered, 0.8, sharpened, 0.2, 0)
        
        # Color normalization for better consistency
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
        result = cv2.merge([l, a, b])
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        
        # For green screen content, apply additional chroma smoothing
        if _is_likely_green_screen(result):
            result = _smooth_green_screen_edges(result)
        
        # Save preprocessed frame
        processed_path = os.path.join(temp_dir, output_name)
        cv2.imwrite(processed_path, result)
        
        _print_info(f"Preprocessed frame saved: {output_name}")
        return processed_path
        
    except Exception as e:
        _print_warning(f"Frame preprocessing failed: {e}. Using original frame.")
        return frame_path

def _is_likely_green_screen(frame):
    """
    Detect if the frame likely contains green screen content.
    """
    try:
        import cv2
        import numpy as np
        
        # Convert to HSV for better green detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define green color range
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        
        # Create mask for green areas
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calculate percentage of green pixels
        green_ratio = np.sum(green_mask > 0) / (frame.shape[0] * frame.shape[1])
        
        # If more than 20% of the image is green, consider it green screen
        return green_ratio > 0.2
        
    except Exception:
        return False

def _smooth_green_screen_edges(frame):
    """
    Apply additional smoothing specifically for green screen edges to reduce artifacts.
    """
    try:
        import cv2
        import numpy as np
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create green mask
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Find edges of the green areas
        edges = cv2.Canny(green_mask, 50, 150)
        
        # Dilate edges to create a border region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edge_region = cv2.dilate(edges, kernel, iterations=2)
        
        # Apply additional smoothing only to edge regions
        smoothed = cv2.GaussianBlur(frame, (5, 5), 1.0)
        
        # Blend smoothed version only in edge regions
        edge_region_3ch = cv2.cvtColor(edge_region, cv2.COLOR_GRAY2BGR) / 255.0
        result = frame * (1 - edge_region_3ch) + smoothed * edge_region_3ch
        
        return result.astype(np.uint8)
        
    except Exception as e:
        _print_warning(f"Green screen edge smoothing failed: {e}")
        return frame

def main():
    parser = argparse.ArgumentParser(description="Advanced Video Looper with RIFE Transitions to a Neutral Frame.")
    
    # Add subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Original looper mode
    loop_parser = subparsers.add_parser('loop', help='Create looped videos with neutral frame transitions')
    loop_parser.add_argument("neutral_loop_video", help="Path to the neutral loop video file (first frame will be used).")
    loop_parser.add_argument("source_clips_folder", help="Path to the folder containing source video clips.")
    loop_parser.add_argument("output_folder", help="Path to the master output folder.")
    loop_parser.add_argument("--rife_exp", type=int, default=2, help="RIFE exponent for interpolation (default: 2). Higher means more frames.")
    loop_parser.add_argument("--workers", type=int, default=2, help="Number of worker threads for parallel processing (default: 2). Reduce if experiencing file access conflicts.")
    loop_parser.add_argument("--output_fps", type=float, default=30.0, help="Output video framerate (default: 30.0).")
    loop_parser.add_argument("--skip_incompatible", action="store_true", help="Skip clips with incompatible dimensions.")
    loop_parser.add_argument("--resume", action="store_true", help="Skip clips that already have processed outputs.")
    loop_parser.add_argument("--sequential", action="store_true", help="Process clips sequentially instead of in parallel (slower but avoids file conflicts).")
    
    # New connect mode
    connect_parser = subparsers.add_parser('connect', help='Connect two videos with RIFE interpolation')
    connect_parser.add_argument("video1", help="Path to the first video.")
    connect_parser.add_argument("video2", help="Path to the second video.")
    connect_parser.add_argument("output", help="Path for the connected output video.")
    connect_parser.add_argument("--rife_exp", type=int, default=2, help="RIFE exponent for interpolation (default: 2). Higher means more frames.")
    connect_parser.add_argument("--output_fps", type=float, default=30.0, help="Output video framerate (default: 30.0).")
    connect_parser.add_argument("--no_audio", action="store_true", help="Don't transfer audio from first video.")
    connect_parser.add_argument("--quality", choices=['fast', 'balanced', 'high_quality'], default='balanced', 
                              help="Quality mode: fast (lower quality, faster), balanced (default), high_quality (best quality, slower)")
    connect_parser.add_argument("--preprocess", action="store_true", 
                              help="Enable frame preprocessing to reduce artifacts (noise reduction, sharpening, color normalization)")
    connect_parser.add_argument("--denoise", action="store_true", 
                              help="Apply additional denoising before interpolation (slower but can reduce artifacts)")
    
    # Global options that apply to both modes
    parser.add_argument("--workers", type=int, default=2, help="Number of worker threads for parallel processing (default: 2). Reduce if experiencing file access conflicts.")
    parser.add_argument("--output_fps", type=float, default=30.0, help="Output video framerate (default: 30.0).")
    parser.add_argument("--skip_incompatible", action="store_true", help="Skip clips with incompatible dimensions.")
    parser.add_argument("--resume", action="store_true", help="Skip clips that already have processed outputs.")
    parser.add_argument("--sequential", action="store_true", help="Process clips sequentially instead of in parallel (slower but avoids file conflicts).")
    
    args = parser.parse_args()

    # Handle mode selection and backward compatibility
    if args.mode == 'connect':
        # Connect mode - validate inputs
        if not os.path.isfile(args.video1):
            _print_error(f"Video 1 not found: {args.video1}"); sys.exit(1)
        if not os.path.isfile(args.video2):
            _print_error(f"Video 2 not found: {args.video2}"); sys.exit(1)
            
        # Ensure output directory exists
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    elif args.mode == 'loop':
        # Loop mode - access subparser arguments
        neutral_video = args.neutral_loop_video
        source_folder = args.source_clips_folder  
        output_folder = args.output_folder
        
        if not os.path.isfile(neutral_video):
            _print_error(f"Neutral loop video not found: {neutral_video}"); sys.exit(1)
        if not os.path.isdir(source_folder):
            _print_error(f"Source clips folder not found: {source_folder}"); sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    # --- Determine and Validate RIFE Script Path --- 
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: # __file__ not defined if running in interactive env like IDLE
        script_dir = os.getcwd() 
        _print_warning("Could not determine script directory reliably, assuming script is run from its own directory.")

    # Construct the path to inference_img.py in the same directory
    rife_script_path = os.path.join(script_dir, 'inference_img.py')

    if not os.path.isfile(rife_script_path):
        _print_error(f"RIFE script not found at expected location: {rife_script_path}")
        _print_error("Please ensure this script ('advanced_video_looper.py') is located in the same directory as 'inference_img.py'.")
        _print_error(f"Script directory detected as: {script_dir}")
        sys.exit(1)
    _print_info(f"Using RIFE script found at: {rife_script_path}")
    # --- End RIFE Script Path Determination ---

    # Handle connect mode
    if args.mode == 'connect':
        with tempfile.TemporaryDirectory(prefix="video_connect_temp_") as temp_dir:
            _print_info(f"Temporary working directory: {temp_dir}")
            
            success = process_connect_two_videos(
                args.video1,
                args.video2,
                rife_script_path,
                args.rife_exp,
                args.output,
                temp_dir,
                args.output_fps,
                transfer_audio=not args.no_audio,
                enable_preprocessing=args.preprocess or args.denoise,
                rife_scale=1.0,  # Remove scale functionality since it doesn't exist
                quality_mode=args.quality
            )
            
            if success:
                _print_step("Video connection completed successfully!")
                sys.exit(0)
            else:
                _print_error("Video connection failed.")
                sys.exit(1)

    # Original loop mode logic continues below...
    # (rest of the original main function for loop mode)
    
    # Variables are already set in the mode selection above
    # neutral_video, source_folder, output_folder are set when args.mode == 'loop'

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        _print_info(f"Created output folder: {output_folder}")

    source_clip_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder)
                        if os.path.isfile(os.path.join(source_folder, f)) and \
                            f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))] 

    if not source_clip_files:
        _print_error(f"No source video clips found in {source_folder}"); sys.exit(1)

    _print_step(f"Found {len(source_clip_files)} source clips to process.")
    
    # Skip already processed clips if --resume is specified
    if args.resume:
        original_count = len(source_clip_files)
        already_processed = []
        
        for clip_path in source_clip_files[:]:  # Create a copy to iterate while modifying
            clip_basename = os.path.splitext(os.path.basename(clip_path))[0]
            output_path = os.path.join(output_folder, f"{clip_basename}_looped_via_neutral.mp4")
            
            if os.path.exists(output_path):
                already_processed.append(clip_path)
                source_clip_files.remove(clip_path)
        
        if already_processed:
            _print_info(f"Skipping {len(already_processed)} already processed clips:")
            for clip in already_processed[:5]:  # Show first 5
                _print_info(f"  - {os.path.basename(clip)}")
            if len(already_processed) > 5:
                _print_info(f"  - ... and {len(already_processed) - 5} more")
            _print_info(f"Proceeding with {len(source_clip_files)}/{original_count} clips that need processing.")
            
        if not source_clip_files:
            _print_info("All clips have already been processed. Nothing to do.")
            sys.exit(0)

    # Check video dimensions early to fail fast
    _print_step("Checking video dimensions compatibility...")
    incompatible_clips = _check_dimensions_compatibility(source_clip_files, neutral_video)
    
    if incompatible_clips:
        neutral_dims = _get_video_dimensions(neutral_video)
        _print_warning(f"Found {len(incompatible_clips)} clips with dimensions incompatible with neutral video ({neutral_dims[0]}x{neutral_dims[1]}):")
        for clip, dims in incompatible_clips:
            dim_str = f"{dims[0]}x{dims[1]}" if dims else "unknown"
            _print_warning(f"  - {os.path.basename(clip)} ({dim_str})")
        
        if args.skip_incompatible:
            _print_warning("These clips will be skipped during processing.")
            source_clip_files = [clip for clip in source_clip_files if clip not in [c[0] for c in incompatible_clips]]
            if not source_clip_files:
                _print_error("No compatible clips left to process."); sys.exit(1)
            _print_info(f"Proceeding with {len(source_clip_files)} compatible clips.")
        else:
            _print_error("Aborting due to dimension incompatibility. Use --skip_incompatible to skip these clips instead.")
            sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="adv_looper_main_temp_") as main_run_temp_dir:
        _print_info(f"Main temporary working directory for this run: {main_run_temp_dir}")
        
        # --- Extract ALL Frames of Neutral Loop (and get path to first) --- 
        _print_step("Extracting ALL frames from neutral loop video...")
        neutral_frame_temp_dir = os.path.join(main_run_temp_dir, "neutral_frame_extracted")
        # Extract ALL frames now, not just the first
        extracted_neutral_frame_list = _extract_frames_util(
            neutral_video, 
            neutral_frame_temp_dir, 
            frame_prefix="neutral_", 
            specific_frame_number=None # Extract all
        )
        if not extracted_neutral_frame_list:
            _print_error(f"Failed to extract frames from neutral loop video: {neutral_video}")
            sys.exit(1)
        
        # Get the path to the first extracted frame for interpolation use
        extracted_neutral_first_frame_path = extracted_neutral_frame_list[0] 
        _print_info(f"Neutral loop frames extracted to: {neutral_frame_temp_dir}")
        _print_info(f"Using neutral frame for interpolation: {extracted_neutral_first_frame_path}")
        # --- End Neutral Frame Extraction ---

        processed_count = 0
        
        # Choose processing mode based on sequential flag
        if args.sequential:
            _print_info("Processing clips sequentially to avoid file conflicts...")
            # Sequential processing
            for i, clip_path in enumerate(source_clip_files):
                clip_basename = os.path.splitext(os.path.basename(clip_path))[0]
                final_output_video_path = os.path.join(
                    output_folder,
                    f"{clip_basename}_looped_via_neutral.mp4"
                )
                
                os.makedirs(output_folder, exist_ok=True)
                
                _print_step(f"Processing clip {i+1}/{len(source_clip_files)}: {os.path.basename(clip_path)}")
                
                try:
                    success = process_clip_with_neutral_loop(
                        clip_path,
                        extracted_neutral_first_frame_path,
                        rife_script_path, 
                        args.rife_exp,
                        final_output_video_path, 
                        main_run_temp_dir,
                        output_fps=args.output_fps 
                    )
                    
                    if success:
                        processed_count += 1
                        _print_info(f"✓ Successfully processed: {os.path.basename(clip_path)}")
                    else:
                        _print_error(f"✗ Processing failed for {os.path.basename(clip_path)}")
                        
                except Exception as e:
                    _print_error(f"✗ Exception during processing of {os.path.basename(clip_path)}: {str(e)}")
        else:
            _print_info(f"Processing clips in parallel with {args.workers} workers...")
            # Parallel processing
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {}
                for clip_path in source_clip_files:
                    clip_basename = os.path.splitext(os.path.basename(clip_path))[0]
                    # Determine the final output path directly in the main output folder
                    final_output_video_path = os.path.join(
                        output_folder,
                        f"{clip_basename}_looped_via_neutral.mp4"
                    )
                    
                    # Create the main output folder if it somehow doesn't exist (should have been created earlier)
                    # No longer creating a subfolder per clip
                    os.makedirs(output_folder, exist_ok=True) 

                    # Pass the direct final path to the processing function
                    futures[clip_path] = executor.submit(process_clip_with_neutral_loop,
                        clip_path,
                        extracted_neutral_first_frame_path, # Pass the path to the extracted first frame
                        rife_script_path, 
                        args.rife_exp,
                        # Pass the direct output path, not a directory
                        final_output_video_path, 
                        main_run_temp_dir,
                        output_fps=args.output_fps 
                    )
                
                # Create a mapping from future to clip_path for as_completed
                future_to_clip = {future: clip_path for clip_path, future in futures.items()}
                
                # Use tqdm with as_completed for real-time progress
                for future in tqdm.tqdm(concurrent.futures.as_completed(futures.values()), 
                                     total=len(futures),
                                     desc="Processing clips"):
                    clip_path = future_to_clip[future]
                    try:
                        if future.result():
                            processed_count += 1
                        else:
                            _print_error(f"Processing failed for {os.path.basename(clip_path)}")
                    except Exception as e:
                        _print_error(f"Exception during processing of {os.path.basename(clip_path)}: {str(e)}")

        _print_step(f"Finished processing source clips. {processed_count}/{len(source_clip_files)} clips processed successfully.")

        # --- Process Neutral Loop for Color Consistency --- 
        _print_step("Processing Neutral Loop for Color Consistency...")
        neutral_loop_basename = os.path.splitext(os.path.basename(neutral_video))[0]
        processed_neutral_output_path = os.path.join(
            output_folder, 
            f"{neutral_loop_basename}_processed.mp4"
        )
        neutral_original_fps = _get_video_fps(neutral_video)

        _print_info(f"Re-assembling neutral loop from extracted frames at {neutral_original_fps:.2f} FPS...")
        neutral_assembly_segments = [
            (neutral_frame_temp_dir, "all", "neutral_") # Use all extracted neutral frames
        ]

        if not _assemble_video_from_segments(neutral_assembly_segments, processed_neutral_output_path, neutral_original_fps):
             _print_error(f"Failed to re-assemble neutral loop video: {processed_neutral_output_path}")
             # Don't necessarily exit, main processing might be done
        else:
             _print_info(f"Successfully processed neutral loop: {processed_neutral_output_path}")
        # --- End Neutral Loop Processing --- 

        _print_info(f"Outputs are in subdirectories within: {output_folder}")
        _print_info(f"Processed neutral loop saved as: {processed_neutral_output_path}")
        _print_info(f"Temporary directory {main_run_temp_dir} will now be cleaned up.")

if __name__ == "__main__":
    main() 