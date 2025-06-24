import subprocess
import shutil
from pathlib import Path
import os
from typing import List, Tuple

def run_ffmpeg_command(
    command: List[Path | str],
    operation_dir_to_clean: Path = None
) -> Tuple[bool, str]:
    """
    Runs an FFmpeg command silently (no banner, no stats, no logs).
    
    Args:
      command: full ffmpeg invocation, e.g.
               ['ffmpeg', '-i', in.mp4, ... , out.mp4]
      operation_dir_to_clean: if provided, this folder will be removed on failure.
      
    Returns:
      (success, message).  message is empty on success, or contains the error.
    """
    # ensure everything is string
    cmd = [str(c) for c in command]
    # inject silent flags right after 'ffmpeg'
    # ffmpeg -hide_banner -loglevel quiet -nostats ...
    cmd = cmd[:1] + ['-hide_banner', '-loglevel', 'quiet', '-nostats'] + cmd[1:]
    
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True, ""
    except subprocess.CalledProcessError as e:
        if operation_dir_to_clean and operation_dir_to_clean.exists():
            shutil.rmtree(operation_dir_to_clean)
        return False, f"FFmpeg exited with code {e.returncode}"
    except FileNotFoundError:
        if operation_dir_to_clean and operation_dir_to_clean.exists():
            shutil.rmtree(operation_dir_to_clean)
        return False, "FFmpeg not found in PATH"
    except Exception as e:
        if operation_dir_to_clean and operation_dir_to_clean.exists():
            shutil.rmtree(operation_dir_to_clean)
        return False, f"Unexpected error: {e}"
    
def transfer_audio(source_video_path: Path, target_video_path: Path, operation_dir: Path) -> tuple[bool, str]:
    """Transfers audio from source_video to target_video using FFmpeg."""
    temp_audio_file = operation_dir / "temp_audio_for_transfer.mkv"
    target_video_no_audio = operation_dir / "target_no_audio.mp4"

    # 1. Extract audio from source
    cmd_extract_audio = ['ffmpeg', '-y', '-i', source_video_path, '-c:a', 'copy', '-vn', temp_audio_file]
    success_extract, msg_extract = run_ffmpeg_command(cmd_extract_audio)
    if not success_extract:
        if temp_audio_file.exists(): temp_audio_file.unlink()
        return False, f"Audio extraction failed: {msg_extract}. Output video will have no audio."

    # Rename target to temp name
    try:
        if target_video_path.exists():
            shutil.move(str(target_video_path), str(target_video_no_audio))
        else:
            if temp_audio_file.exists(): temp_audio_file.unlink()
            return False, "Target video for audio merge not found."
    except Exception as e:
        if temp_audio_file.exists(): temp_audio_file.unlink()
        return False, f"Failed to rename target video for audio merge: {str(e)}"

    # 2. Merge audio
    cmd_merge_audio = ['ffmpeg', '-y', '-i', target_video_no_audio, '-i', temp_audio_file, '-c', 'copy', target_video_path]
    success_merge, msg_merge = run_ffmpeg_command(cmd_merge_audio)

    if not success_merge or target_video_path.stat().st_size == 0:
        print(f"Lossless audio transfer failed ({msg_merge}). Retrying with AAC transcode...")
        temp_audio_aac = operation_dir / "temp_audio.m4a"
        cmd_transcode = ['ffmpeg', '-y', '-i', source_video_path, '-c:a', 'aac', '-b:a', '160k', '-vn', temp_audio_aac]
        success_transcode, msg_transcode = run_ffmpeg_command(cmd_transcode)

        if success_transcode:
            cmd_merge_aac = ['ffmpeg', '-y', '-i', target_video_no_audio, '-i', temp_audio_aac, '-c', 'copy', target_video_path]
            success_merge_aac, msg_merge_aac = run_ffmpeg_command(cmd_merge_aac)
            if temp_audio_aac.exists(): temp_audio_aac.unlink()
            
            if success_merge_aac and target_video_path.stat().st_size > 0:
                if target_video_no_audio.exists(): target_video_no_audio.unlink()
                if temp_audio_file.exists(): temp_audio_file.unlink()
                return True, "Audio transferred with AAC transcode."
            else:
                shutil.move(str(target_video_no_audio), str(target_video_path))
                if temp_audio_file.exists(): temp_audio_file.unlink()
                return False, f"AAC audio merge also failed ({msg_merge_aac}). No audio."
        else:
            shutil.move(str(target_video_no_audio), str(target_video_path))
            if temp_audio_file.exists(): temp_audio_file.unlink()
            return False, f"Audio transcode to AAC failed ({msg_transcode}). No audio."
    else:
        if target_video_no_audio.exists(): target_video_no_audio.unlink()
        if temp_audio_file.exists(): temp_audio_file.unlink()
        return True, "Audio transferred successfully (lossless)."

def scale_and_pad_image(input_img_path: Path, target_w: int, target_h: int, output_img_path: Path) -> tuple[bool, str]:
    """Scales and pads an image to a target resolution using FFmpeg."""
    vf_filter = f"scale=w={target_w}:h={target_h}:force_original_aspect_ratio=1,pad=w={target_w}:h={target_h}:x=(ow-iw)/2:y=(oh-ih)/2:color=black"
    command = [
        'ffmpeg', '-y', '-i', input_img_path,
        '-vf', vf_filter,
        '-pix_fmt', 'rgb24',
        output_img_path
    ]
    return run_ffmpeg_command(command) 