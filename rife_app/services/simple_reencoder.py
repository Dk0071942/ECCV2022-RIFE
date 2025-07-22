"""
Simple video re-encoding service with encoding validation.
Checks current encoding first, only re-encodes if needed.
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path
import sys

# Add path for video analyzer
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from rife_app.utils.video_analyzer import VideoAnalyzer


class SimpleVideoReencoder:
    """Video re-encoder with smart encoding validation."""
    
    def __init__(self):
        self.analyzer = VideoAnalyzer()
    
    def reencode_video(self, input_video_path):
        """
        Re-encode a video using standardized FFmpeg parameters.
        Checks current encoding first - skips if already meets standards.
        Returns (output_path, status_message).
        """
        try:
            # DETAILED INPUT LOGGING FOR DEBUGGING
            print(f"🔍 INPUT DEBUG - Raw input type: {type(input_video_path)}")
            print(f"🔍 INPUT DEBUG - Raw input value: {repr(input_video_path)}")
            
            # Handle input path
            if not input_video_path:
                print("❌ INPUT DEBUG - No video file provided (None or empty)")
                return None, "No video file provided"
            
            # Log all attributes of the input object for debugging
            if hasattr(input_video_path, '__dict__'):
                print(f"🔍 INPUT DEBUG - Object attributes: {vars(input_video_path)}")
            else:
                print(f"🔍 INPUT DEBUG - Object dir: {[attr for attr in dir(input_video_path) if not attr.startswith('_')]}")
            
            # Convert gradio file object to path if needed
            original_input = input_video_path
            if hasattr(input_video_path, 'name'):
                print(f"🔍 INPUT DEBUG - Found .name attribute: {input_video_path.name}")
                input_video_path = input_video_path.name
            elif hasattr(input_video_path, 'path'):
                print(f"🔍 INPUT DEBUG - Found .path attribute: {input_video_path.path}")
                input_video_path = input_video_path.path
            else:
                print(f"🔍 INPUT DEBUG - No .name or .path attribute, using as-is")
            
            input_path = str(input_video_path)
            print(f"🔍 INPUT DEBUG - Final path: {input_path}")
            print(f"🔍 INPUT DEBUG - Path exists: {os.path.exists(input_path)}")
            
            if not os.path.exists(input_path):
                print(f"❌ INPUT DEBUG - File not found at: {input_path}")
                print(f"🔍 INPUT DEBUG - Current working directory: {os.getcwd()}")
                print(f"🔍 INPUT DEBUG - Trying alternative path extraction...")
                
                # Try alternative ways to extract path
                if hasattr(original_input, 'file'):
                    print(f"🔍 INPUT DEBUG - Found .file attribute: {original_input.file}")
                if hasattr(original_input, 'url'):
                    print(f"🔍 INPUT DEBUG - Found .url attribute: {original_input.url}")
                if hasattr(original_input, 'value'):
                    print(f"🔍 INPUT DEBUG - Found .value attribute: {original_input.value}")
                
                return None, f"Video file not found: {input_path}\nOriginal input type: {type(original_input)}\nOriginal input: {repr(original_input)}"
            
            # STEP 1: Analyze current encoding (if tools available)
            print("🔍 Analyzing current video encoding...")
            meets_standards, analysis_report, video_params = self.analyzer.analyze_video(input_path)
            
            # If video already meets standards, return original file
            if meets_standards:
                return input_path, f"✅ Video already meets encoding standards!\n\n{analysis_report}\n\n⚡ No re-encoding needed - original file returned."
            
            # If analysis failed due to missing tools, proceed with re-encoding
            if analysis_report.startswith("❌ Failed to analyze"):
                print("⚠️ Video analysis unavailable - proceeding with re-encoding")
                analysis_report = "⚠️ Video analysis unavailable (ffprobe not found)\nProceeding with re-encoding to ensure standards compliance."
            
            # Create output directory and filename
            output_dir = Path("./temp_gradio/reencoded_videos")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            input_file = Path(input_path)
            output_filename = f"{input_file.stem}_reencoded{input_file.suffix}"
            output_path = output_dir / output_filename
            
            # Build simple FFmpeg command with standardized parameters
            # Note: The colorspace filter should include transfer characteristics
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p',
                '-vf', 'format=yuv420p,colorspace=all=bt709:iall=bt709:itrc=bt709:fast=1',
                '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
                '-movflags', '+faststart',
                '-c:a', 'aac', '-b:a', '192k', '-ar', '16000',
                str(output_path)
            ]
            
            # STEP 2: Re-encode the video
            print("🔄 Re-encoding video to meet standards...")
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            end_time = time.time()
            
            if result.returncode == 0 and output_path.exists():
                processing_time = end_time - start_time
                success_msg = (
                    f"✅ Video re-encoded successfully in {processing_time:.1f}s\n\n"
                    f"📊 ORIGINAL VIDEO ANALYSIS:\n{analysis_report}\n\n"
                    f"🎯 Applied standard encoding parameters:\n"
                    f"• H.264 codec with CRF 18\n"
                    f"• BT.709 color space\n"
                    f"• AAC audio at 192k\n"
                    f"• Web optimized"
                )
                return str(output_path), success_msg
            else:
                error_msg = result.stderr if result.stderr else "Unknown FFmpeg error"
                return None, f"❌ Re-encoding failed: {error_msg}\n\nOriginal analysis:\n{analysis_report}"
                
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def get_info(self):
        """Get encoding information."""
        return """🎥 Smart Video Re-encoding

🔍 INTELLIGENT ANALYSIS:
• Checks current encoding first
• Skips re-encoding if already optimal
• Only processes when needed

🎯 STANDARD PARAMETERS:
• Codec: H.264 (libx264) with CRF 18
• Preset: slow (high quality)
• Color: BT.709 (HD standard)
• Audio: AAC 192k at 16kHz
• Web optimized with faststart

⚡ EFFICIENCY:
• Saves time on compliant videos
• Detailed analysis reports
• No unnecessary processing"""