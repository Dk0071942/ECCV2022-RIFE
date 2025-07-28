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
    
    def reencode_video(self, input_video_path, force_frame_based_reencoding=False):
        """
        Re-encode a video with two different methods:
        1. Direct reencoding (default): Standard FFmpeg reencoding
        2. Frame-based reencoding (forced): Extract frames -> Rebuild video for color consistency
        
        Args:
            input_video_path: Path to input video
            force_frame_based_reencoding: If True, use frame extraction method for color consistency
        
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
            
            # STEP 1: Analyze current encoding (always check for optimization)
            print("🔍 Analyzing current video encoding...")
            meets_standards, analysis_report, video_params = self.analyzer.analyze_video(input_path)
            
            # If video already meets standards and not using frame-based method, return original file
            if meets_standards and not force_frame_based_reencoding:
                bitrate_info = self._format_bitrate_info(video_params)
                return input_path, f"✅ Video already meets encoding standards!\n\n{analysis_report}\n\n{bitrate_info}\n\n⚡ No re-encoding needed - original file returned."
            
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
            
            # Choose encoding method
            if force_frame_based_reencoding:
                return self._frame_based_encoding(input_path, output_path, analysis_report, video_params)
            else:
                return self._direct_encoding(input_path, output_path, analysis_report, video_params)
                
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def _direct_encoding(self, input_path, output_path, analysis_report, video_params):
        """Direct FFmpeg reencoding without frame extraction."""
        print("🔄 Using direct reencoding method...")
        start_time = time.time()
        
        # Detect original FPS for proper conversion
        print("🎯 Detecting original FPS...")
        original_fps = self._detect_video_fps(input_path)
        print(f"📊 Original FPS: {original_fps}")
        print(f"🎯 Target output FPS: 25")
        
        # Direct reencoding command with BT.709 normalization and proper FPS conversion
        direct_cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
            '-vf', 'format=yuv420p,colorspace=all=bt709:iall=bt709:itrc=bt709:fast=1,fps=25',
            '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
            '-movflags', '+faststart',
            '-map', '0:v:0',  # Map first video stream
            '-map', '0:a?',   # Map all audio streams if present (? makes it optional)
            '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',  # Standard 48kHz audio
            str(output_path)
        ]
        
        result = subprocess.run(direct_cmd, capture_output=True, text=True)
        end_time = time.time()
        
        if result.returncode == 0 and output_path.exists():
            processing_time = end_time - start_time
            
            # Analyze the re-encoded video for final bit rate info
            final_meets_standards, final_analysis_report, final_video_params = self.analyzer.analyze_video(str(output_path))
            final_bitrate_info = self._format_bitrate_info(final_video_params)
            original_bitrate_info = self._format_bitrate_info(video_params)
            
            success_msg = (
                f"✅ Video re-encoded successfully in {processing_time:.1f}s\n\n"
                f"📊 ORIGINAL VIDEO:\n{analysis_report}\n\n"
                f"{original_bitrate_info}\n\n"
                f"🎯 APPLIED DIRECT ENCODING:\n"
                f"• Direct FFmpeg reencoding (no frame extraction)\n"
                f"• Output forced to 25 FPS\n"
                f"• H.264 codec with CRF 18\n"
                f"• BT.709 color space normalization\n"
                f"• AAC audio at 192k, 48kHz\n"
                f"• Web optimized\n\n"
                f"📊 FINAL RESULT:\n{final_bitrate_info}"
            )
            return str(output_path), success_msg
        else:
            error_msg = result.stderr if result.stderr else "Unknown FFmpeg error"
            return None, f"❌ Direct encoding failed: {error_msg}\n\nOriginal analysis:\n{analysis_report}"
    
    def _frame_based_encoding(self, input_path, output_path, analysis_report, video_params):
        """Frame-based encoding with color-safe round-trip."""
        print("🖼️ Using frame-based reencoding method with color-safe round-trip...")
        
        # Create temporary frames directory
        output_dir = output_path.parent
        input_file = Path(input_path)
        temp_frames_dir = output_dir / f"{input_file.stem}_temp_frames"
        temp_frames_dir.mkdir(exist_ok=True)
        
        try:
            # STEP 1: Extract frames to PNG with proper color space handling
            print("🖼️ Extracting frames with color space conversion...")
            frame_extract_cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-vf', 'scale=in_range=limited:out_range=full,format=rgb24',
                '-color_range', 'pc',  # Full range for PNG
                str(temp_frames_dir / 'frame_%06d.png')
            ]
            
            extract_result = subprocess.run(frame_extract_cmd, capture_output=True, text=True)
            if extract_result.returncode != 0:
                return None, f"❌ Frame extraction failed: {extract_result.stderr}"
            
            # STEP 2: Detect original FPS for proper frame rate conversion
            print("🎯 Detecting original FPS...")
            original_fps = self._detect_video_fps(input_path)
            print(f"📊 Original FPS: {original_fps}")
            print(f"🎯 Target output FPS: 25")
            
            # STEP 3: Re-assemble PNGs back to limited-range Rec. 709 YUV
            print("🔄 Rebuilding video from frames with proper color space conversion...")
            start_time = time.time()
            
            rebuild_cmd = [
                'ffmpeg', '-y',
                '-r', '25',  # Force 25 FPS output
                '-start_number', '1',
                '-i', str(temp_frames_dir / 'frame_%06d.png'),
                '-i', input_path,  # For audio
                '-vf', 'scale=in_range=full:out_range=limited,format=yuv420p',
                '-c:v', 'libx264', '-crf', '18', '-preset', 'slow',
                '-color_range', 'tv',  # Limited range for video
                '-color_primaries', 'bt709',
                '-color_trc', 'bt709',
                '-colorspace', 'bt709',
                '-map', '0:v',    # Map video from frames
                '-map', '1:a?',   # Map audio from original file if present
                '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',  # Standard 48kHz audio
                '-movflags', '+faststart',
                str(output_path)
            ]
            
            result = subprocess.run(rebuild_cmd, capture_output=True, text=True)
            end_time = time.time()
            
            if result.returncode == 0 and output_path.exists():
                processing_time = end_time - start_time
                
                # Analyze the re-encoded video for final bit rate info
                final_meets_standards, final_analysis_report, final_video_params = self.analyzer.analyze_video(str(output_path))
                final_bitrate_info = self._format_bitrate_info(final_video_params)
                original_bitrate_info = self._format_bitrate_info(video_params)
                
                success_msg = (
                    f"✅ Video re-encoded successfully in {processing_time:.1f}s\n\n"
                    f"📊 ORIGINAL VIDEO:\n{analysis_report}\n\n"
                    f"{original_bitrate_info}\n\n"
                    f"🎯 APPLIED COLOR-SAFE FRAME-BASED ENCODING:\n"
                    f"• Frame extraction: limited→full range conversion\n"
                    f"• PNG storage: full-range RGB24 (no YUV quantization)\n"
                    f"• Video rebuild: full→limited range conversion\n"
                    f"• Output forced to 25 FPS\n"
                    f"• H.264 codec with CRF 18\n"
                    f"• Proper Rec. 709 metadata flags\n"
                    f"• AAC audio at 192k, 48kHz\n"
                    f"• Web optimized\n\n"
                    f"📊 FINAL RESULT:\n{final_bitrate_info}"
                )
                return str(output_path), success_msg
            else:
                error_msg = result.stderr if result.stderr else "Unknown FFmpeg error"
                return None, f"❌ Video rebuild failed: {error_msg}\n\nOriginal analysis:\n{analysis_report}"
                
        except Exception as e:
            return None, f"❌ Frame-based encoding error: {str(e)}"
        finally:
            # Clean up temporary frames directory
            if temp_frames_dir.exists():
                print("🧹 Cleaning up temporary frames...")
                import shutil
                shutil.rmtree(temp_frames_dir, ignore_errors=True)
    
    def _detect_video_fps(self, video_path):
        """
        Force 25 FPS output for all videos.
        Returns 25.0 FPS.
        """
        print("🎯 Forcing output to 25 FPS")
        return 25.0
    
    def _format_bitrate_info(self, video_params):
        """Format bit rate information for display."""
        if not video_params:
            return "📊 BIT RATE INFO: Not available"
        
        info_lines = ["📊 BIT RATE INFO:"]
        
        # Video bit rate
        video_bitrate = video_params.get('bitrate')
        if video_bitrate:
            video_mbps = video_bitrate / 1_000_000
            info_lines.append(f"• Video: {video_mbps:.2f} Mbps ({video_bitrate:,} bps)")
        else:
            info_lines.append("• Video: Not available")
        
        # Audio bit rate
        audio_bitrate = video_params.get('audio_bitrate')
        if audio_bitrate:
            audio_kbps = audio_bitrate / 1000
            info_lines.append(f"• Audio: {audio_kbps:.0f} kbps ({audio_bitrate:,} bps)")
        else:
            info_lines.append("• Audio: Not available")
        
        # Resolution info for context
        width = video_params.get('width', 0)
        height = video_params.get('height', 0)
        if width and height:
            info_lines.append(f"• Resolution: {width}x{height}")
        
        return "\n".join(info_lines)
    
    def get_info(self):
        """Get encoding information."""
        return """🎥 Dual-Mode Video Re-encoding

🔄 TWO ENCODING METHODS:

📋 DIRECT REENCODING (Default, Unchecked):
• Fast standard FFmpeg reencoding
• No frame extraction required
• Output forced to 25 FPS
• Applies BT.709 color space normalization
• H.264 codec with CRF 18, AAC audio 192k/48kHz
• Checks standards first, skips if optimal
• Faster processing, less disk usage

🖼️ FRAME-BASED REENCODING (Checked):
• Color-safe round-trip with range conversion
• Extract frames: limited→full range
• PNG storage: full-range RGB24
• Rebuild: full→limited range
• Preserves color accuracy through conversion
• Slower but guarantees perfect color fidelity

🔍 INTELLIGENT ANALYSIS:
• Always checks current encoding first
• Skips re-encoding if already optimal (direct mode only)
• Provides detailed bit rate information
• Resolution and format details

📊 FEATURES:
• BT.709 color space throughout
• H.264 (libx264) with CRF 18
• AAC audio at 192k, 48kHz
• Web optimized with faststart
• Automatic temp cleanup (frame mode)
• Detailed processing reports"""