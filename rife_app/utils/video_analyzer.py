"""
Video analyzer to check encoding parameters and validate against standards.
"""

import json
import subprocess
from typing import Dict, Tuple, Optional


class VideoAnalyzer:
    """Analyzes video files to determine their encoding parameters."""
    
    # Our standard encoding parameters for comparison
    STANDARD_PARAMS = {
        'codec': 'h264',
        'crf_range': (16, 20),  # CRF 18 ± 2 for flexibility
        'pixel_format': 'yuv420p',
        'color_primaries': 'bt709',
        'color_trc': 'bt709',
        'colorspace': 'bt709',
        'audio_codec': 'aac',
        'audio_sample_rate_range': (15000, 17000),  # 16kHz ± 1kHz
        'audio_bitrate_range': (180, 220)  # 192k ± 12k
    }
    
    def get_video_info(self, video_path: str) -> Optional[Dict]:
        """
        Get detailed video information using ffprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information or None if failed
        """
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', 
                '-show_streams', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                print(f"ffprobe failed: {result.stderr}")
                return None
                
        except FileNotFoundError:
            print("ffprobe not found - video analysis disabled")
            return None
        except Exception as e:
            print(f"Error running ffprobe: {e}")
            return None
    
    def extract_video_params(self, video_info: Dict) -> Optional[Dict]:
        """
        Extract relevant video encoding parameters from ffprobe output.
        
        Args:
            video_info: Output from ffprobe
            
        Returns:
            Dictionary with extracted parameters
        """
        try:
            video_stream = None
            audio_stream = None
            
            # Find video and audio streams
            for stream in video_info.get('streams', []):
                if stream.get('codec_type') == 'video' and video_stream is None:
                    video_stream = stream
                elif stream.get('codec_type') == 'audio' and audio_stream is None:
                    audio_stream = stream
            
            if not video_stream:
                return None
            
            # Extract color transfer characteristics - ffprobe may use different field names
            color_trc = video_stream.get('color_trc', '')
            if not color_trc:
                color_trc = video_stream.get('color_transfer', '')
            if not color_trc:
                color_trc = video_stream.get('transfer_characteristics', '')
            
            params = {
                'video_codec': video_stream.get('codec_name', '').lower(),
                'pixel_format': video_stream.get('pix_fmt', ''),
                'color_primaries': video_stream.get('color_primaries', ''),
                'color_trc': color_trc,
                'colorspace': video_stream.get('color_space', ''),
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'bitrate': int(video_stream.get('bit_rate', 0)) if video_stream.get('bit_rate') else None,
            }
            
            # Add audio parameters if available
            if audio_stream:
                params.update({
                    'audio_codec': audio_stream.get('codec_name', '').lower(),
                    'audio_sample_rate': int(audio_stream.get('sample_rate', 0)) if audio_stream.get('sample_rate') else None,
                    'audio_bitrate': int(audio_stream.get('bit_rate', 0)) if audio_stream.get('bit_rate') else None,
                })
            
            return params
            
        except Exception as e:
            print(f"Error extracting video parameters: {e}")
            return None
    
    def check_meets_standards(self, video_params: Dict) -> Tuple[bool, str]:
        """
        Check if video parameters meet our encoding standards.
        
        Args:
            video_params: Extracted video parameters
            
        Returns:
            Tuple of (meets_standards, detailed_report)
        """
        checks = []
        meets_standard = True
        
        # Check video codec
        if video_params.get('video_codec') == self.STANDARD_PARAMS['codec']:
            checks.append("✅ Video codec: H.264")
        else:
            checks.append(f"❌ Video codec: {video_params.get('video_codec')} (expected H.264)")
            meets_standard = False
        
        # Check pixel format
        if video_params.get('pixel_format') == self.STANDARD_PARAMS['pixel_format']:
            checks.append("✅ Pixel format: yuv420p")
        else:
            checks.append(f"❌ Pixel format: {video_params.get('pixel_format')} (expected yuv420p)")
            meets_standard = False
        
        # Check color space parameters
        color_params = ['color_primaries', 'color_trc', 'colorspace']
        for param in color_params:
            expected = self.STANDARD_PARAMS[param]
            actual = video_params.get(param, '')
            if actual.lower() == expected:
                checks.append(f"✅ {param.replace('_', ' ').title()}: {expected}")
            else:
                checks.append(f"❌ {param.replace('_', ' ').title()}: {actual} (expected {expected})")
                meets_standard = False
        
        # Check audio codec if available
        audio_codec = video_params.get('audio_codec')
        if audio_codec:
            if audio_codec == self.STANDARD_PARAMS['audio_codec']:
                checks.append("✅ Audio codec: AAC")
            else:
                checks.append(f"❌ Audio codec: {audio_codec} (expected AAC)")
                meets_standard = False
            
            # Check audio sample rate
            audio_sr = video_params.get('audio_sample_rate')
            if audio_sr:
                sr_min, sr_max = self.STANDARD_PARAMS['audio_sample_rate_range']
                if sr_min <= audio_sr <= sr_max:
                    checks.append(f"✅ Audio sample rate: {audio_sr}Hz")
                else:
                    checks.append(f"❌ Audio sample rate: {audio_sr}Hz (expected ~16000Hz)")
                    meets_standard = False
        else:
            checks.append("⚠️ No audio stream found")
        
        report = "\n".join(checks)
        return meets_standard, report
    
    def analyze_video(self, video_path: str) -> Tuple[bool, str, Optional[Dict]]:
        """
        Complete video analysis - get info, extract params, and check standards.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (meets_standards, report, video_params)
        """
        # Get video information
        video_info = self.get_video_info(video_path)
        if not video_info:
            return False, "❌ Failed to analyze video file", None
        
        # Extract parameters
        video_params = self.extract_video_params(video_info)
        if not video_params:
            return False, "❌ Failed to extract video parameters", None
        
        # Check against standards
        meets_standards, report = self.check_meets_standards(video_params)
        
        # Add summary header
        status = "✅ MEETS STANDARDS" if meets_standards else "❌ NEEDS RE-ENCODING"
        full_report = f"📊 VIDEO ANALYSIS REPORT\n{status}\n\n{report}"
        
        return meets_standards, full_report, video_params