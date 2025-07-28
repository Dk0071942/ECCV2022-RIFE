#!/usr/bin/env python3
"""
Test script to demonstrate FPS detection and conversion.
"""

import cv2

def detect_video_fps(video_path):
    """
    Detect the actual FPS of the input video.
    Returns the detected FPS or 25.0 as fallback.
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("⚠️ Could not open video for FPS detection, using 25 FPS")
            return 25.0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if fps <= 0 or fps > 120:  # Sanity check
            print(f"⚠️ Invalid FPS detected: {fps}, using 25 FPS")
            return 25.0
        
        print(f"🎯 Detected FPS: {fps}")
        print(f"📊 Total frames: {frame_count}")
        print(f"⏱️ Duration: {frame_count/fps:.2f} seconds")
        return fps
    except Exception as e:
        print(f"⚠️ Error detecting FPS: {e}, using 25 FPS")
        return 25.0

# Test with available videos
test_videos = ["1.mp4", "2.mp4"]

for video in test_videos:
    print(f"\n📹 Testing {video}:")
    fps = detect_video_fps(video)
    print(f"✅ Result: {fps} FPS")