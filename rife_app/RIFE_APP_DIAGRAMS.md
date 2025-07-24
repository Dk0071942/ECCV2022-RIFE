# RIFE Application Tab Diagrams

This document contains Mermaid diagrams for each tab in the RIFE application (`app.py`).

## Tab 1: Select Frames from Video

```mermaid
graph TD
    A[Tab 1: Select Frames from Video] --> B[Video Upload]
    B --> C{Video Uploaded?}
    C -->|Yes| D[get_video_info]
    C -->|No| E[Display: Video not loaded]
    D --> F[Display Video Info]
    D --> G[Enable Frame Selection]
    G --> H[Start Frame Input]
    G --> I[End Frame Input]
    H --> J[Extract Frames Button]
    I --> J
    J --> K[handle_frame_extraction]
    K --> L{Valid Frame Range?}
    L -->|No| M[Error: Invalid frames]
    L -->|Yes| N[extract_frames]
    N --> O[Display Start Frame]
    N --> P[Display End Frame]
    N --> Q[Auto-populate Tab 2]
    Q --> R[End Frame → First Image]
    Q --> S[Start Frame → Second Image]
```

## Tab 2: Interpolate Between Images

```mermaid
graph TD
    A[Tab 2: Interpolate Between Images] --> B[Image Input Section]
    B --> C[First Image/Source]
    B --> D[Second Image/Target]
    
    E[Configuration] --> F[Number of Passes: 1-6]
    E --> G[Interpolation Method]
    
    G --> H{Method Selection}
    H -->|Standard| I[Recursive Mode]
    H -->|Disk-Based| J[Best Quality Mode]
    
    F --> K[Memory Estimation]
    C --> K
    D --> K
    
    L[Generate Button] --> M{Images Loaded?}
    M -->|No| N[Error: Upload images]
    M -->|Yes| O[ImageInterpolator.interpolate]
    
    O --> P{Mode Check}
    P -->|Disk-Based| Q[Constant Memory Usage]
    P -->|Standard| R[Recursive Processing]
    
    Q --> S[Store frames on disk]
    R --> T[Process in memory]
    
    S --> U[Generate Video @ 25 FPS]
    T --> U
    
    U --> V[Output Video]
    U --> W[Status Message]
```

## Tab 3: Chained Video Interpolation

```mermaid
graph TD
    A[Tab 3: Chained Video Interpolation] --> B[Video Inputs]
    B --> C[Anchor Video/Start]
    B --> D[Middle Video]
    B --> E[End Video]
    
    F[Configuration] --> G[Number of Passes]
    F --> H[Final FPS]
    F --> I[Interpolation Method]
    
    I --> J{Method Choice}
    J -->|Standard| K[image_interpolation]
    J -->|Disk-Based| L[disk_based]
    
    M[Generate Button] --> N[ChainedInterpolator.interpolate]
    
    N --> O[Extract transition frames]
    O --> P[Anchor last → Middle first]
    O --> Q[Middle last → End first]
    
    P --> R[Generate transition 1]
    Q --> S[Generate transition 2]
    
    R --> T[Merge Videos]
    S --> T
    T --> U[Anchor + Trans1 + Middle + Trans2 + End]
    
    U --> V[Output Chained Video]
    U --> W[Status Message]
```

## Tab 4: Video Interpolation

```mermaid
graph TD
    A[Tab 4: Video Interpolation] --> B[Video Upload]
    B --> C[Number of Passes: 1-4]
    
    C --> D[Pass Information]
    D --> E[1 pass = 2x FPS]
    D --> F[2 passes = 4x FPS]
    D --> G[3 passes = 8x FPS]
    D --> H[4 passes = 16x FPS]
    
    I[Interpolate Button] --> J[handle_advanced_video_interpolation]
    
    J --> K[Initialize output directory]
    K --> L[Loop through passes]
    
    L --> M{For each pass}
    M --> N[main_interpolate with exp=1]
    N --> O[2x frame rate increase]
    O --> P[Use output as next input]
    P -->|More passes?| M
    P -->|Done| Q[Final multiplier = 2^passes]
    
    Q --> R[Output Video]
    Q --> S[Status: X passes → Yx frame rate]
    
    T[Key Features] --> U[Maintains video duration]
    T --> V[Increases frame rate only]
    T --> W[Multiple 2x passes for quality]
```

## Tab 5: Video Re-encoding

```mermaid
graph TD
    A[Tab 5: Video Re-encoding] --> B[Video Upload]
    B --> C[Re-encode Button]
    
    C --> D[handle_video_reencoding]
    D --> E[Log Input Details]
    E --> F[Type checking]
    E --> G[Representation logging]
    E --> H[Boolean evaluation]
    
    D --> I{Video exists?}
    I -->|No| J[Error: No video uploaded]
    I -->|Yes| K[SimpleVideoReencoder.reencode_video]
    
    K --> L[FFmpeg Processing]
    L --> M[Codec: h264]
    L --> N[Profile: high]
    L --> O[Pixel Format: yuv420p]
    L --> P[Preset: medium]
    L --> Q[Quality: CRF 18]
    L --> R[Color Space: BT.709]
    
    M --> S[Generate output path]
    N --> S
    O --> S
    P --> S
    Q --> S
    R --> S
    
    S --> T[Output Video]
    S --> U[Status Message]
    S --> V[Encoding Info Display]
```

## Summary

These diagrams illustrate the flow and logic of each tab in the RIFE application:

1. **Tab 1** - Frame Extraction: Handles video frame extraction with validation and auto-populates Tab 2 with reversed frame order
2. **Tab 2** - Image Interpolation: Performs image interpolation with memory-efficient options (standard recursive or disk-based)
3. **Tab 3** - Chained Video Interpolation: Chains multiple videos with smooth transitions between them
4. **Tab 4** - Video Interpolation: Increases video frame rate using multiple 2x passes while maintaining duration
5. **Tab 5** - Video Re-encoding: Re-encodes videos with professional standards using FFmpeg

Each diagram shows the key components, decision points, and data flow for its respective functionality.