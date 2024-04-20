# Video Scene Detector

This Flask application automates the process of detecting scenes in video files, providing timestamps for each detected scene and offering an optional functionality to split the videos into individual scenes. It is ideal for organizing video content from various sources, enhancing video-based projects, and improving content accessibility.

## Features

- **Video Upload:** Users can upload video files to be processed.
- **Scene Detection:** Automatic detection of scene changes within the videos.
- **Timestamps:** Each detected scene is accompanied by its start and end timestamps.
- **Video Splitting:** Optional splitting of the video into separate files based on detected scenes.

## Technologies Used

- Flask: For the web application framework.
- Python: Main programming language.
- OpenCV: For video frame analysis and scene detection.
- FFMPEG: For optional video splitting functionality.
- Docker: Recommended for deployment and environment management.

## Getting Started

### Prerequisites

- Python 3.6+
- Flask
- OpenCV-Python
- FFMPEG (for splitting videos)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/video_scene_detector.git
   cd video_scene_detector
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   python run.py
   ```

### Usage

Navigate to `http://localhost:5000` in your web browser to access the application. Upload a video file and the application will process it, displaying the detected scenes and timestamps. Optionally, choose to split the video based on these scenes.

## Configuration

Modify `instance/config.py` to adjust application settings such as upload folder path, allowed video formats, and file size limits.