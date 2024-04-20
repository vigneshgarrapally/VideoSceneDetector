from flask import (
    render_template,
    request,
    flash,
    redirect,
    url_for,
    send_from_directory,
)
from werkzeug.utils import secure_filename
from pathlib import Path
import datetime
from app import app
import cv2
import binascii


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("video", None)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_folder = app.config["UPLOAD_FOLDER"]
            # create a folder with unique name based on current date time and pathlib
            unique_folder = (
                Path.cwd()
                / upload_folder
                / datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            )
            unique_folder.mkdir(parents=True, exist_ok=True)

            file_path = unique_folder / filename
            file.save(str(file_path))

            # Process the video to extract scene timestamps and thumbnails
            scenes = detect_scenes(file_path, unique_folder)

            return render_template("upload.html", scenes=scenes)

    return render_template("upload.html")


def detect_scenes(file_path, output_folder):
    # Load the video using OpenCV
    cap = cv2.VideoCapture(str(file_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    scene_interval = 20  # Interval for capturing scenes (every 60 seconds)
    scenes = []
    print(f"Total frames: {frame_count}, Frame rate: {frame_rate}")
    # Create a folder for storing thumbnails if it doesn't exist
    thumbnails_folder = Path(output_folder) / "thumbnails"
    thumbnails_folder.mkdir(parents=True, exist_ok=True)
    # Loop through the video frames and capture scenes at specific intervals
    for i in range(0, frame_count, scene_interval * frame_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if not ret:
            break

        # Generate a unique thumbnail filename based on the frame number
        thumbnail_filename = f"thumbnail_{i}.jpg"
        thumbnail_path = thumbnails_folder / thumbnail_filename

        # Save the frame as an image
        cv2.imwrite(str(thumbnail_path), frame)
        relative_thumbnail_path = thumbnail_path.relative_to(Path.cwd())
        # Calculate the timestamp for this scene
        timestamp = i / frame_rate  # Convert to seconds

        # Append scene information
        scenes.append(
            {
                "timestamp": timestamp,
                "thumbnail": binascii.hexlify(
                    str(relative_thumbnail_path).encode("utf-8")
                ).decode(),
            }
        )

    cap.release()
    print(f"Detected {len(scenes)} scenes")
    print(scenes)
    return scenes


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/cdn/<filepath>")
def download_file(filepath):
    """
    Download a file from the specified filepath.

    Args:
        filepath (str): The path of the file to be downloaded.

    Returns:
        str: The file content if found, or "File not found" with status code 404.
    """
    filepath_decoded = binascii.unhexlify(filepath.encode("utf-8")).decode()
    file_path = Path.cwd() / filepath_decoded
    print(file_path)
    if not file_path.is_file():
        return "File not found", 404
    return send_from_directory(
        str(file_path.parent), file_path.name, as_attachment=False
    )
