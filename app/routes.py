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
from app.adaptive_detector import AdaptiveDetector


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
            adaptive_threshold = float(
                request.form.get("adaptive_threshold", 3.0)
            )
            min_scene_len = int(request.form.get("min_scene_len", 15))
            window_width = int(request.form.get("window_width", 2))
            min_content_val = float(request.form.get("min_content_val", 15.0))
            content_detector = AdaptiveDetector(
                adaptive_threshold=adaptive_threshold,
                min_scene_len=min_scene_len,
                window_width=window_width,
                min_content_val=min_content_val,
            )
            # Process the video to extract scene timestamps and thumbnails
            scenes = detect_scenes(
                file_path, unique_folder, scene_detector=content_detector
            )
            return render_template("upload.html", scenes=scenes)

    return render_template("upload.html")


# Convert frames to a formatted timecode "HH:MM:SS[.nnn]"
def frames_to_timecode(frame_count, fps, precision=3):
    seconds = frame_count / fps
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    if precision > 0:
        secs = round(secs, precision)
    return f"{hrs:02}:{mins:02}:{secs:06.3f}"


def detect_scenes(file_path, output_folder, scene_detector):
    # Load the video using OpenCV
    cap = cv2.VideoCapture(str(file_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    scene_list = []
    frame_number = 0
    previous_scene_frame = 0

    # Create a folder for storing scenecut start frames
    scenecuts_folder = Path(output_folder) / "scenecuts"
    scenecuts_folder.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect scene cuts
        is_scene_cut = scene_detector.process_frame(frame_number, frame)
        if is_scene_cut:
            if previous_scene_frame != is_scene_cut[0]:
                scene_list.append((previous_scene_frame, is_scene_cut[0]))
            previous_scene_frame = is_scene_cut[0]

        frame_number += 1

    if previous_scene_frame != frame_number:
        scene_list.append((previous_scene_frame, frame_number))

    detected_scenes = []
    for i, (start, end) in enumerate(scene_list):
        start_time = frames_to_timecode(start, fps)
        end_time = frames_to_timecode(end, fps)
        scene_duration = frames_to_timecode(end - start, fps, precision=0)
        # save scene start frame as an image
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        ret, frame = cap.read()
        scene_img_path = scenecuts_folder / f"scene_{i:03d}.jpg"
        cv2.imwrite(str(scene_img_path), frame)
        detected_scenes.append(
            {
                "scene_number": i + 1,
                "start_time": start_time,
                "end_time": end_time,
                "duration": scene_duration,
                "scene_img_path": binascii.hexlify(
                    str(scene_img_path.relative_to(Path.cwd())).encode(
                        "utf-8"
                    )
                ).decode(),
            }
        )

    cap.release()
    return detected_scenes


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
