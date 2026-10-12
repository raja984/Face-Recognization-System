# backend.py
# Brain of the system - face recognition logic. No UI here.

import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time


# ─────────────────────────────────────────────
#  SILENCE OpenCV's noisy warnings
# ─────────────────────────────────────────────
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"


# ─────────────────────────────────────────────
#  SHARED CAMERA
# ─────────────────────────────────────────────
_CAMERA = None
_CAMERA_LOCKED = False


def _open_camera():
    global _CAMERA
    if _CAMERA is None or not _CAMERA.isOpened():
        _CAMERA = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not _CAMERA.isOpened():
            _CAMERA = cv2.VideoCapture(0)
        _CAMERA.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        _CAMERA.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return _CAMERA


def get_camera():
    return _open_camera()


def release_camera():
    global _CAMERA
    if _CAMERA is not None:
        _CAMERA.release()
        _CAMERA = None


def set_camera_locked(locked: bool):
    global _CAMERA_LOCKED
    _CAMERA_LOCKED = locked


def is_camera_locked() -> bool:
    return _CAMERA_LOCKED


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def _ensure_dirs():
    for d in ["TrainingImage", "TrainingImageLabel",
              "EmployeeDetails", "Attendance", "ImagesUnknown"]:
        os.makedirs(d, exist_ok=True)


def _load_employee_csv():
    """Load employee CSV safely — strips spaces from column names."""
    csv_file = os.path.join("EmployeeDetails", "EmployeeDetails.csv")
    if not os.path.isfile(csv_file):
        return None
    try:
        df = pd.read_csv(csv_file, dtype={'Id': str})
        df.columns = df.columns.str.strip()
        if 'Id' in df.columns:
            df['Id'] = df['Id'].astype(str).str.strip()
        if 'Name' in df.columns:
            df['Name'] = df['Name'].astype(str).str.strip()
        return df
    except Exception:
        return None


def _today_date_str():
    return datetime.datetime.now().strftime('%Y-%m-%d')


def _today_attendance_file():
    """Path to today's attendance CSV (one file per day)."""
    return os.path.join("Attendance", f"Attendance_{_today_date_str()}.csv")


def _load_today_attendance():
    """Load today's attendance if it exists, else create empty DataFrame."""
    col_names = ['Id', 'Name', 'Date', 'Time']
    path = _today_attendance_file()
    if os.path.isfile(path):
        try:
            df = pd.read_csv(path, dtype={'Id': str})
            df.columns = df.columns.str.strip()
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=col_names)


def _save_today_attendance(df):
    """Save today's attendance to its per-day CSV file."""
    path = _today_attendance_file()
    df.to_csv(path, index=False)


def _was_present_today(emp_id):
    """Check if this employee was already marked present today."""
    df = _load_today_attendance()
    if len(df) == 0 or 'Id' not in df.columns:
        return False
    return str(emp_id) in df['Id'].astype(str).values


def _mark_present(emp_id, name):
    """Add a record for this employee for today."""
    if _was_present_today(emp_id):
        return False, f"⏳ Already marked present today: {name}"

    now = datetime.datetime.now()
    record = {
        'Id': str(emp_id),
        'Name': str(name),
        'Date': now.strftime('%Y-%m-%d'),
        'Time': now.strftime('%H:%M:%S'),
    }

    df = _load_today_attendance()
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    _save_today_attendance(df)

    return True, f"✅ {name} marked PRESENT on {record['Date']} at {record['Time']}"


# ─────────────────────────────────────────────
#  ★ NEW: DUPLICATE DETECTION HELPERS
# ─────────────────────────────────────────────
def _id_exists(emp_id):
    """Returns True if this employee ID is already registered."""
    df = _load_employee_csv()
    if df is None or 'Id' not in df.columns:
        return False
    return str(emp_id).strip() in df['Id'].astype(str).str.strip().values


def _name_exists(name):
    """Returns True if this name is already registered (case-insensitive)."""
    df = _load_employee_csv()
    if df is None or 'Name' not in df.columns:
        return False
    existing = df['Name'].astype(str).str.strip().str.lower().values
    return name.strip().lower() in existing


def _find_id_by_name(name):
    """Return the ID associated with this name, or None."""
    df = _load_employee_csv()
    if df is None or 'Name' not in df.columns or 'Id' not in df.columns:
        return None
    match = df.loc[df['Name'].astype(str).str.strip().str.lower() ==
                   name.strip().lower(), 'Id'].values
    return str(match[0]) if len(match) > 0 else None


def _face_already_registered(frame_gray, tolerance=70):
    """
    Check if this face is already in the trained model.
    Returns (True, existing_id, confidence) if matched, else (False, None, None).
    """
    trainer_path = os.path.join("TrainingImageLabel", "Trainner.yml")
    if not os.path.isfile(trainer_path):
        return False, None, None

    df = _load_employee_csv()
    if df is None or 'Id' not in df.columns:
        return False, None, None

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(trainer_path)

        harcascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(harcascadePath)
        faces = faceCascade.detectMultiScale(frame_gray, 1.2, 5)

        if len(faces) == 0:
            return False, None, None

        # Use the biggest face
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        Id, conf = recognizer.predict(frame_gray[y:y + h, x:x + w])

        # Low confidence value = strong match
        if conf < tolerance:
            Id_str = str(Id)
            match = df.loc[df['Id'] == Id_str, 'Name'].values
            name = str(match[0]) if len(match) > 0 else f"ID-{Id_str}"
            return True, name, conf
    except Exception as e:
        print(f"Face check error: {e}")

    return False, None, None


# ─────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────
def get_today_attendance_list():
    """Returns a list of dicts for today's attendance."""
    df = _load_today_attendance()
    if len(df) == 0:
        return []
    return df.to_dict('records')


def get_today_count():
    """Return how many people are present today."""
    return len(_load_today_attendance())


def get_today_date_str():
    """Return today's date as YYYY-MM-DD."""
    return _today_date_str()


def get_all_employees():
    """Return a list of all registered employees."""
    df = _load_employee_csv()
    if df is None:
        return []
    return df.to_dict('records')


# ─────────────────────────────────────────────
#  TAKE IMAGES (WITH DUPLICATE CHECKS)
# ─────────────────────────────────────────────
def TakeImages(Id, name, status_callback=None):
    """Capture images with duplicate detection."""
    if not (is_number(Id) and name.isalpha()):
        if is_number(Id):
            return "❌ Enter Alphabetical Name"
        if name.isalpha():
            return "❌ Enter Numeric Id"
        return "❌ Invalid ID and Name"

    Id = str(Id).strip()
    name = name.strip()

    # ★ CHECK 1: Is the ID already registered?
    if _id_exists(Id):
        existing_name = None
        df = _load_employee_csv()
        if df is not None:
            match = df.loc[df['Id'] == Id, 'Name'].values
            existing_name = str(match[0]) if len(match) > 0 else "?"
        return (f"⚠️ ID '{Id}' is ALREADY registered "
                f"as '{existing_name}'. Use a different ID.")

    # ★ CHECK 2: Is the name already registered?
    if _name_exists(name):
        existing_id = _find_id_by_name(name)
        return (f"⚠️ Name '{name}' is ALREADY registered "
                f"with ID '{existing_id}'. Use a different name.")

    # ★ CHECK 3: Is this face already in the trained model?
    # We take a quick preview frame to check
    cam = _open_camera()
    ret, preview = cam.read()
    if ret and preview is not None:
        gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
        found, existing_name, conf = _face_already_registered(gray, tolerance=70)
        if found:
            return (f"⚠️ This face is ALREADY registered as "
                    f"'{existing_name}' (confidence: {conf:.1f}). "
                    f"Cannot add duplicate person!")

    # ─── All checks passed → start capturing ───
    _ensure_dirs()
    set_camera_locked(True)

    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    sampleNum = 0
    window_name = "Capturing Faces - Press Q to stop"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, img = cam.read()
            if not ret or img is None:
                time.sleep(0.05)
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                sampleNum += 1
                file_path = os.path.join(
                    "TrainingImage",
                    f"{name}.{Id}.{sampleNum}.jpg"
                )
                cv2.imwrite(file_path, gray[y:y + h, x:x + w])
                if status_callback:
                    status_callback(f"📸 Captured {sampleNum}/100")

            cv2.putText(img, f"{name} ({Id}) - {sampleNum}/100",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

            cv2.imshow(window_name, img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if sampleNum >= 100:
                break
    finally:
        cv2.destroyWindow(window_name)
        set_camera_locked(False)

    # Save to CSV
    csv_file = os.path.join("EmployeeDetails", "EmployeeDetails.csv")
    file_exists = os.path.isfile(csv_file) and os.path.getsize(csv_file) > 0
    with open(csv_file, 'a+', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Id", "Name"])
        writer.writerow([Id, name])

    return f"✅ Images Saved for ID: {Id}, Name: {name}"


# ─────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────
def getImagesAndLabels(path):
    if not os.path.isdir(path):
        return [], []
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    faces, Ids = [], []
    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces.append(imageNp)
            Ids.append(Id)
        except Exception as e:
            print(f"Skipping {imagePath}: {e}")
    return faces, Ids


def TrainImages(status_callback=None):
    _ensure_dirs()
    if status_callback:
        status_callback("🧠 Training model...")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, Id = getImagesAndLabels("TrainingImage")
    if len(faces) == 0:
        return "❌ No training images found!"

    recognizer.train(faces, np.array(Id))
    recognizer.save(os.path.join("TrainingImageLabel", "Trainner.yml"))
    return f"✅ Trained on {len(faces)} images!"


# ─────────────────────────────────────────────
#  TRACK
# ─────────────────────────────────────────────
def TrackImages(status_callback=None):
    """Track faces and mark attendance once per day per person."""
    _ensure_dirs()
    trainer_path = os.path.join("TrainingImageLabel", "Trainner.yml")
    if not os.path.isfile(trainer_path):
        return "❌ Please train the model first!"

    df = _load_employee_csv()
    if df is None or 'Id' not in df.columns:
        return "❌ EmployeeDetails.csv missing or has no 'Id' column!"

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_path)

    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    cam = _open_camera()
    set_camera_locked(True)

    font = cv2.FONT_HERSHEY_SIMPLEX
    window_name = "Tracking Attendance - Press Q to stop"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    new_marks = 0
    already_marks = 0
    seen_in_session = set()   # ← prevents re-marking same person in one session

    try:
        while True:
            ret, im = cam.read()
            if not ret or im is None:
                time.sleep(0.05)
                continue

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 200, 255), 2)
                Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                Id_str = str(Id)

                label_text = "Unknown"
                label_color = (0, 0, 255)

                if conf < 60:
                    try:
                        match = df.loc[df['Id'] == Id_str, 'Name'].values
                        name_str = str(match[0]) if len(match) > 0 else f"ID-{Id_str}"
                    except Exception:
                        name_str = f"ID-{Id_str}"

                    if _was_present_today(Id_str):
                        label_text = f"{name_str} (PRESENT)"
                        label_color = (0, 255, 0)
                        already_marks += 1
                    else:
                        # Only mark ONCE per session
                        if Id_str in seen_in_session:
                            label_text = f"{name_str} (PRESENT)"
                            label_color = (0, 255, 0)
                        else:
                            marked, msg = _mark_present(Id_str, name_str)
                            seen_in_session.add(Id_str)
                            if marked:
                                label_text = f"{name_str} (MARKED)"
                                label_color = (0, 255, 0)
                                new_marks += 1
                                if status_callback:
                                    status_callback(msg)
                            else:
                                label_text = f"{name_str} (ALREADY)"
                                label_color = (255, 200, 0)
                                already_marks += 1
                else:
                    if conf > 75:
                        noOfFile = len(os.listdir("ImagesUnknown")) + 1
                        cv2.imwrite(
                            os.path.join("ImagesUnknown", f"Image{noOfFile}.jpg"),
                            im[y:y + h, x:x + w]
                        )
                    label_text = "Unknown"
                    label_color = (0, 0, 255)

                cv2.putText(im, label_text, (x, y - 10), font, 0.8,
                            label_color, 2)

            cv2.imshow(window_name, im)
            if cv2.waitKey(1) == ord('q'):
                break

    except Exception as e:
        return f"❌ Error during tracking: {e}"
    finally:
        cv2.destroyWindow(window_name)
        set_camera_locked(False)

    total = len(_load_today_attendance())

    if new_marks == 0 and already_marks == 0:
        return "⚠️ No faces recognized. Try again with better lighting."

    return (
        f"✅ Session done. "
        f"New: {new_marks} | Already present: {already_marks} | "
        f"Today's total: {total} | File: Attendance_{_today_date_str()}.csv"
    )