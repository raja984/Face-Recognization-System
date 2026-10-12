# setup.py
# Build script for cx_Freeze

from cx_Freeze import setup, Executable
import sys
import os

PYTHON_INSTALL_DIR = os.path.dirname(os.path.dirname(os.__file__))
os.environ['TCL_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tk8.6')

# Try to find customtkinter's data folder (themes, fonts, etc.)
try:
    import customtkinter
    ctk_path = os.path.dirname(customtkinter.__file__)
except ImportError:
    ctk_path = None

base = None
if sys.platform == 'win32':
    base = "Win32GUI"   # hides console window

# Packages cx_Freeze must include
packages = [
    "os", "sys", "cx_Freeze", "tkinter",
    "cv2", "numpy", "PIL", "pandas",
    "datetime", "time", "customtkinter",
    "backend", "new_ui",
]

# Files/folders to copy next to the .exe
include_files = [
    "haarcascade_frontalface_default.xml",
    "im0.jpg",
    "im1.jpg",
    "TrainingImage",
    "TrainingImageLabel",
    "EmployeeDetails",
    "Attendance",
    "ImagesUnknown",
]

# Add customtkinter data files
if ctk_path:
    include_files.append((ctk_path, "lib/customtkinter"))

executables = [Executable("main.py", base=base, target_name="SmartAttendance.exe")]

options = {
    'build_exe': {
        'packages': packages,
        'include_files': include_files,
        'excludes': ['matplotlib', 'scipy', 'PyQt5'],
    },
}

setup(
    name="SmartAttendance",
    options=options,
    version="1.0.0",
    description="Face Recognition Attendance System",
    executables=executables
)