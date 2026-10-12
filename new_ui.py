# new_ui.py
# The animated UI with LIVE ATTENDANCE LIST.

import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import threading
import backend
import time
import datetime


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# Color palette for person avatars (cycles)
AVATAR_COLORS = [
    "#00adb5",  # teal
    "#ff5722",  # orange
    "#9c27b0",  # purple
    "#4caf50",  # green
    "#e91e63",  # pink
    "#3f51b5",  # indigo
    "#ff9800",  # amber
    "#009688",  # dark teal
]


class AttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🎓 Smart Attendance System")
        self.geometry("1400x850")
        self.minsize(1200, 750)

        # Fade-in
        self.attributes('-alpha', 0.0)
        self.fade_in()

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        self.camera_running = True
        self.cap = None
        self._last_ids_seen = set()   # for pulse animation
        self._toast_ref = None

        self.build_left_panel()
        self.build_right_panel()

        self.cap = backend.get_camera()
        self.update_camera()

        # Auto-refresh attendance list every 2 seconds
        self.refresh_attendance_list()

        self.protocol("WM_DELETE_WINDOW", self.quit_app)

    # ───────── FADE-IN ANIMATION ─────────
    def fade_in(self, alpha=0.0):
        alpha += 0.05
        if alpha <= 1.0:
            self.attributes('-alpha', alpha)
            self.after(20, lambda: self.fade_in(alpha))

    # ───────── LEFT: LIVE CAMERA ─────────
    def build_left_panel(self):
        left = ctk.CTkFrame(self, corner_radius=20, fg_color="#1a1a2e")
        left.grid(row=0, column=0, padx=(20, 10), pady=20, sticky="nsew")
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            left, text="📹  LIVE CAMERA",
            font=("Segoe UI", 22, "bold"),
            text_color="#00ffcc"
        ).grid(row=0, column=0, pady=(20, 10))

        cam_frame = ctk.CTkFrame(left, fg_color="#0f0f1e", corner_radius=15)
        cam_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        cam_frame.grid_rowconfigure(0, weight=1)
        cam_frame.grid_columnconfigure(0, weight=1)

        self.cam_label = ctk.CTkLabel(cam_frame, text="")
        self.cam_label.grid(row=0, column=0)

        self.cam_info = ctk.CTkLabel(
            left, text="● Camera Active",
            font=("Segoe UI", 12), text_color="#888"
        )
        self.cam_info.grid(row=2, column=0, pady=(0, 20))

    # ───────── RIGHT: CONTROLS + LIST ─────────
    def build_right_panel(self):
        right = ctk.CTkFrame(self, corner_radius=20, fg_color="#16213e")
        right.grid(row=0, column=1, padx=(10, 20), pady=20, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)

        # Title
        ctk.CTkLabel(
            right, text="🎓  SMART ATTENDANCE",
            font=("Segoe UI", 24, "bold"),
            text_color="#ffd700"
        ).pack(pady=(20, 2))

        ctk.CTkLabel(
            right, text="Face Recognition System",
            font=("Segoe UI", 12, "italic"),
            text_color="#888"
        ).pack(pady=(0, 15))

        # Input fields
        self.emp_id_entry = self.create_input(right, "👤  EMPLOYEE ID")
        self.emp_name_entry = self.create_input(right, "📝  FULL NAME")

        # Action buttons
        self.create_button(right, "✨  TAKE IMAGES", "#00adb5", self.on_take_images)
        self.create_button(right, "🧠  TRAIN MODEL", "#4a4e69", self.on_train_images)
        self.create_button(right, "🎯  TRACK ATTENDANCE", "#ff5722", self.on_track_images)
        self.create_button(right, "❌  QUIT", "#b00020", self.quit_app)

        # ═══════════════════════════════════════
        #  LIVE ATTENDANCE LIST PANEL
        # ═══════════════════════════════════════
        self.attendance_panel = ctk.CTkFrame(
            right, fg_color="#0f0f1e", corner_radius=15,
            border_width=1, border_color="#00ffcc"
        )
        self.attendance_panel.pack(pady=(15, 10), padx=20, fill="both", expand=True)

        # Header row: title + count
        header_row = ctk.CTkFrame(self.attendance_panel, fg_color="transparent")
        header_row.pack(fill="x", padx=15, pady=(12, 5))

        self.list_title = ctk.CTkLabel(
            header_row,
            text=f"📋  TODAY'S ATTENDANCE — {backend.get_today_date_str()}",
            font=("Segoe UI", 13, "bold"),
            text_color="#00ffcc"
        )
        self.list_title.pack(side="left")

        self.count_label = ctk.CTkLabel(
            header_row,
            text="0",
            font=("Segoe UI", 13, "bold"),
            text_color="#ffd700",
            width=40,
            fg_color="#1f2b47",
            corner_radius=10
        )
        self.count_label.pack(side="right")

        # Scrollable list of people
        self.people_frame = ctk.CTkScrollableFrame(
            self.attendance_panel,
            fg_color="transparent",
            scrollbar_button_color="#00ffcc",
            scrollbar_button_hover_color="#00cc99"
        )
        self.people_frame.pack(fill="both", expand=True, padx=10, pady=(5, 12))

        # Empty placeholder
        self.empty_label = ctk.CTkLabel(
            self.people_frame,
            text="No one marked present yet today.\nStart TRACKING to add people! 🎯",
            font=("Segoe UI", 12, "italic"),
            text_color="#555"
        )
        self.empty_label.pack(pady=30)

        # Status footer
        self.status_text = ctk.CTkLabel(
            right, text="Ready",
            font=("Segoe UI", 12, "bold"),
            text_color="#aaa"
        )
        self.status_text.pack(pady=(0, 15))

    # ─────── INPUT ───────
    def create_input(self, parent, label_text):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(pady=6, padx=40, fill="x")
        ctk.CTkLabel(
            frame, text=label_text,
            font=("Segoe UI", 12, "bold"),
            text_color="#00ffcc", anchor="w"
        ).pack(fill="x")
        entry = ctk.CTkEntry(
            frame, height=38, corner_radius=10,
            font=("Segoe UI", 13),
            placeholder_text=f"Enter {label_text.split()[-1].lower()}..."
        )
        entry.pack(fill="x", pady=(3, 0))
        return entry

    # ─────── BUTTON ───────
    def create_button(self, parent, text, color, command):
        btn = ctk.CTkButton(
            parent, text=text, height=45, corner_radius=12,
            font=("Segoe UI", 14, "bold"),
            fg_color=color, hover_color=self.lighten(color),
            command=command
        )
        btn.pack(pady=5, padx=40, fill="x")
        btn.bind("<Enter>", lambda e: btn.configure(height=49))
        btn.bind("<Leave>", lambda e: btn.configure(height=45))
        return btn

    def lighten(self, hex_color):
        hex_color = hex_color.lstrip("#")
        r, g, b = [int(hex_color[i:i + 2], 16) for i in (0, 2, 4)]
        r, g, b = min(255, r + 40), min(255, g + 40), min(255, b + 40)
        return f"#{r:02x}{g:02x}{b:02x}"

    # ═══════════════════════════════════════════
    #  LIVE CAMERA LOOP
    # ═══════════════════════════════════════════
    def update_camera(self):
        if not self.camera_running:
            return

        if backend.is_camera_locked():
            self.cam_info.configure(text="● Camera busy (backend working)",
                                    text_color="#ffaa00")
            self.after(200, self.update_camera)
            return

        self.cam_info.configure(text="● Camera Active", text_color="#888")

        if self.cap is None or not self.cap.isOpened():
            self.cap = backend.get_camera()

        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (700, 500))
                img = ImageTk.PhotoImage(Image.fromarray(frame))
                self.cam_label.configure(image=img)
                self.cam_label.image = img

        self.after(30, self.update_camera)

    # ═══════════════════════════════════════════
    #  AUTO-REFRESH ATTENDANCE LIST
    # ═══════════════════════════════════════════
    def refresh_attendance_list(self):
        """Rebuild the list from backend every 2 seconds."""
        try:
            people = backend.get_today_attendance_list()
            self.render_attendance(people)
        except Exception as e:
            print(f"Refresh error: {e}")

        self.after(2000, self.refresh_attendance_list)

    def render_attendance(self, people):
        """Draw each person as a colored row."""
        # Clear existing widgets
        for widget in self.people_frame.winfo_children():
            widget.destroy()

        if not people:
            self.empty_label = ctk.CTkLabel(
                self.people_frame,
                text="No one marked present yet today.\nStart TRACKING to add people! 🎯",
                font=("Segoe UI", 12, "italic"),
                text_color="#555"
            )
            self.empty_label.pack(pady=30)
            self.count_label.configure(text="0")
            return

        # Update count badge
        self.count_label.configure(text=str(len(people)))

        # Build a card per person
        current_ids = set()
        for idx, person in enumerate(people):
            emp_id = str(person.get('Id', '?'))
            name = str(person.get('Name', 'Unknown'))
            time_str = str(person.get('Time', '--:--:--'))
            current_ids.add(emp_id)

            # Pick a color (stable per ID)
            color = AVATAR_COLORS[hash(emp_id) % len(AVATAR_COLORS)]

            # Card
            card = ctk.CTkFrame(
                self.people_frame,
                fg_color="#1f2b47",
                corner_radius=10,
                height=48
            )
            card.pack(fill="x", pady=3, padx=2)

            # Left: colored avatar circle
            avatar = ctk.CTkLabel(
                card,
                text=name[0].upper() if name else "?",
                font=("Segoe UI", 16, "bold"),
                text_color="white",
                fg_color=color,
                corner_radius=20,
                width=36, height=36
            )
            avatar.pack(side="left", padx=(10, 8), pady=6)

            # Middle: name + ID
            info = ctk.CTkFrame(card, fg_color="transparent")
            info.pack(side="left", fill="both", expand=True)

            ctk.CTkLabel(
                info, text=name,
                font=("Segoe UI", 13, "bold"),
                text_color="#ffffff", anchor="w"
            ).pack(anchor="w", pady=(4, 0))

            ctk.CTkLabel(
                info, text=f"ID · {emp_id}",
                font=("Segoe UI", 10),
                text_color="#8899aa", anchor="w"
            ).pack(anchor="w")

            # Right: time + present badge
            right_side = ctk.CTkFrame(card, fg_color="transparent")
            right_side.pack(side="right", padx=(0, 10))

            ctk.CTkLabel(
                right_side, text=time_str,
                font=("Segoe UI", 11, "bold"),
                text_color="#00ffcc"
            ).pack(anchor="e")

            ctk.CTkLabel(
                right_side, text="✅ PRESENT",
                font=("Segoe UI", 9, "bold"),
                text_color="#4caf50"
            ).pack(anchor="e")

        # ─── PULSE ANIMATION for newly added people ───
        new_ids = current_ids - self._last_ids_seen
        if new_ids and self._last_ids_seen:
            # Someone new showed up — show a special toast
            self.show_toast(f"🎉 {len(new_ids)} new arrival(s)!", "#4caf50")
        self._last_ids_seen = current_ids

    # ═══════════════════════════════════════════
    #  BUTTON ACTIONS
    # ═══════════════════════════════════════════
    def on_take_images(self):
        Id = self.emp_id_entry.get().strip()
        name = self.emp_name_entry.get().strip()
        if not Id or not name:
            self.set_status("⚠️ Enter ID and Name first", "#ffaa00")
            return
        self.set_status("📸 Capturing...", "#00adb5")
        threading.Thread(
            target=self._run_take_images,
            args=(Id, name), daemon=True
        ).start()

    def _run_take_images(self, Id, name):
        try:
            res = backend.TakeImages(
                Id, name,
                status_callback=lambda m: self.after(0, self.set_status, m, "#00adb5")
            )
            self.after(0, self.set_status, res, "#00ff00")
            self.after(0, self.show_toast, res, "#00adb5")
        except Exception as e:
            self.after(0, self.set_status, f"❌ {e}", "#ff0000")

    def on_train_images(self):
        self.set_status("🧠 Training...", "#4a4e69")
        threading.Thread(target=self._run_train_images, daemon=True).start()

    def _run_train_images(self):
        try:
            res = backend.TrainImages(
                status_callback=lambda m: self.after(0, self.set_status, m, "#4a4e69")
            )
            self.after(0, self.set_status, res, "#00ff00")
            self.after(0, self.show_toast, res, "#4a4e69")
        except Exception as e:
            self.after(0, self.set_status, f"❌ {e}", "#ff0000")

    def on_track_images(self):
        self.set_status("🎯 Tracking...", "#ff5722")
        threading.Thread(target=self._run_track_images, daemon=True).start()

    def _run_track_images(self):
        try:
            res = backend.TrackImages(
                status_callback=lambda m: self.after(0, self.set_status, m, "#ff5722")
            )
            self.after(0, self.set_status, res, "#00ff00")
            self.after(0, self.show_toast, res, "#ff5722")
        except Exception as e:
            self.after(0, self.set_status, f"❌ {e}", "#ff0000")

    # ═══════════════════════════════════════════
    #  STATUS & TOAST
    # ═══════════════════════════════════════════
    def set_status(self, text, color="#00ff00"):
        self.status_text.configure(text=text, text_color=color)

    def show_toast(self, msg, color):
        """Slide-in toast (only one at a time)."""
        if self._toast_ref is not None:
            try:
                self._toast_ref.destroy()
            except Exception:
                pass

        toast = ctk.CTkFrame(self, fg_color=color, corner_radius=15)
        toast.place(relx=1.05, rely=0.05, anchor="ne")
        self._toast_ref = toast

        ctk.CTkLabel(
            toast, text=msg,
            font=("Segoe UI", 13, "bold"),
            text_color="white", wraplength=300
        ).pack(padx=20, pady=15)

        def slide_in(x=1.05):
            if x > 0.98:
                x -= 0.02
                toast.place(relx=x, rely=0.05, anchor="ne")
                self.after(15, lambda: slide_in(x))
            else:
                self.after(2500, lambda: slide_out(0.98))

        def slide_out(x=0.98):
            if x < 1.15:
                x += 0.03
                toast.place(relx=x, rely=0.05, anchor="ne")
                self.after(15, lambda: slide_out(x))
            else:
                toast.destroy()
                self._toast_ref = None

        slide_in()

    # ───────── QUIT ─────────
    def quit_app(self):
        self.camera_running = False
        try:
            backend.release_camera()
        except Exception:
            pass
        self.destroy()