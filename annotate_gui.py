
import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional

import cv2
from PIL import Image, ImageTk


class FrameAnnotatorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Video Frame Annotator")
        self.root.geometry("900x700")

        # State
        self.video_path: Optional[str] = None
        self.frames: list = []
        self.current_frame_idx: int = 0
        self.annotations: Dict[str, Dict[str, str]] = {}
        self.total_frames: int = 0

        # UI Components
        self._build_ui()
        self._bind_keyboard_shortcuts()

    def _build_ui(self):
        # Top control panel
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(control_frame, text="Load Video", command=self._load_video).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(control_frame, text="Export Annotations", command=self._export_annotations).pack(
            side=tk.LEFT, padx=5
        )
        self.progress_label = ttk.Label(control_frame, text="No video loaded")
        self.progress_label.pack(side=tk.LEFT, padx=20)

        # Frame display area
        display_frame = ttk.Frame(self.root, padding=10)
        display_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(display_frame, bg="gray", width=640, height=480)
        self.canvas.pack()

        # Annotation controls
        annotation_frame = ttk.LabelFrame(self.root, text="Annotation", padding=15)
        annotation_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Eye state
        eye_frame = ttk.Frame(annotation_frame)
        eye_frame.pack(side=tk.LEFT, padx=20)
        ttk.Label(eye_frame, text="Eye State:").pack(anchor=tk.W)
        self.eye_var = tk.StringVar(value="Open")
        ttk.Radiobutton(eye_frame, text="Open", variable=self.eye_var, value="Open").pack(anchor=tk.W)
        ttk.Radiobutton(eye_frame, text="Closed", variable=self.eye_var, value="Closed").pack(anchor=tk.W)

        # Posture
        posture_frame = ttk.Frame(annotation_frame)
        posture_frame.pack(side=tk.LEFT, padx=20)
        ttk.Label(posture_frame, text="Posture:").pack(anchor=tk.W)
        self.posture_var = tk.StringVar(value="Straight")
        ttk.Radiobutton(posture_frame, text="Straight", variable=self.posture_var, value="Straight").pack(
            anchor=tk.W
        )
        ttk.Radiobutton(posture_frame, text="Hunched", variable=self.posture_var, value="Hunched").pack(
            anchor=tk.W
        )

        # Navigation
        nav_frame = ttk.Frame(self.root, padding=10)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Button(nav_frame, text="◀ Previous (Left Arrow)", command=self._prev_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Save & Next ▶ (Right Arrow)", command=self._next_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Skip Frame (S)", command=self._skip_frame).pack(side=tk.LEFT, padx=5)

        # Keyboard shortcuts info
        info_label = ttk.Label(
            self.root,
            text="Keyboard: 1=Open  2=Closed  3=Straight  4=Hunched  Left/Right=Nav  S=Skip  Ctrl+S=Export",
            font=("Arial", 9),
            foreground="blue",
        )
        info_label.pack(side=tk.BOTTOM, pady=5)

    def _bind_keyboard_shortcuts(self):
        """Bind keyboard shortcuts for annotation and navigation."""
        # Eye state shortcuts
        self.root.bind("1", lambda e: self.eye_var.set("Open"))
        self.root.bind("2", lambda e: self.eye_var.set("Closed"))

        # Posture shortcuts
        self.root.bind("3", lambda e: self.posture_var.set("Straight"))
        self.root.bind("4", lambda e: self.posture_var.set("Hunched"))

        # Navigation
        self.root.bind("<Left>", lambda e: self._prev_frame())
        self.root.bind("<Right>", lambda e: self._next_frame())
        self.root.bind("s", lambda e: self._skip_frame())
        self.root.bind("S", lambda e: self._skip_frame())

        # Export
        self.root.bind("<Control-s>", lambda e: self._export_annotations())
        self.root.bind("<Control-S>", lambda e: self._export_annotations())

    def _load_video(self):
        video_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")],
        )
        if not video_path:
            return

        self.video_path = video_path
        self.frames = []
        self.annotations = {}
        self.current_frame_idx = 0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Unable to open video: {video_path}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Store frames as RGB for PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frames.append(rgb_frame)

        cap.release()
        self.total_frames = len(self.frames)

        if self.total_frames == 0:
            messagebox.showerror("Error", "No frames extracted from video.")
            return

        messagebox.showinfo("Success", f"Loaded {self.total_frames} frames from video.")
        self._display_frame()

    def _display_frame(self):
        if not self.frames:
            return

        frame = self.frames[self.current_frame_idx]
        img = Image.fromarray(frame)

        # Resize to fit canvas
        canvas_width = 640
        canvas_height = 480
        img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)

        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.photo)

        # Update progress
        self.progress_label.config(
            text=f"Frame {self.current_frame_idx + 1} / {self.total_frames}   "
            f"(Annotated: {len(self.annotations)})"
        )

        # Load existing annotation if present
        frame_key = str(self.current_frame_idx)
        if frame_key in self.annotations:
            self.eye_var.set(self.annotations[frame_key]["eye_state"])
            self.posture_var.set(self.annotations[frame_key]["posture"])
        else:
            # Reset to defaults
            self.eye_var.set("Open")
            self.posture_var.set("Straight")

    def _save_current_annotation(self):
        frame_key = str(self.current_frame_idx)
        self.annotations[frame_key] = {
            "eye_state": self.eye_var.get(),
            "posture": self.posture_var.get(),
        }

    def _next_frame(self):
        if not self.frames:
            return

        self._save_current_annotation()

        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self._display_frame()
        else:
            messagebox.showinfo("Done", "You have reached the last frame.")

    def _prev_frame(self):
        if not self.frames:
            return

        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self._display_frame()
        else:
            messagebox.showinfo("Info", "Already at the first frame.")

    def _skip_frame(self):
        if not self.frames:
            return

        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self._display_frame()
        else:
            messagebox.showinfo("Done", "You have reached the last frame.")

    def _export_annotations(self):
        if not self.annotations:
            messagebox.showwarning("Warning", "No annotations to export.")
            return

        output_path = filedialog.asksaveasfilename(
            title="Save Annotations",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            initialfile="labelled_Data.json",
        )
        if not output_path:
            return

        output_data = {"labels_per_frame": self.annotations}

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)
            messagebox.showinfo("Success", f"Annotations saved to:\n{output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations:\n{e}")


def main():
    root = tk.Tk()
    app = FrameAnnotatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
