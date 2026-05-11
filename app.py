# ============================================
# FAKE VIDEO DETECTOR — DESKTOP APP
# DFD Model + FFT + Face Detection
# ============================================

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading

# ============================================
# CONFIGURATION
# ============================================

# ← CHANGED: new model name
MODEL_PATH           = os.path.join(
    os.path.dirname(__file__), "DFD_model_final.pth"
)
IMG_SIZE             = 224
FRAMES_PER_VIDEO     = 8      # ← CHANGED: 10 not 15
CONFIDENCE_THRESHOLD = 0.5

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================
# FFT FUNCTION ← NEW
# ============================================

def apply_fft(image):
    """
    Fast Fourier Transform
    Reveals hidden deepfake artifacts
    Must match training preprocessing!
    """
    gray      = np.mean(image, axis=2)
    fft       = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    magnitude = np.log(magnitude + 1e-8)
    mag_min   = magnitude.min()
    mag_max   = magnitude.max()
    if mag_max - mag_min > 0:
        magnitude = (magnitude - mag_min) / \
                    (mag_max - mag_min)
    magnitude         = (magnitude * 255)\
                         .astype(np.uint8)
    combined          = image.copy()
    combined[:, :, 2] = magnitude
    return combined

# ============================================
# LOAD MODEL
# ============================================

def load_model():
    """Load DFD trained EfficientNet-B4 model"""
    try:
        model = timm.create_model(
            'efficientnet_b4',
            pretrained  = False,
            num_classes = 2,
            drop_rate   = 0.3
        )
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device
        )
        model.load_state_dict(
            checkpoint['model_state_dict']
        )
        model = model.to(device)
        model.eval()

        # Get model info
        accuracy = checkpoint.get('accuracy', 0)
        dataset  = checkpoint.get('dataset', 'DFD')
        use_fft  = checkpoint.get('use_fft', True)

        return model, accuracy, dataset, use_fft

    except Exception as e:
        messagebox.showerror(
            "Error",
            f"Could not load model!\n{str(e)}\n\n"
            f"Make sure DFD_model_final.pth is in "
            f"the same folder as app.py"
        )
        return None, 0, 'Unknown', False

# ============================================
# FACE DETECTION
# ============================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

def extract_face_frames(video_path,
                        num_frames=FRAMES_PER_VIDEO):
    """Extract face frames from video"""
    cap          = cv2.VideoCapture(video_path)
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames == 0:
        cap.release()
        return [], 0, 0

    fps      = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    frame_indices = np.linspace(
        0, total_frames-1, num_frames, dtype=int
    )
    face_frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            continue

        frame_rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )
        gray = cv2.cvtColor(
            frame, cv2.COLOR_BGR2GRAY
        )

        try:
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor  = 1.1,
                minNeighbors = 5,
                minSize      = (30, 30)
            )

            if len(faces) > 0:
                largest    = max(
                    faces, key=lambda f: f[2]*f[3]
                )
                x, y, w, h = largest
                pad_x  = int(w * 0.2)
                pad_y  = int(h * 0.2)
                fh, fw = frame_rgb.shape[:2]
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(fw, x + w + pad_x)
                y2 = min(fh, y + h + pad_y)
                face = frame_rgb[y1:y2, x1:x2]

                if face.size > 0:
                    face = cv2.resize(
                        face, (IMG_SIZE, IMG_SIZE)
                    )
                else:
                    face = cv2.resize(
                        frame_rgb,
                        (IMG_SIZE, IMG_SIZE)
                    )
            else:
                face = cv2.resize(
                    frame_rgb, (IMG_SIZE, IMG_SIZE)
                )

            # ← CHANGED: Apply FFT to each frame
            face = apply_fft(face)
            face_frames.append(face)

        except Exception:
            fallback = cv2.resize(
                frame_rgb, (IMG_SIZE, IMG_SIZE)
            )
            # ← CHANGED: Apply FFT to fallback too
            fallback = apply_fft(fallback)
            face_frames.append(fallback)

    cap.release()
    return face_frames, total_frames, duration

# ============================================
# PREDICTION
# ============================================

# ← CHANGED: Same transform as training
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_video(video_path, model,
                  progress_callback=None):
    """Predict if video is REAL or FAKE"""

    frames, total_frames, duration = \
        extract_face_frames(video_path)

    if len(frames) == 0:
        return None

    fake_probs = []

    with torch.no_grad():
        for i, frame in enumerate(frames):
            # Frame already has FFT applied
            pil_frame    = Image.fromarray(
                frame.astype(np.uint8)
            )
            tensor_frame = transform(pil_frame)\
                           .unsqueeze(0).to(device)
            output       = model(tensor_frame)
            prob         = torch.softmax(output, dim=1)
            fake_probs.append(prob[0][1].item())

            if progress_callback:
                progress = int(
                    (i + 1) / len(frames) * 100
                )
                progress_callback(progress)

    avg_fake_prob   = np.mean(fake_probs)
    predicted_label = "FAKE" \
                      if avg_fake_prob > CONFIDENCE_THRESHOLD \
                      else "REAL"
    confidence      = avg_fake_prob * 100 \
                      if predicted_label == "FAKE" \
                      else (1 - avg_fake_prob) * 100

    return {
        'verdict'      : predicted_label,
        'confidence'   : confidence,
        'fake_probs'   : fake_probs,
        'total_frames' : total_frames,
        'duration'     : duration,
        'fake_votes'   : sum(
            1 for p in fake_probs
            if p > CONFIDENCE_THRESHOLD
        ),
        'real_votes'   : sum(
            1 for p in fake_probs
            if p <= CONFIDENCE_THRESHOLD
        )
    }

# ============================================
# DESKTOP APP UI
# ============================================

class FakeVideoDetectorApp:
    def __init__(self, root):
        self.root       = root
        self.model      = None
        self.result     = None
        self.video_path = None
        self.use_fft    = True

        self.root.title(
            "🎬 Fake Video Detector — DFD + FFT"
        )
        self.root.geometry("900x700")
        self.root.configure(bg='#1a1a2e')
        self.root.resizable(True, True)

        self.setup_ui()
        self.load_model_async()

    def setup_ui(self):
        """Build the user interface"""

        # Title
        title_frame = tk.Frame(
            self.root, bg='#1a1a2e'
        )
        title_frame.pack(fill='x', pady=20)

        tk.Label(
            title_frame,
            text="🎬 Fake Video Detector",
            font=('Helvetica', 24, 'bold'),
            fg='#e94560',
            bg='#1a1a2e'
        ).pack()

        # ← CHANGED: Updated subtitle
        tk.Label(
            title_frame,
            text="EfficientNet-B4 + FFT + "
                 "Face Detection | DFD Dataset",
            font=('Helvetica', 11),
            fg='#a8a8b3',
            bg='#1a1a2e'
        ).pack()

        # Model status
        self.model_label = tk.Label(
            self.root,
            text="⏳ Loading model...",
            font=('Helvetica', 10),
            fg='#f5a623',
            bg='#1a1a2e'
        )
        self.model_label.pack()

        # Control panel
        control_frame = tk.Frame(
            self.root, bg='#16213e',
            relief='raised', bd=2
        )
        control_frame.pack(
            fill='x', padx=20, pady=10
        )

        self.browse_btn = tk.Button(
            control_frame,
            text="📁 Browse Video",
            command=self.browse_video,
            font=('Helvetica', 12, 'bold'),
            bg='#0f3460',
            fg='white',
            padx=20, pady=10,
            relief='flat',
            cursor='hand2'
        )
        self.browse_btn.pack(
            side='left', padx=10, pady=10
        )

        self.analyze_btn = tk.Button(
            control_frame,
            text="🔍 Analyze Video",
            command=self.analyze_video,
            font=('Helvetica', 12, 'bold'),
            bg='#e94560',
            fg='white',
            padx=20, pady=10,
            relief='flat',
            cursor='hand2',
            state='disabled'
        )
        self.analyze_btn.pack(
            side='left', padx=10, pady=10
        )

        self.video_label = tk.Label(
            control_frame,
            text="No video selected",
            font=('Helvetica', 10),
            fg='#a8a8b3',
            bg='#16213e'
        )
        self.video_label.pack(
            side='left', padx=10
        )

        # Progress bar
        progress_frame = tk.Frame(
            self.root, bg='#1a1a2e'
        )
        progress_frame.pack(
            fill='x', padx=20, pady=5
        )

        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            length=400
        )
        self.progress_bar.pack(
            side='left', padx=5
        )

        self.progress_label = tk.Label(
            progress_frame,
            text="",
            font=('Helvetica', 10),
            fg='#a8a8b3',
            bg='#1a1a2e'
        )
        self.progress_label.pack(side='left')

        # Result panel
        self.result_frame = tk.Frame(
            self.root, bg='#16213e',
            relief='raised', bd=2
        )
        self.result_frame.pack(
            fill='x', padx=20, pady=10
        )

        self.verdict_label = tk.Label(
            self.result_frame,
            text="🎯 Analyze a video to see results",
            font=('Helvetica', 20, 'bold'),
            fg='#a8a8b3',
            bg='#16213e'
        )
        self.verdict_label.pack(pady=15)

        self.details_label = tk.Label(
            self.result_frame,
            text="",
            font=('Helvetica', 11),
            fg='#a8a8b3',
            bg='#16213e',
            justify='center'
        )
        self.details_label.pack(pady=5)

        # Chart frame
        self.chart_frame = tk.Frame(
            self.root, bg='#1a1a2e'
        )
        self.chart_frame.pack(
            fill='both', expand=True,
            padx=20, pady=10
        )

    def load_model_async(self):
        """Load model in background thread"""
        def load():
            model, accuracy, dataset, use_fft = \
                load_model()
            if model:
                self.model   = model
                self.use_fft = use_fft
                self.model_label.config(
                    text=f"✅ Model loaded! "
                         f"Dataset: {dataset} | "
                         f"Accuracy: {accuracy:.2f}% | "
                         f"FFT: {'✅' if use_fft else '❌'}",
                    fg='#4caf50'
                )
            else:
                self.model_label.config(
                    text="❌ Model failed to load!",
                    fg='#e94560'
                )

        thread = threading.Thread(target=load)
        thread.daemon = True
        thread.start()

    def browse_video(self):
        """Open file dialog"""
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files",
                 "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )

        if path:
            self.video_path = path
            filename = os.path.basename(path)
            self.video_label.config(
                text=f"📹 {filename}",
                fg='white'
            )
            if self.model:
                self.analyze_btn.config(
                    state='normal'
                )
            self.verdict_label.config(
                text="🎯 Click Analyze to start",
                fg='#a8a8b3'
            )
            self.details_label.config(text="")

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_var.set(value)
        self.progress_label.config(
            text=f"Analyzing... {value}%"
        )
        self.root.update_idletasks()

    def analyze_video(self):
        """Run analysis"""
        if not self.video_path or not self.model:
            return

        self.analyze_btn.config(state='disabled')
        self.browse_btn.config(state='disabled')
        self.progress_var.set(0)

        def run_analysis():
            result = predict_video(
                self.video_path,
                self.model,
                self.update_progress
            )
            self.root.after(
                0, self.show_result, result
            )

        thread = threading.Thread(
            target=run_analysis
        )
        thread.daemon = True
        thread.start()

    def show_result(self, result):
        """Display results"""

        self.analyze_btn.config(state='normal')
        self.browse_btn.config(state='normal')
        self.progress_label.config(
            text="✅ Analysis complete!"
        )

        if result is None:
            self.verdict_label.config(
                text="❌ Could not analyze video!",
                fg='#e94560'
            )
            return

        self.result = result

        if result['verdict'] == "FAKE":
            verdict_text  = "🔴 FAKE VIDEO DETECTED"
            verdict_color = "#e94560"
        else:
            verdict_text  = "🟢 REAL VIDEO"
            verdict_color = "#4caf50"

        self.verdict_label.config(
            text=verdict_text,
            fg=verdict_color
        )

        details = (
            f"Confidence: {result['confidence']:.1f}%  |  "
            f"Duration: {result['duration']:.1f}s  |  "
            f"Frames: {FRAMES_PER_VIDEO}  |  "
            f"FFT: ✅  |  "
            f"Fake votes: {result['fake_votes']}"
            f"/{FRAMES_PER_VIDEO}  |  "
            f"Real votes: {result['real_votes']}"
            f"/{FRAMES_PER_VIDEO}"
        )
        self.details_label.config(
            text=details, fg='white'
        )

        self.draw_chart(result)

    def draw_chart(self, result):
        """Draw frame analysis chart"""

        for widget in self.chart_frame\
                          .winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')

        probs_y  = [p * 100
                    for p in result['fake_probs']]
        frames_x = range(1, len(probs_y) + 1)
        colors   = ['#e94560' if p > 50
                    else '#4caf50' for p in probs_y]

        bars = ax.bar(
            frames_x, probs_y,
            color=colors, alpha=0.8,
            edgecolor='white', linewidth=0.5
        )

        ax.axhline(
            y=50, color='yellow',
            linestyle='--', linewidth=2,
            label='Decision Threshold (50%)'
        )

        for bar, prob in zip(bars, probs_y):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f'{prob:.0f}%',
                ha='center', va='bottom',
                fontsize=7, color='white',
                fontweight='bold'
            )

        ax.set_title(
            f'Frame-by-Frame Analysis — '
            f'Verdict: {result["verdict"]} '
            f'({result["confidence"]:.1f}% confidence)'
            f' | FFT: ✅',
            fontsize=11, fontweight='bold',
            color='white'
        )
        ax.set_xlabel('Frame Number', color='white')
        ax.set_ylabel('Fake Probability (%)',
                      color='white')
        ax.set_ylim([0, 115])
        ax.set_xticks(list(frames_x))
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(
            facecolor='#16213e',
            labelcolor='white'
        )
        ax.grid(True, alpha=0.2,
                axis='y', color='white')

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(
            fig, master=self.chart_frame
        )
        canvas.draw()
        canvas.get_tk_widget().pack(
            fill='both', expand=True
        )

# ============================================
# RUN THE APP
# ============================================

if __name__ == "__main__":
    root = tk.Tk()
    app  = FakeVideoDetectorApp(root)
    root.mainloop()