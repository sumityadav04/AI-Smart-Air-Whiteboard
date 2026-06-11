"""
main.py  —  Air Drawing App
============================
Virtual whiteboard controlled by hand gestures + mouse.
         
HOW TO USE
----------
  ✋ 1 finger (index only)       → Draw on canvas
  ✌️  2 fingers (index + middle)  → Pause drawing; point at toolbar to pick tool
  🖱  Mouse click on toolbar      → Pick color / brush / eraser / clear
  ⌨️  Q or ESC                    → Quit

TOOLBAR (top strip)           
-------------------
  [ Color swatches ] [ Brush sizes ] [ Erase ] [ Clear ]

REQUIREMENTS
------------
  pip install opencv-python mediapipe numpy
"""

import time
from typing import List, Tuple

import cv2
import numpy as np

from drawing_utils import TOOLBAR_HEIGHT, DrawingCanvas
from hand_tracker import HandTracker


WINDOW_NAME = "✏️  Air Draw  |  1-finger=Draw  2-fingers=Select  Q=Quit"

# Resolutions to try, best first
PREFERRED_RESOLUTIONS: List[Tuple[int, int]] = [
    (1920, 1080),
    (1280, 720),
    (960, 540),
    (640, 480),
]


# ─────────────────────────────────────────────────────────────────────────────
class FPSCounter:
    """Exponential moving average FPS — no jitter."""

    def __init__(self, smooth: float = 0.92):
        self._smooth   = smooth
        self._fps      = 0.0
        self._prev     = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        dt  = now - self._prev
        self._prev = now
        if dt > 0:
            self._fps = self._smooth * self._fps + (1 - self._smooth) / dt
        return self._fps


# ─────────────────────────────────────────────────────────────────────────────
class Camera:
    """
    Opens the webcam and negotiates the best available resolution.
    Mirrors every frame so left↔right feels natural.
    """

    def __init__(self, index: int = 0):
        # On Windows, DirectShow is faster than the default backend
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(
                "❌  Cannot open webcam.\n"
                "    Make sure your camera is connected and not used by another app."
            )

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # reduce latency

        self.width, self.height = self._pick_resolution(cap)
        self._cap    = cap
        self._buffer = np.empty((self.height, self.width, 3), dtype=np.uint8)
        print(f"  Camera opened at {self.width}×{self.height}")

    @staticmethod
    def _pick_resolution(cap: cv2.VideoCapture) -> Tuple[int, int]:
        best_w, best_h = 0, 0
        for target_w, target_h in PREFERRED_RESOLUTIONS:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  target_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
            cap.set(cv2.CAP_PROP_FPS, 30)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            if w * h > best_w * best_h:
                best_w, best_h = w, h
            if w >= 1280:   # good enough, stop searching
                break
        if best_w == 0:
            best_w, best_h = 640, 480
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  best_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best_h)
        return best_w, best_h

    def read(self) -> Tuple[bool, np.ndarray]:
        ok, raw = self._cap.read()
        if not ok or raw is None:
            return False, self._buffer
        h, w = raw.shape[:2]
        if w != self.width or h != self.height:
            interp = cv2.INTER_AREA if w > self.width else cv2.INTER_CUBIC
            cv2.resize(raw, (self.width, self.height),
                       dst=self._buffer, interpolation=interp)
        else:
            np.copyto(self._buffer, raw)
        cv2.flip(self._buffer, 1, dst=self._buffer)   # mirror
        return True, self._buffer

    def release(self):
        self._cap.release()


# ─────────────────────────────────────────────────────────────────────────────
def on_mouse(event, x, y, flags, canvas: DrawingCanvas):
    """
    Mouse callback — only the toolbar region is interactive via mouse.
    This lets you click colors/clear without needing hand gestures.
    """
    if y <= TOOLBAR_HEIGHT:
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            canvas.set_toolbar_hover((x, y))
        if event == cv2.EVENT_LBUTTONDOWN:
            canvas.end_stroke()
            canvas.handle_toolbar_point((x, y))
    elif event == cv2.EVENT_MOUSEMOVE:
        canvas.set_toolbar_hover(None)


# ─────────────────────────────────────────────────────────────────────────────
def add_canvas_tint(frame: np.ndarray, toolbar_h: int):
    """
    Subtle dark tint on the drawing area so white strokes pop.
    """
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (0, toolbar_h),
                  (frame.shape[1], frame.shape[0]),
                  (25, 22, 20), -1)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*50)
    print("  ✏️   Air Draw  —  Virtual Whiteboard")
    print("="*50)
    print("  Gestures:")
    print("    Index finger only  →  Draw")
    print("    Index + Middle     →  Select toolbar")
    print("  Mouse: click toolbar anytime")
    print("  Q / ESC to quit")
    print("="*50 + "\n")

    camera  = Camera()
    tracker = HandTracker()
    canvas  = DrawingCanvas(width=camera.width, height=camera.height)
    fps     = FPSCounter()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, camera.width, camera.height)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, canvas)

    was_drawing = False

    while True:
        ok, frame = camera.read()
        if not ok:
            print("⚠️  Could not read from webcam.")
            break

        # Subtle background tint for better drawing visibility
        add_canvas_tint(frame, TOOLBAR_HEIGHT)

        # ── Hand detection ────────────────────────────────────────────────
        hand      = tracker.process(frame)
        is_drawing = False
        mode_label = "Ready"

        if hand is not None:
            tip = hand.index_tip

            if hand.is_drawing_gesture:
                mode_label = "✏️  Drawing"
                is_drawing  = True
                canvas.set_toolbar_hover(None)
                canvas.add_stroke_point(tip)

            elif hand.is_pause_gesture:
                mode_label = "☝️  Selecting"
                canvas.end_stroke()
                canvas.set_toolbar_hover(tip)
                canvas.handle_toolbar_point(tip)

            else:
                mode_label = "✋ Idle"
                canvas.end_stroke()
                canvas.set_toolbar_hover(None)

            # Draw fingertip cursor
            canvas.draw_cursor(frame, tip, drawing=is_drawing)

            # End the current stroke if we just stopped drawing
            if was_drawing and not is_drawing:
                canvas.end_stroke()

            was_drawing = is_drawing

        else:
            canvas.end_stroke()
            was_drawing = False
            mode_label  = "No Hand"

        # ── Render layers ─────────────────────────────────────────────────
        canvas.composite(frame)        # paint strokes onto camera frame
        canvas.draw_toolbar(frame)     # toolbar on top
        canvas.draw_status(
            frame,
            fps=fps.tick(),
            mode_label=mode_label,
            hand_detected=hand is not None,
            resolution=f"{camera.width}×{camera.height}",
        )

        cv2.imshow(WINDOW_NAME, frame)

        # Q or ESC to quit
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
            break

    # ── Cleanup ───────────────────────────────────────────────────────────
    print("\n  Closing Air Draw. Goodbye! 👋")
    camera.release()
    tracker.close()
    cv2.destroyAllWindows()

        
if __name__ == "__main__":
    main()
