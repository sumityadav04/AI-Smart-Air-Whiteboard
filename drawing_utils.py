"""
drawing_utils.py
----------------
Handles everything related to drawing:
  - The transparent canvas where strokes are painted
  - The toolbar (colors, brush sizes, eraser, clear)
  - Compositing the canvas onto the camera frame
  - Status bar (FPS, mode, resolution)
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


# ── Layout constants ──────────────────────────────────────────────────────────
TOOLBAR_HEIGHT = 80          # pixels tall
STATUS_BAR_HEIGHT = 30       # bottom status strip
CURSOR_RADIUS = 10           # fingertip cursor ring

# ── Color palette (BGR) ───────────────────────────────────────────────────────
COLORS = {
    "White"  : (255, 255, 255),
    "Red"    : (0,   0,   220),
    "Orange" : (0,   140, 255),
    "Yellow" : (0,   220, 220),
    "Green"  : (0,   200,  80),
    "Cyan"   : (200, 200,   0),
    "Blue"   : (220,  80,   0),
    "Purple" : (180,  40, 160),
    "Pink"   : (180,  80, 220),
}

BRUSH_SIZES = [4, 8, 14, 22]   # thin → thick
ERASER_SIZE = 40


# ── Small helpers ─────────────────────────────────────────────────────────────

def _rounded_rect(img, x1, y1, x2, y2, radius, color, thickness=-1):
    """Draw a filled or outlined rounded rectangle."""
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(img, (cx, cy), radius, color, thickness)


# ─────────────────────────────────────────────────────────────────────────────
class DrawingCanvas:
    """
    Maintains a transparent (BGRA) canvas and all toolbar logic.
    """

    def __init__(self, width: int, height: int):
        self.width  = width
        self.height = height

        # Transparent layer where we paint strokes
        self._canvas = np.zeros((height, width, 4), dtype=np.uint8)

        # Current tool state
        self._color      : Tuple[int,int,int] = COLORS["White"]
        self._brush_size : int  = BRUSH_SIZES[1]
        self._eraser_on  : bool = False

        # Stroke buffering (we connect dots → smooth lines)
        self._prev_point : Optional[Tuple[int,int]] = None

        # Hover highlight for toolbar items
        self._hovered_item = None   # string key of hovered item

        # Build toolbar button layout once
        self._buttons = self._build_buttons()

    # ── Public drawing API ────────────────────────────────────────────────────

    def add_stroke_point(self, pt: Tuple[int, int]):
        """Call every frame while the user is drawing."""
        if pt[1] <= TOOLBAR_HEIGHT:     # don't draw inside toolbar
            self._prev_point = None
            return

        color   = (255, 255, 255, 255) if self._eraser_on else (*self._color, 255)
        size    = ERASER_SIZE if self._eraser_on else self._brush_size
        blend   = cv2.MIXED if self._eraser_on else None   # not used directly

        if self._prev_point:
            if self._eraser_on:
                # Erase: draw black+transparent over canvas
                cv2.line(self._canvas, self._prev_point, pt,
                         (0, 0, 0, 0), size * 2)
            else:
                cv2.line(self._canvas, self._prev_point, pt,
                         color, size, cv2.LINE_AA)
                # Draw circles at endpoints for smooth thick strokes
                cv2.circle(self._canvas, pt, size // 2, color, -1, cv2.LINE_AA)
        else:
            if not self._eraser_on:
                cv2.circle(self._canvas, pt, size // 2, color, -1, cv2.LINE_AA)

        self._prev_point = pt

    def end_stroke(self):
        """Call when the user lifts their finger or changes mode."""
        self._prev_point = None

    def handle_toolbar_point(self, pt: Tuple[int, int]):
        """
        Check whether *pt* is inside a toolbar button and act on it.
        Used both by mouse clicks and by the 2-finger gesture.
        """
        if pt[1] > TOOLBAR_HEIGHT:
            return
        for key, (x1, y1, x2, y2) in self._buttons.items():
            if x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2:
                self._activate_button(key)
                break

    def set_toolbar_hover(self, pt: Optional[Tuple[int, int]]):
        """Highlight the button under *pt* (or clear highlight if None)."""
        if pt is None or pt[1] > TOOLBAR_HEIGHT:
            self._hovered_item = None
            return
        for key, (x1, y1, x2, y2) in self._buttons.items():
            if x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2:
                self._hovered_item = key
                return
        self._hovered_item = None

    # ── Rendering ─────────────────────────────────────────────────────────────

    def composite(self, frame: np.ndarray):
        """Blend the canvas layer onto *frame* in-place."""
        alpha = self._canvas[:, :, 3:4].astype(np.float32) / 255.0
        for c in range(3):
            frame[:, :, c] = (
                self._canvas[:, :, c] * alpha[:, :, 0] +
                frame[:, :, c]        * (1.0 - alpha[:, :, 0])
            ).astype(np.uint8)

            def get_canvas_snapshot(self):
              """
              Returns the drawing canvas as PNG bytes.
              OCR will use this image.
              """

              rgb = self._canvas[:, :, :3]

              success, png = cv2.imencode(".png", rgb)

              if not success:
               return None

              return png.tobytes()


    def has_drawing(self):
         """
         True if something has been drawn.
         """

         return np.any(self._canvas[:, :, 3] > 0)

    def draw_toolbar(self, frame: np.ndarray):
        """Paint the toolbar strip at the top of *frame*."""
        # Dark background
        cv2.rectangle(frame, (0, 0), (self.width, TOOLBAR_HEIGHT),
                      (20, 20, 20), -1)
        cv2.line(frame, (0, TOOLBAR_HEIGHT), (self.width, TOOLBAR_HEIGHT),
                 (60, 60, 60), 1)

        for key, (x1, y1, x2, y2) in self._buttons.items():
            hovered  = (key == self._hovered_item)
            selected = self._is_selected(key)
            self._draw_button(frame, key, x1, y1, x2, y2, selected, hovered)

    def draw_cursor(self, frame: np.ndarray, pt: Tuple[int,int], drawing: bool):
        """Draw a ring at the fingertip position."""
        color = (0, 255, 180) if drawing else (180, 180, 180)
        cv2.circle(frame, pt, CURSOR_RADIUS, color, 2, cv2.LINE_AA)
        cv2.circle(frame, pt, 3, color, -1, cv2.LINE_AA)

    def draw_status(self, frame: np.ndarray, fps: float, mode_label: str,
                    hand_detected: bool, resolution: str):
        """Bottom status bar."""
        y = self.height - STATUS_BAR_HEIGHT
        cv2.rectangle(frame, (0, y), (self.width, self.height), (15, 15, 15), -1)

        hand_color = (0, 220, 100) if hand_detected else (80, 80, 80)
        hand_text  = "Hand: Detected" if hand_detected else "Hand: None"

        items = [
            (f"FPS: {fps:5.1f}",  (200, 200, 200)),
            (mode_label,           (0, 200, 255)),
            (hand_text,            hand_color),
            (resolution,           (150, 150, 150)),
        ]
        x = 12
        for text, color in items:
            cv2.putText(frame, text, (x, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
            x += 200

    # ── Button layout ─────────────────────────────────────────────────────────

    def _build_buttons(self) -> dict:
        """
        Returns a dict: { key: (x1, y1, x2, y2) }
        Keys: color names, 'brush_N', 'eraser', 'clear'
        """
        buttons = {}
        pad   = 8
        swatch = 44   # color swatch width
        sy    = TOOLBAR_HEIGHT // 2 - swatch // 2    # vertical center

        # Color swatches
        x = pad
        for name in COLORS:
            buttons[name] = (x, sy, x + swatch, sy + swatch)
            x += swatch + 4

        # Gap
        x += 8

        # Brush size buttons
        for i, size in enumerate(BRUSH_SIZES):
            w = 34
            buttons[f"brush_{size}"] = (x, sy, x + w, sy + swatch)
            x += w + 4

        x += 8

        # Eraser button
        buttons["eraser"] = (x, sy, x + 60, sy + swatch)
        x += 68

        # Clear button
        buttons["clear"] = (x, sy, x + 60, sy + swatch)

        return buttons

    def _is_selected(self, key: str) -> bool:
        if key in COLORS:
            return (not self._eraser_on) and (self._color == COLORS[key])
        if key.startswith("brush_"):
            size = int(key.split("_")[1])
            return (not self._eraser_on) and (self._brush_size == size)
        if key == "eraser":
            return self._eraser_on
        return False

    def _activate_button(self, key: str):
        if key in COLORS:
            self._color     = COLORS[key]
            self._eraser_on = False
        elif key.startswith("brush_"):
            self._brush_size = int(key.split("_")[1])
            self._eraser_on  = False
        elif key == "eraser":
            self._eraser_on = not self._eraser_on
        elif key == "clear":
            self._canvas[:] = 0

    def _draw_button(self, frame, key, x1, y1, x2, y2,
                     selected: bool, hovered: bool):
        """Render a single toolbar button."""
        radius = 6
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Selection glow
        if selected:
            _rounded_rect(frame, x1-3, y1-3, x2+3, y2+3, radius+2, (0, 220, 180), -1)
        elif hovered:
            _rounded_rect(frame, x1-2, y1-2, x2+2, y2+2, radius+1, (80, 80, 80), -1)

        # Color swatches
        if key in COLORS:
            bgr = COLORS[key]
            _rounded_rect(frame, x1, y1, x2, y2, radius, bgr, -1)
            _rounded_rect(frame, x1, y1, x2, y2, radius, (80,80,80), 1)

        # Brush size buttons
        elif key.startswith("brush_"):
            size = int(key.split("_")[1])
            _rounded_rect(frame, x1, y1, x2, y2, radius, (40, 40, 40), -1)
            dot_r = max(2, size // 2)
            cv2.circle(frame, (cx, cy), dot_r, (220, 220, 220), -1, cv2.LINE_AA)

        # Eraser
        elif key == "eraser":
            bg = (60, 50, 50) if not selected else (40, 40, 40)
            _rounded_rect(frame, x1, y1, x2, y2, radius, bg, -1)
            cv2.putText(frame, "Erase", (x1+4, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 180, 180), 1, cv2.LINE_AA)

        # Clear
        elif key == "clear":
            _rounded_rect(frame, x1, y1, x2, y2, radius, (50, 30, 30), -1)
            cv2.putText(frame, "Clear", (x1+6, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 220), 1, cv2.LINE_AA)
