"""
hand_tracker.py
---------------
Detects hand landmarks using MediaPipe and figures out
which gesture the user is making.

Gestures:
  Drawing  -> only index finger is up
  Select   -> index + middle fingers are up
  Idle     -> any other combination
"""
    
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

# ── Try both old and new MediaPipe APIs ──────────────────────────────────────
try:
    # New MediaPipe API (0.10.x+)
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import mediapipe as mp
    USE_NEW_API = True
except Exception:
    USE_NEW_API = False

if not USE_NEW_API:
    import mediapipe as mp


@dataclass
class HandInfo:
    """Everything we need to know about the detected hand."""
    index_tip: Tuple[int, int]
    is_drawing_gesture: bool   # 1 finger up  -> draw
    is_pause_gesture:   bool   # 2 fingers up -> select toolbar


class HandTracker:
    """
    Wraps MediaPipe Hands — works with both old and new MediaPipe APIs.
    """

    _TIP_IDS = [4, 8, 12, 16, 20]
    _PIP_IDS = [3, 6, 10, 14, 18]

    def __init__(self):
        # Try legacy solutions API first (mediapipe < 0.10)
        try:
            self._hands    = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.6,
            )
            self._mp_draw  = mp.solutions.drawing_utils
            self._mp_style = mp.solutions.drawing_styles
            self._mp_hands = mp.solutions.hands
            self._new_api  = False
            print("  MediaPipe: using legacy solutions API")
        except AttributeError:
            # New API path
            self._init_new_api()
            self._new_api = True
            print("  MediaPipe: using new tasks API")

    def _init_new_api(self):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        import urllib.request, os, tempfile

        model_path = os.path.join(tempfile.gettempdir(), "hand_landmarker.task")
        if not os.path.exists(model_path):
            print("  Downloading hand landmark model (~8 MB)...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("  Model downloaded.")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            running_mode=vision.RunningMode.VIDEO,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._mp = mp
        self._frame_ts = 0

    def process(self, frame: np.ndarray) -> Optional[HandInfo]:
        if self._new_api:
            return self._process_new(frame)
        return self._process_legacy(frame)

    def _process_legacy(self, frame: np.ndarray) -> Optional[HandInfo]:
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        landmarks = results.multi_hand_landmarks[0]
        self._mp_draw.draw_landmarks(
            frame, landmarks,
            self._mp_hands.HAND_CONNECTIONS,
            self._mp_style.get_default_hand_landmarks_style(),
            self._mp_style.get_default_hand_connections_style(),
        )

        h, w = frame.shape[:2]
        fingers_up = self._count_fingers_legacy(landmarks, w, h)
        index_tip  = self._get_tip_legacy(landmarks, 8, w, h)

        return HandInfo(
            index_tip          = index_tip,
            is_drawing_gesture = (fingers_up == [0, 1, 0, 0, 0]),
            is_pause_gesture   = (fingers_up == [0, 1, 1, 0, 0]),
        )

    def _process_new(self, frame: np.ndarray) -> Optional[HandInfo]:
        import mediapipe as mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts += 33
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        if not result.hand_landmarks:
            return None

        landmarks = result.hand_landmarks[0]
        h, w = frame.shape[:2]

        # Draw skeleton manually
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 200, 100), -1)

        fingers_up = self._count_fingers_new(landmarks)
        tip = (int(landmarks[8].x * w), int(landmarks[8].y * h))

        return HandInfo(
            index_tip          = tip,
            is_drawing_gesture = (fingers_up == [0, 1, 0, 0, 0]),
            is_pause_gesture   = (fingers_up == [0, 1, 1, 0, 0]),
        )

    def close(self):
        if self._new_api:
            self._landmarker.close()
        else:
            self._hands.close()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_tip_legacy(self, lm, tip_id, w, h):
        pt = lm.landmark[tip_id]
        return int(pt.x * w), int(pt.y * h)

    def _count_fingers_legacy(self, lm, w, h):
        fingers = []
        thumb_tip = lm.landmark[self._TIP_IDS[0]]
        thumb_pip = lm.landmark[self._PIP_IDS[0]]
        fingers.append(1 if thumb_tip.x < thumb_pip.x else 0)
        for tip_id, pip_id in zip(self._TIP_IDS[1:], self._PIP_IDS[1:]):
            tip = lm.landmark[tip_id]
            pip = lm.landmark[pip_id]
            fingers.append(1 if tip.y < pip.y else 0)
        return fingers

    def _count_fingers_new(self, landmarks):
        fingers = []
        tip = landmarks[self._TIP_IDS[0]]
        pip = landmarks[self._PIP_IDS[0]]
        fingers.append(1 if tip.x < pip.x else 0)
        for tip_id, pip_id in zip(self._TIP_IDS[1:], self._PIP_IDS[1:]):
            fingers.append(1 if landmarks[tip_id].y < landmarks[pip_id].y else 0)
        return fingers
