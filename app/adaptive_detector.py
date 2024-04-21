import cv2
import math
import numpy as np
from dataclasses import dataclass
from typing import List, NamedTuple, Optional


@dataclass
class FrameData:
    """Data calculated for a given frame."""

    hue: np.ndarray
    """Frame hue map [2D 8-bit]."""
    sat: np.ndarray
    """Frame saturation map [2D 8-bit]."""
    lum: np.ndarray
    """Frame luma/brightness map [2D 8-bit]."""
    edges: Optional[np.ndarray]
    """Frame edge map [2D 8-bit, edges are 255, non-edges 0]."""


class Components(NamedTuple):
    """Components that make up a frame's score."""

    delta_hue: float = 1.0
    delta_sat: float = 1.0
    delta_lum: float = 1.0
    delta_edges: float = 0.0


class AdaptiveDetector:
    """Two-pass detector that calculates frame scores and applies a rolling average for cuts."""

    def __init__(
        self,
        adaptive_threshold: float = 3.0,
        min_scene_len: int = 15,
        window_width: int = 2,
        min_content_val: float = 15.0,
        weights: Components = Components(),
        kernel_size: Optional[int] = None,
    ):
        self.adaptive_threshold = adaptive_threshold
        self.min_scene_len = min_scene_len
        self.window_width = window_width
        self.min_content_val = min_content_val
        self.weights = weights
        self.kernel_size = kernel_size or _estimate_kernel_size(
            1920, 1080
        )  # Default 1080p
        self.kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        self.buffer = []
        self.last_cut = None
        self.last_frame = None
        self._first_frame_num = None

    def detect_edges(self, lum: np.ndarray) -> np.ndarray:
        """Detect edges using the luma channel of a frame."""
        sigma = 1.0 / 3.0
        median = np.median(lum)
        low = int(max(0, (1.0 - sigma) * median))
        high = int(min(255, (1.0 + sigma) * median))
        edges = cv2.Canny(lum, low, high)
        return cv2.dilate(edges, self.kernel)

    def _mean_pixel_distance(
        self, left: np.ndarray, right: np.ndarray
    ) -> float:
        """Calculate mean average distance between adjacent frames."""
        num_pixels = left.shape[0] * left.shape[1]
        return (
            np.sum(np.abs(left.astype(int) - right.astype(int))) / num_pixels
        )

    def _calculate_frame_score(
        self, frame_num: int, frame_img: np.ndarray
    ) -> float:
        """Calculate the score representing relative motion/content change."""
        hue, sat, lum = cv2.split(cv2.cvtColor(frame_img, cv2.COLOR_BGR2HSV))
        edges = (
            self.detect_edges(lum) if self.weights.delta_edges > 0 else None
        )

        if self.last_frame is None:
            self.last_frame = FrameData(hue, sat, lum, edges)
            return 0.0

        components = Components(
            delta_hue=self._mean_pixel_distance(hue, self.last_frame.hue),
            delta_sat=self._mean_pixel_distance(sat, self.last_frame.sat),
            delta_lum=self._mean_pixel_distance(lum, self.last_frame.lum),
            delta_edges=(
                0
                if edges is None
                else self._mean_pixel_distance(edges, self.last_frame.edges)
            ),
        )

        score = sum(c * w for c, w in zip(components, self.weights)) / sum(
            abs(w) for w in self.weights
        )

        self.last_frame = FrameData(hue, sat, lum, edges)
        return score

    def process_frame(
        self, frame_num: int, frame_img: Optional[np.ndarray]
    ) -> List[int]:
        """Process the next frame and detect scene cuts."""
        if self.last_cut is None:
            self.last_cut = frame_num

        frame_score = self._calculate_frame_score(frame_num, frame_img)
        self.buffer.append((frame_num, frame_score))
        self.buffer = self.buffer[-(2 * self.window_width + 1) :]

        if len(self.buffer) < (2 * self.window_width + 1):
            return []

        target = self.buffer[self.window_width]
        window_scores = [
            frame[1]
            for i, frame in enumerate(self.buffer)
            if i != self.window_width
        ]
        average_score = sum(window_scores) / len(window_scores)

        adaptive_ratio = (
            target[1] / average_score if average_score > 0.00001 else 255.0
        )

        if (
            adaptive_ratio >= self.adaptive_threshold
            and target[1] >= self.min_content_val
            and frame_num - self.last_cut >= self.min_scene_len
        ):
            self.last_cut = target[0]
            return [target[0]]

        return []


def _estimate_kernel_size(frame_width: int, frame_height: int) -> int:
    """Estimate kernel size based on video resolution."""
    size = 4 + round(math.sqrt(frame_width * frame_height) / 192)
    return size if size % 2 == 1 else size + 1
