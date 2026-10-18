import cv2
import numpy as np

MIN_SLOPE = 0.5


def _x_at(lines, y):
    """Averages the segments of one lane border and returns its x at height y."""
    x1, y1, x2, y2 = np.mean(lines, axis=0)
    if x2 == x1:
        return x1
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return (y - intercept) / slope


def detect_ego_lane(frame, roi_bottom=1.0):
    """
    Detects the ego-lane using classical computer vision (Canny + Hough).

    roi_bottom is the fraction of the height where the search region ends, so the
    car hood/dashboard at the bottom of dashcam footage can be excluded.

    Returns (lane_center_x, reference_y, status):
      - lane_center_x: x of the lane center at reference_y, or None when it cannot
        be estimated;
      - status: 'ok' (both borders found), 'partial' (only one border, the lane
        width is unknown so no center is estimated), 'none' (no border found) or
        'invalid' (borders found but geometrically inconsistent).
    """
    height, width = frame.shape[:2]
    roi_top = height // 2
    roi_end = int(height * roi_bottom)
    reference_y = roi_top + 50

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    mask = np.zeros_like(edges)
    mask[roi_top:roi_end, :] = 255
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=40, maxLineGap=150)

    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            # Filter lines by slope to find distinct left and right lanes
            if slope < -MIN_SLOPE and x1 < width / 2 and x2 < width / 2:
                left_lines.append((x1, y1, x2, y2))
            elif slope > MIN_SLOPE and x1 > width / 2 and x2 > width / 2:
                right_lines.append((x1, y1, x2, y2))

    if not left_lines and not right_lines:
        return None, reference_y, 'none'
    if not left_lines or not right_lines:
        return None, reference_y, 'partial'

    left_x = _x_at(left_lines, reference_y)
    right_x = _x_at(right_lines, reference_y)
    lane_center_x = int((left_x + right_x) / 2)
    if left_x >= right_x or not 0 <= lane_center_x < width:
        return None, reference_y, 'invalid'

    return lane_center_x, reference_y, 'ok'
