import cv2
import numpy as np

def detect_ego_lane(frame):
    """
    Detects the ego-lane using classical computer vision.
    Returns the estimated x-coordinate of the lane center at the horizon/vanishing point,
    and the frame with drawn lane boundaries for debugging.
    """
    height, width = frame.shape[:2]
    
    # Define Region of Interest (ROI) - bottom half of the image
    roi_vertices = np.array([
        [(0, height), (width, height), (width, height // 2), (0, height // 2)]
    ], dtype=np.int32)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Hough Line Transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=40, maxLineGap=150)
    
    left_lines = []
    right_lines = []
    
    lane_frame = frame.copy()
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            # Filter lines by slope to find distinct left and right lanes
            if slope < -0.5 and x1 < width / 2 and x2 < width / 2:
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.5 and x1 > width / 2 and x2 > width / 2:
                right_lines.append((x1, y1, x2, y2))
                
    # A simple approach to find the center is taking the average x of left and right lines
    # near the horizon (height/2)
    horizon_y = height // 2 + 50
    left_x_horizon = 0
    right_x_horizon = width
    
    if left_lines:
        avg_left = np.mean(left_lines, axis=0)
        x1, y1, x2, y2 = avg_left
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else -0.1
        intercept = y1 - slope * x1
        left_x_horizon = int((horizon_y - intercept) / slope)
        cv2.line(lane_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 5)
        
    if right_lines:
        avg_right = np.mean(right_lines, axis=0)
        x1, y1, x2, y2 = avg_right
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0.1
        intercept = y1 - slope * x1
        right_x_horizon = int((horizon_y - intercept) / slope)
        cv2.line(lane_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
        
    lane_center_x = (left_x_horizon + right_x_horizon) // 2
    
    # Draw center line indicator (Vanishing Point projection)
    cv2.circle(lane_frame, (lane_center_x, horizon_y), 10, (0, 255, 255), -1)
    cv2.line(lane_frame, (lane_center_x, horizon_y), (lane_center_x, height), (0, 255, 255), 2)
    
    return lane_center_x, lane_frame
