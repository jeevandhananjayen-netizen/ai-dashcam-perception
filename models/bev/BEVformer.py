import cv2
import numpy as np
import math

PI = 3.1415926

frameWidth = 640
frameHeight = 480

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Real-world reference: average car width ~1.8 meters, lane width ~3.5 meters
REAL_LANE_WIDTH_M = 3.5
LANE_WIDTH_PX_BEV = 200  # Approximate pixel width of lane in BEV (tune this)
PIXELS_PER_METER = LANE_WIDTH_PX_BEV / REAL_LANE_WIDTH_M


def detect_cars(image):
    """Returns list of (x, y, w, h) bounding boxes for cars."""
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences = [], []
    h, w = image.shape[:2]

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "car":
                cx, cy = int(detection[0] * w), int(detection[1] * h)
                bw, bh = int(detection[2] * w), int(detection[3] * h)
                x, y = cx - bw // 2, cy - bh // 2
                boxes.append([x, y, bw, bh])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    final_boxes = [boxes[i] for i in indices.flatten()] if len(indices) > 0 else []
    return final_boxes


def compute_bev_transform(focalLength, dist, alpha, beta, gamma, w, h):
    A1 = np.array([[1, 0, -w / 2],
                   [0, 1, -h / 2],
                   [0, 0, 0],
                   [0, 0, 1]], dtype=np.float32)

    RX = np.array([[1, 0, 0, 0],
                   [0, math.cos(alpha), -math.sin(alpha), 0],
                   [0, math.sin(alpha), math.cos(alpha), 0],
                   [0, 0, 0, 1]], dtype=np.float32)

    RY = np.array([[math.cos(beta), 0, -math.sin(beta), 0],
                   [0, 1, 0, 0],
                   [math.sin(beta), 0, math.cos(beta), 0],
                   [0, 0, 0, 1]], dtype=np.float32)

    RZ = np.array([[math.cos(gamma), -math.sin(gamma), 0, 0],
                   [math.sin(gamma), math.cos(gamma), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype=np.float32)

    R = np.dot(np.dot(RX, RY), RZ)

    T = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dist],
                  [0, 0, 0, 1]], dtype=np.float32)

    K = np.array([[focalLength, 0, w / 2, 0],
                  [0, focalLength, h / 2, 0],
                  [0, 0, 1, 0]], dtype=np.float32)

    return np.dot(np.dot(np.dot(K, T), R), A1)


def update_perspective(val):
    alpha = (cv2.getTrackbarPos("Alpha", "Result") - 90) * PI / 180
    beta  = (cv2.getTrackbarPos("Beta",  "Result") - 90) * PI / 180
    gamma = (cv2.getTrackbarPos("Gamma", "Result") - 90) * PI / 180
    focalLength = cv2.getTrackbarPos("f", "Result")
    dist = cv2.getTrackbarPos("Distance", "Result")

    if focalLength == 0:
        return

    w, h = frameWidth, frameHeight
    image_size = (w, h)

    transformationMat = compute_bev_transform(focalLength, dist, alpha, beta, gamma, w, h)

    # Apply BEV warp
    bev = cv2.warpPerspective(source, transformationMat, image_size,
                              flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)

    # Detect cars in original image
    boxes = detect_cars(source)

    # Ego car bottom-center point (your car's position in BEV)
    ego_pt = np.array([[[w / 2, h - 10]]], dtype=np.float32)
    ego_bev = cv2.perspectiveTransform(ego_pt, transformationMat)
    ego_x, ego_y = int(ego_bev[0][0][0]), int(ego_bev[0][0][1])

    for (bx, by, bw, bh) in boxes:
        # Bottom-center of detected car bounding box (feet of car on road)
        car_bottom_center = np.array([[[bx + bw / 2, by + bh]]], dtype=np.float32)

        # Transform that point to BEV space
        car_bev_pt = cv2.perspectiveTransform(car_bottom_center, transformationMat)
        car_x, car_y = int(car_bev_pt[0][0][0]), int(car_bev_pt[0][0][1])

        # Pixel distance in BEV → real-world meters
        pixel_dist = math.sqrt((car_x - ego_x) ** 2 + (car_y - ego_y) ** 2)
        real_dist_m = pixel_dist / PIXELS_PER_METER

        # Draw on BEV image
        cv2.circle(bev, (car_x, car_y), 6, (0, 165, 255), -1)  # Orange dot = detected car
        cv2.circle(bev, (ego_x, ego_y), 6, (0, 255, 0), -1)    # Green dot = your car
        cv2.line(bev, (ego_x, ego_y), (car_x, car_y), (0, 165, 255), 2)

        label = f"{real_dist_m:.1f} m"
        mid_x = (ego_x + car_x) // 2
        mid_y = (ego_y + car_y) // 2
        cv2.putText(bev, label, (mid_x + 5, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.imshow("Result", bev)


# ── Main ──────────────────────────────────────────────────────────────────────
source = cv2.imread('D:\Pdf Files\IIITDM Files\Sem 6\Multimedia\Project\Images\Dash Cam Footage.jpg')
if source is None:
    raise FileNotFoundError("frame.jpg not found! Place it in the same folder.")

source = cv2.resize(source, (frameWidth, frameHeight))

cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

cv2.createTrackbar("Alpha",    "Result",  90,  180, update_perspective)
cv2.createTrackbar("Beta",     "Result",  90,  180, update_perspective)
cv2.createTrackbar("Gamma",    "Result",  90,  180, update_perspective)
cv2.createTrackbar("f",        "Result", 500, 2000, update_perspective)
cv2.createTrackbar("Distance", "Result", 500, 2000, update_perspective)

cv2.waitKey(100)  # Let trackbars initialize
update_perspective(0)
cv2.waitKey(0)
cv2.destroyAllWindows()