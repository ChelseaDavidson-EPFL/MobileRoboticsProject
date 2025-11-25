import numpy as np
import cv2

# -------------------------------------------
#        HOMOGRAPHY + GLOBAL MAPPING
# -------------------------------------------

def computeHomography(arena_corners_pixels, arena_width_m, arena_height_m):
    """
    arena_corners_pixels: list of 4 (u,v) pixel points in the order:
        bottom-left, bottom-right, top-right, top-left
    arena_width_m: real arena width in meters
    arena_height_m: real arena height in meters
    """

    # Real-world coordinates of the arena corners (meters)
    world_corners = np.array([
        [0, 0],                          # Bottom-left
        [arena_width_m, 0],              # Bottom-right
        [arena_width_m, arena_height_m], # Top-right
        [0, arena_height_m]              # Top-left
    ], dtype=float)

    img_pts = np.array(arena_corners_pixels, dtype=float)
    world_pts = world_corners.reshape(-1, 1, 2)

    H, _ = cv2.findHomography(img_pts, world_pts)
    return H


def pixelToGlobal(H, pixel_point):
    """
    Convert a pixel point (u,v) into global (X,Y) using homography H.
    """
    u, v = pixel_point
    pt = np.array([u, v, 1.0])
    world = H @ pt
    world /= world[2]  # Normalize

    X = float(world[0])
    Y = float(world[1])

    return X, Y


def getGlobalLocation(arena_corners_pixels, arena_width_m, arena_height_m,
                      marker_cam_x, marker_cam_y):

    H = computeHomography(arena_corners_pixels, arena_width_m, arena_height_m)
    return pixelToGlobal(H, (marker_cam_x, marker_cam_y))


# -------------------------------------------
#        ARENA CORNER DETECTION
# -------------------------------------------

def getArenaCornerPixels(image):
    """
    Finds 4 red arena corner markers using HSV red thresholding.
    Returns (bl, br, tr, tl) in pixel coordinate order.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red thresholds
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2
    mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:  # ignore small noise
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            red_centers.append((cx, cy))

    if len(red_centers) != 4:
        print(f"Expected 4 arena corners, but found {len(red_centers)}")
        return [0, 0, 0, 0]

    # Sort into TL, TR, BL, BR structure
    red_centers = sorted(red_centers, key=lambda c: (c[1], c[0]))  # sort by row, then column
    top = sorted(red_centers[:2], key=lambda c: c[0])
    bottom = sorted(red_centers[2:], key=lambda c: c[0])

    tl, tr = top
    bl, br = bottom

    # Return in required order: bl, br, tr, tl
    return [bl, br, tr, tl]


# -------------------------------------------
#        ARUCO MARKER DETECTION
# -------------------------------------------

def getArucoLocationCameraFrame(cameraOrigin, image):
    """
    Detects first ArUco marker and returns:
    - center pixel (cx, cy)
    - heading angle in radians
    - 4 corner points
    """

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return 0, 0, 0, None

    # Use first marker
    c = corners[0][0]  # shape (4,2)
    # Corner order: TL, TR, BR, BL in OpenCV ArUco
    tl, tr, br, bl = c

    # Compute center
    cx = int(c[:, 0].mean())
    cy = int(c[:, 1].mean())

    # Compute heading direction vector: TR - TL
    dir_vec = tr - tl
    heading_angle = np.arctan2(dir_vec[1], dir_vec[0])  # radians

    return cx, cy, heading_angle, c



# -------------------------------------------
#          MAIN LOCALIZATION PIPELINE
# -------------------------------------------

def locateRobot(arena_width_m, arena_height_m):
    # Capture a single frame from webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera failed to capture the frame.")
            break

        # ----------------------------
        # 1. Detect arena corners
        # ----------------------------
        arena_corners_pixels = getArenaCornerPixels(frame)

        if (arena_corners_pixels == [0,0,0,0]):
            continue

        # ----------------------------
        # 2. Detect ArUco marker center
        # ----------------------------
        marker_cam_x, marker_cam_y, heading_angle, marker_corners = getArucoLocationCameraFrame(arena_corners_pixels[0], frame)

        if marker_corners is None:
            continue



        # ----------------------------
        # 3. Convert to global coordinates
        # ----------------------------
        X, Y = getGlobalLocation(
            arena_corners_pixels,
            arena_width_m,
            arena_height_m,
            marker_cam_x,
            marker_cam_y
        )

        print(f"Global position of the robot: X={X:.3f} m, Y={Y:.3f} m")

        # ----------------------------------------------------
        #           VISUALIZATION ON THE IMAGE
        # ----------------------------------------------------
        vis = frame.copy()

        # Draw arena outline (green polygon)
        pts = np.array(arena_corners_pixels, dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

        # Draw arena corner names
        labels = ["BL", "BR", "TR", "TL"]
        for (x, y), name in zip(arena_corners_pixels, labels):
            cv2.putText(vis, name, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw ArUco marker location
        int_corners = marker_corners.astype(np.int32)
        cv2.polylines(vis, [int_corners], True, (255, 0, 0), 3)

        # Draw center point
        cv2.circle(vis, (marker_cam_x, marker_cam_y), 6, (0, 0, 255), -1)

        # Draw heading vector (scaled)
        arrow_len = 50
        dx = int(np.cos(heading_angle) * arrow_len)
        dy = int(np.sin(heading_angle) * arrow_len)

        cv2.arrowedLine(
            vis,
            (marker_cam_x, marker_cam_y),
            (marker_cam_x + dx, marker_cam_y + dy),
            (0, 255, 255),
            3,
            tipLength=0.3
        )

        # Display heading in degrees
        heading_deg = np.degrees(heading_angle) * -1
        cv2.putText(vis, f"Heading: {heading_deg:.1f} deg",
                    (marker_cam_x + 20, marker_cam_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Label with global coordinate
        coord_text = f"({X:.2f}m, {Y:.2f}m)"
        cv2.putText(vis, coord_text, (marker_cam_x + 15, marker_cam_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show the result
        cv2.imshow("Arena + Robot Position", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return X, Y, heading_angle


if __name__ == "__main__":
    locateRobot(0.26, 0.26)