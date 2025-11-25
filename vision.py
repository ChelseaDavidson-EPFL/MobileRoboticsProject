import numpy as np
import cv2

MARKER_SIZE_M = 0.05   # 5 cm marker size


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

def getArenaCornerPixelsAndRealArenaSize(image, marker_size_m=MARKER_SIZE_M):
    """
    Detects ArUco markers:
        ID 1 = Bottom-left (BL)
        ID 2 = Top-right (TR)

    Returns:
        pixel_corners = [BL, BR, TR, TL]
        arena_width_m
        arena_height_m
    """

    # Dictionary for 4x4 markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(
        aruco_dict,
        cv2.aruco.DetectorParameters()
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        print("No ArUco markers detected")
        return None, None, None

    ids = ids.flatten()

    # Must detect both markers
    if not (1 in ids and 2 in ids):
        print(f"Missing arena markers: found {ids}, expected IDs 1 and 2")
        return None, None, None

    # ---------------------------
    # Extract pixel centers
    # ---------------------------
    idx_bl = list(ids).index(1)
    idx_tr = list(ids).index(2)

    c_bl = corners[idx_bl][0]  # 4 corner points (TL,TR,BR,BL)
    c_tr = corners[idx_tr][0]

    # Pixel center of BL marker
    bl_px = (int(c_bl[:,0].mean()), int(c_bl[:,1].mean()))
    # Pixel center of TR marker
    tr_px = (int(c_tr[:,0].mean()), int(c_tr[:,1].mean()))

    # Build rectangle:
    x1, y1 = bl_px
    x2, y2 = tr_px
    br_px = (x2, y1)
    tl_px = (x1, y2)

    pixel_corners = [bl_px, br_px, tr_px, tl_px]

    # ---------------------------
    # Compute meters per pixel
    # ---------------------------

    # Marker width in pixels (use top-left -> top-right edge)
    # For BL marker (ID 1)
    bl_marker_width_px = np.linalg.norm(c_bl[1] - c_bl[0])  # TR - TL

    if bl_marker_width_px < 1:
        print("Invalid marker pixel width")
        return None, None, None

    meters_per_pixel = marker_size_m / bl_marker_width_px

    # ---------------------------
    # Convert arena pixel distances to meters
    # ---------------------------
    arena_width_pixels  = abs(x2 - x1)
    arena_height_pixels = abs(y2 - y1)

    arena_width_m  = arena_width_pixels  * meters_per_pixel
    arena_height_m = arena_height_pixels * meters_per_pixel

    return pixel_corners, arena_width_m, arena_height_m



# -------------------------------------------
#        ARUCO MARKER DETECTION
# -------------------------------------------

def getArucoLocationCameraFrame(cameraOrigin, image):
    """
    Detects ONLY ArUco marker with ID 0.
    Returns:
    - center pixel (cx, cy)
    - heading angle in radians
    - 4 corner points (TL, TR, BR, BL)
    """

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    # No markers at all
    if ids is None:
        return 0, 0, 0, None

    # Convert ids to a simple list
    ids = ids.flatten()

    # Check if ID 0 exists
    if 0 not in ids:
        return 0, 0, 0, None

    # Find index of ID 0
    idx = list(ids).index(0)

    # Get its corners
    c = corners[idx][0]  # shape (4,2)
    tl, tr, br, bl = c

    # Center pixel of the marker
    cx = int(c[:, 0].mean())
    cy = int(c[:, 1].mean())

    # Heading direction vector (TR - TL)
    dir_vec = tr - tl
    heading_angle = np.arctan2(dir_vec[1], dir_vec[0])  # in radians

    return cx, cy, heading_angle, c



# -------------------------------------------
#          MAIN LOCALIZATION PIPELINE
# -------------------------------------------

def locateRobot():
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
        arena_corners_pixels, arena_width_m, arena_height_m = getArenaCornerPixelsAndRealArenaSize(frame)

        if (arena_corners_pixels == [0,0,0,0] or arena_width_m == None or arena_width_m == None):
            continue

        print(f"Arena width: {arena_width_m:.5f} m, Arena height: {arena_width_m:.5f}")

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

        return X, Y, heading_deg


if __name__ == "__main__":
    X, Y, angle = locateRobot()
    print(f"Global position of the robot: X={X:.3f} m, Y={Y:.3f} m, heading={angle:.3f} degs")
