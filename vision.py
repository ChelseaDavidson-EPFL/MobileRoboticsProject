import numpy as np
import cv2

"""
Setup notes:
- ID 1 must go bottom left
- ID 2 must go top right
"""

class Vision:
    def __init__(self):
        self.arenaMarkerSize = 0.02 #2cm
        self.arenaMarkerDict = cv2.aruco.DICT_4X4_50
        self.robotMarkerDict = cv2.aruco.DICT_5X5_50

        # Arena information
        self.arena_corners_pixels = None
        self.arena_width_m = None
        self.arena_height_m = None

        self.meters_per_pixel = None
        self.grid = None

    def getEnvironment(self):
         # Capture a single frame from webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not access the webcam.")
            exit()

        waitNumber = 30
        waitIndx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera failed to capture the frame.")
                break

            waitIndx += 1
            if (waitIndx < waitNumber): #TODO - Make this nicer
                continue
            
            # ----------------------------
            # Detect arena corners
            # ----------------------------
            self.arena_corners_pixels, self.arena_width_m, self.arena_height_m, self.meters_per_pixel = self.getArenaCornerPixelsAndRealArenaSize(frame)
            if (self.arena_corners_pixels == None or self.arena_width_m == None or self.arena_height_m == None or self.meters_per_pixel == None):
                print("Couldn't locate arena")
                continue

            # ----------------------------
            # 2. Detect robot center
            # ----------------------------
            robot_cam_x, robot_cam_y, robot_heading_angle, robot_marker_corners = self.getRobotPoseCameraFrame(frame)

            if (robot_cam_x is None or robot_cam_y is None or robot_heading_angle is None or robot_marker_corners is None):
                print("Couldn't locate robot")
                continue

            # Convert to global frame
            X, Y = self.getGlobalLocation(robot_cam_x, robot_cam_y)

            # ----------------------------
            # 4. Find obstacles
            # ----------------------------
            polygons = self.locateObstaclesRed(frame)
            global_polygons = self.convertPolygonsToWorld(polygons)

            # ----------------------------
            # 5. Create and store occupancy grid
            # ----------------------------
            self.createGrid(X, Y, self.arena_width_m, self.arena_height_m, global_polygons)
 
            # ----------------------------
            # 6. Visualisations
            # ----------------------------
            vis = frame.copy()

            self.visualiseArena(vis, self.arena_corners_pixels)
            self.visualiseRobotPose(vis, robot_marker_corners, robot_cam_x, robot_cam_y, robot_heading_angle, X, Y)
            self.visualiseObstacles(vis, polygons)
            break
        
        # Show the result
        cv2.imshow("Initial Environment", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def createGrid(self, robot_x, robot_y, arena_w, arena_h, obstacle_polygons):
        """
        Creates a grid in world frame coordinates where the bottom left corner of the arena is 0,0 so the top right corner 
        will be arena_w, arena_h. Both the robot position and obstacle_polygons are relative to this 0,0 frame
        """
        self.grid = None # TODO

    def getGrid(self):
        return self.grid

    def getArenaCornerPixelsAndRealArenaSize(self, image):
        """
        Detects ArUco markers:
            ID 1 = Bottom-left (BL)
            ID 2 = Top-right (TR)

        Returns:
            pixel_corners = [BL, BR, TR, TL]
            arena_width_m
            arena_height_m
        """

        # Dictionary for aruco markers
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.arenaMarkerDict)
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

        # Convert to gray scale for better detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            print("No ArUco markers detected")
            return None, None, None, None

        ids = ids.flatten()

        # Must detect both markers
        if not (1 in ids and 2 in ids):
            print(f"Missing arena markers: found {ids}, expected IDs 1 and 2")
            return None, None, None, None

        # Find pixel centers of markers
        idx_bl = list(ids).index(1)
        idx_tr = list(ids).index(2)

        corners_bl = corners[idx_bl][0]  # 4 corner points (TL,TR,BR,BL)
        corners_tr = corners[idx_tr][0]

        # Pixel center of BL marker
        bl_px = (int(corners_bl[:,0].mean()), int(corners_bl[:,1].mean()))
        # Pixel center of TR marker
        tr_px = (int(corners_tr[:,0].mean()), int(corners_tr[:,1].mean()))

        # Build rectangle:
        x1, y1 = bl_px
        x2, y2 = tr_px
        br_px = (x2, y1)
        tl_px = (x1, y2)

        pixel_corners = [bl_px, br_px, tr_px, tl_px]

        # Compute meters per pixel

        # Marker width in pixels (use top-left -> top-right edge)
        # For BL marker (ID 1)
        bl_marker_width_px = np.linalg.norm(corners_bl[1] - corners_bl[0])  # TR - TL

        if bl_marker_width_px < 1:
            print("Invalid marker pixel width")
            return None, None, None

        meters_per_pixel = self.arenaMarkerSize / bl_marker_width_px

        # Convert arena pixel distances to meters
        arena_width_pixels  = abs(x2 - x1)
        arena_height_pixels = abs(y2 - y1)

        arena_width_m  = arena_width_pixels  * meters_per_pixel
        arena_height_m = arena_height_pixels * meters_per_pixel

        return pixel_corners, arena_width_m, arena_height_m, meters_per_pixel
    
    def getRobotPose(self, image):
        # Get robot pose in camera frame
        robot_cam_x, robot_cam_y, robot_heading_angle, robot_marker_corners = self.getRobotPoseCameraFrame(image)

        if (robot_cam_x is None or robot_cam_y is None or robot_heading_angle is None or robot_marker_corners is None):
            print("Couldn't locate robot so exiting getRobotPose function")
            return None, None, None

        # Convert to global coordinates
        X, Y = self.getGlobalLocation(robot_cam_x, robot_cam_y)

        return X, Y, robot_heading_angle
    
    def getRobotPoseAndVisualise(self, image, vis):
        # Get robot pose in camera frame
        robot_cam_x, robot_cam_y, robot_heading_angle, robot_marker_corners = self.getRobotPoseCameraFrame(image)

        if (robot_cam_x is None or robot_cam_y is None or robot_heading_angle is None or robot_marker_corners is None):
            print("Couldn't locate robot so exiting geRobottPoseAndVisualise function")
            return None, None, None

        # Convert to global coordinates
        X, Y = self.getGlobalLocation(robot_cam_x, robot_cam_y)

        # Visualise
        self.visualiseRobotPose(vis, robot_marker_corners, robot_cam_x, robot_cam_y, robot_heading_angle, X, Y)

        return X, Y, robot_heading_angle

        
    def getRobotPoseCameraFrame(self, image):
        """
        Detects ONLY ArUco marker with ID 0.
        Returns:
        - center pixel (cx, cy)
        - heading angle in radians
        - 4 corner points (TL, TR, BR, BL)
        """

        # Dictionary for aruco markers
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.robotMarkerDict)
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

        # Convert to gray scale for better detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        # No markers at all
        if ids is None:
            return None, None, None, None

        # Convert ids to a simple list
        ids = ids.flatten()

        # Check if ID 0 exists
        if 0 not in ids:
            return None, None, None, None

        # Find index of ID 0
        idx = list(ids).index(0)

        # Get its corners
        robotCorners = corners[idx][0]  # shape (4,2)
        tl, tr, br, bl = robotCorners

        # Find center pixel of the marker
        cx = int(robotCorners[:, 0].mean())
        cy = int(robotCorners[:, 1].mean())

        # Heading direction vector (TR - TL)
        dir_vec = tr - tl
        heading_angle = np.arctan2(-dir_vec[1], dir_vec[0])  # (radians) NOTE: do negative of y since camera y is top bottom not bottom top 

        return cx, cy, heading_angle, robotCorners

    def computeHomography(self, arena_corners_pixels, arena_width_m, arena_height_m):
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


    def pixelToGlobal(self, H, pixel_point):
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


    def getGlobalLocation(self, marker_cam_x, marker_cam_y):
        H = self.computeHomography(self.arena_corners_pixels, self.arena_width_m, self.arena_height_m)
        return self.pixelToGlobal(H, (marker_cam_x, marker_cam_y))
    
    def convertPolygonsToWorld(self, polygons):
        """
        Convert a list of pixel-based polygons into world coordinates.
        Returns a list of polygons in meters.
        """
        # Compute homography
        H = self.computeHomography(self.arena_corners_pixels, self.arena_width_m, self.arena_height_m)

        world_polygons = []

        for poly in polygons:
            world_poly = []
            for (u, v) in poly:
                X, Y = self.pixelToGlobal(H, (u, v))
                world_poly.append((X, Y))
            world_polygons.append(world_poly)

        return world_polygons

    def locateObstaclesRed(self, image):
        """
        Detect red blobs in an image, approximate their shapes as polygons,
        and return a list of polygons (each polygon is a list of (x,y) points).
        Handles both HSV red regions (0–10° and 170–180°).
        """

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # --- Red spans the start and end of the hue circle ---
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        # Create red masks
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []

        for cnt in contours:
            # Filter out small artifacts
            if cv2.contourArea(cnt) < 100:
                continue

            # Approximate to polygon
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            poly = cv2.approxPolyDP(cnt, epsilon, True)

            # Convert polygon format for output
            polygon_points = [(int(p[0][0]), int(p[0][1])) for p in poly]
            polygons.append(polygon_points)

        return polygons

    def visualiseArena(self, vis, arena_corners_pixels):
        # Draw arena outline (green polygon)
        pts = np.array(arena_corners_pixels, dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

        # Draw arena corner names
        labels = ["BL", "BR", "TR", "TL"]
        for (x, y), name in zip(arena_corners_pixels, labels):
            cv2.putText(vis, name, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def visualiseRobotPose(self, vis, robot_marker_corners, robot_center_cam_x, robot_center_cam_y, robot_heading_angle, robot_center_world_x, robot_center_world_y):
         # Draw ArUco marker location
        int_corners = robot_marker_corners.astype(np.int32)
        cv2.polylines(vis, [int_corners], True, (255, 0, 0), 3)

        # Draw center point
        cv2.circle(vis, (robot_center_cam_x, robot_center_cam_y), 6, (0, 0, 255), -1)

        # Draw heading vector (scaled)
        arrow_len = 50
        # NOTE: Need to visualise the negative of the angle since camera y direction is top bottom not bottom top
        dx = int(np.cos(-robot_heading_angle) * arrow_len)
        dy = int(np.sin(-robot_heading_angle) * arrow_len)

        cv2.arrowedLine(
            vis,
            (robot_center_cam_x, robot_center_cam_y),
            (robot_center_cam_x + dx, robot_center_cam_y + dy),
            (0, 255, 255),
            3,
            tipLength=0.3
        )

        # Display heading in degrees
        heading_deg = np.degrees(robot_heading_angle)
        cv2.putText(vis, f"Heading: {heading_deg:.1f} deg",
                    (robot_center_cam_x + 20, robot_center_cam_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Label with global coordinate
        coord_text = f"({robot_center_world_x:.5f}m, {robot_center_world_y:.5f}m)"
        cv2.putText(vis, coord_text, (robot_center_cam_x + 15, robot_center_cam_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def visualiseObstacles(self, vis, polygons):
        # Outline obstacles
        for poly in polygons:
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 0, 255), thickness=2)



if __name__ == "__main__":
    visionInstance = Vision()

    # Get the initial environment and saves it to class variables (detect arena, compute homography, get obstacles)
    visionInstance.getEnvironment()

    # Now start the live robot pose visualisation
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera failed to capture the frame.")
            break

        vis = frame.copy()

        # --- Always redraw arena outline ---
        visionInstance.visualiseArena(vis, visionInstance.arena_corners_pixels)

        # --- Always detect and draw robot pose ---
        X, Y, robot_heading_angle = visionInstance.getRobotPoseAndVisualise(frame, vis)
        if (X is None or Y is None or robot_heading_angle is None):
            continue
        print(f"X: {X:.5f}, Y: {Y:.5f}, Direction: {robot_heading_angle:.5f}")

        # --- Show the live window ---
        cv2.imshow("Live Robot Pose", vis)

        # Exit on Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
