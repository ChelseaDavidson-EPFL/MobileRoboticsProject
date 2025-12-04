import numpy as np
import cv2
import sys
from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import utils

"""
Owner: Chelsea Davidson

Setup notes:
- ID 1 must go bottom left
- ID 2 must go top right

Usage notes:
- Heading angle is in rads between (-pi, pi] where east is 0 rads
- Real positions are in meters
- Uses a Windows only backend 
"""

class Vision:
    def __init__(self):
        # Aruco marker information
        self.arenaMarkerSize = 0.03 #3cm
        self.arenaMarkerDict = cv2.aruco.DICT_4X4_50  # dictionary names contain the grid size of the marker (4×4, 5×5...) and the number of possible unique IDs
        self.robotMarkerDict = cv2.aruco.DICT_5X5_50
        self.goalMarkerDict = cv2.aruco.DICT_5X5_50

        # Arena information (stored during getEnvironment)
        self.arena_corners_pixels = None
        self.arena_width_m = None
        self.arena_height_m = None

        # Grid information (stored during getEnvironment)
        self.grid_dim = utils.GRID_DIM
        self.grid = None
        self.cell_size_cm = None

        # Robot initial pose (stored during getEnvironment)
        self.initial_robot_pos = None  # (X, Y) in meters relative to left bottom corner of arena
        self.initial_robot_orient = None  # heading in radians

        # Goal position (stored during getEnvironment)
        self.goal_pos = None # (x, y) in meters relative to left bottom corner of arena
        self.goal_cam_pos = None # (x, y) in camera frame
        self.goal_marker_corners = None # corners in camera frame

        # Store video capture
        self.cap = None
        self.getEnvironment()

    def getEnvironment(self):
        """
        Locates and visualises all features in the environment (arena, start pose, goal, global obstacles) and creates a grid to store this information in. 
        """
        # Capture a single frame from webcam - VideoCapture(1) opens camera device 1 on your computer
        # Use DirectShow backend on Windows for faster initialization (cv2.CAP_DSHOW)
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) # 1 for Arthur, 0 for Chelsea and Eleo

        if not self.cap.isOpened():
            print("Error: Could not access the webcam.")
            sys.exit()

        waitNumber = 30
        waitIndx = 0
        while True:
            # Keep capturing a frame until you get all required features from the environment 
            ret, frame = self.cap.read()
            if not ret:
                print("Camera failed to capture the frame.")
                sys.exit()

            vis = frame.copy() # Make a copy of the frame to visualise on top of so that it doesn't screw up the detection on the original frame

            # Add in a wait at the very start - noticed the first few seconds after the camera was started were red-tinted images (messed with detection)
            waitIndx += 1
            if (waitIndx < waitNumber):
                continue
            
            # ----------------------------
            # 1. Detect arena corners and store arena info as class variables
            # ----------------------------
            self.arena_corners_pixels, self.arena_width_m, self.arena_height_m = self.getArenaCornerPixelsAndRealArenaSize(frame)
            if (self.arena_corners_pixels == None or self.arena_width_m == None or self.arena_height_m == None):
                print("Couldn't locate arena")
                continue

            # ----------------------------
            # 2. Detect robot marker corners
            # ----------------------------
            robot_cam_x, robot_cam_y, robot_heading_angle, robot_marker_corners = self.getRobotPoseCameraFrame(frame)

            if (robot_cam_x is None or robot_cam_y is None or robot_heading_angle is None or robot_marker_corners is None):
                print("Couldn't locate robot")
                continue
            
            # ----------------------------
            # 3. Get the robot pose in real world coords
            # ----------------------------
            # Convert robot location to global frame
            X, Y = self.cameraToGlobal(robot_cam_x, robot_cam_y)

            # Store this robot pose as the initial robot pose
            self.initial_robot_pos = (X, Y)
            self.initial_robot_orient = robot_heading_angle

            # ----------------------------
            # 4. Find obstacles and convert their vertices to world coordinates 
            # ----------------------------
            polygons = self.locateObstaclesRed(frame)
            global_polygons = self.convertPolygonsToWorld(polygons)
            
            # ----------------------------
            # 5. Find and store goal position
            # ----------------------------
            self.goal_pos, self.goal_cam_pos, self.goal_marker_corners = self.findGoalPos(frame)
            if (self.goal_pos is None or self.goal_cam_pos is None or self.goal_marker_corners is None):
                print("Couldn't locate goal")
                continue
            
            # ----------------------------
            # 6. Create and store occupancy grid
            # ----------------------------
            self.createGrid(self.arena_width_m, self.arena_height_m, global_polygons)
           
            # ----------------------------
            # 7. Visualisations
            # ----------------------------
            self.visualiseArena(vis, self.arena_corners_pixels)
            self.visualiseRobotPose(vis, robot_marker_corners, robot_cam_x, robot_cam_y, robot_heading_angle, X, Y)
            self.visualiseObstacles(vis, polygons)
            self.visualiseGoalPos(vis, self.goal_marker_corners, self.goal_cam_pos[0], self.goal_cam_pos[1], self.goal_pos[0], self.goal_pos[1])
            break
        
        # Show the result (auto-closes after 3 seconds to avoid Jupyter kernel crash)
        cv2.imshow("Initial Environment", vis)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    def createGrid(self, arena_w, arena_h, obstacles):
        """
        Creates a grid in world frame coordinates where the bottom left corner of the arena is 0,0 so the top right corner will be arena_w, arena_h.
        Both the robot position and obstacle_polygons are relative to this 0,0 frame. The grid has (0=free, -1=obstacle) using matplotlib.path.Path.

        Credit: Quitterie
        """
        self.cell_size_cm = (arena_w/self.grid_dim)*100
        cell_size_m = self.cell_size_cm/100

        # Set global variables in utils
        utils.cell_size_cm = self.cell_size_cm
        utils.arena_width_cm = arena_w * 100  # Convert m to cm
        utils.arena_height_cm = arena_h * 100  # Convert m to cm
        grid = np.zeros((self.grid_dim, self.grid_dim), dtype=np.int8) # Use -1 for obstacles

        # Create a meshgrid of all cell centers in meters
        x_coords = np.linspace(0.5 * cell_size_m, arena_w - 0.5 * cell_size_m, self.grid_dim)
        y_coords = np.linspace(0.5 * cell_size_m, arena_w - 0.5 * cell_size_m, self.grid_dim)
        
        # Array of all (x, y) points corresponding to cell centers
        X, Y = np.meshgrid(x_coords, y_coords)
        points = np.vstack((X.flatten(), Y.flatten())).T

        occupied_indices = np.zeros(self.grid_dim * self.grid_dim, dtype=bool)

        for polygon in obstacles:
            # Reshape polygon vertices to (N, 2)
            verts = polygon.reshape(-1, 2)
            
            # Create a Path object from the polygon vertices
            poly_path = Path(verts)
            
            # Check which cell centers are contained within the polygon
            contained = poly_path.contains_points(points, radius=0)
            
            # Combine the results for all polygons (logical OR)
            occupied_indices = occupied_indices | contained

        # Map Occupancy back to the 2D Grid
        # True -> -1 (Obstacle), False -> 0 (Free)
        grid_flat = occupied_indices.astype(np.int8) * -1
        occupancy_grid = grid_flat.reshape(self.grid_dim, self.grid_dim)

        occupancy_grid = np.flipud(occupancy_grid)
        
        self.grid = occupancy_grid

    def getGrid(self):
        """Returns the occupancy grid created during initialization"""
        return self.grid

    def getCellSizeCm(self):
        """Returns the cell size of the occupancy grid created during initialization in cm"""
        return self.cell_size_cm

    def getInitialRobotPose(self):
        """Returns the robot pose detected during initialization in real world coordinates: ((X, Y) in meters, heading in radians)"""
        return self.initial_robot_pos, self.initial_robot_orient
    
    def getGoalPos(self):
        """Returns the goal position detected during initialization in real world coordinates: ((X, Y) in meters)"""
        return self.goal_pos
    
    def getGoalPosRealAndCamera(self):
        """
        Returns:
            - the goal position detected during initialization in real world coordinates: ((X, Y) in meters)
            - the goal position in camera coordinates
            - the corners of the goal ArUco marker in camera coordinates
        """
        return self.goal_pos, self.goal_cam_pos, self.goal_marker_corners

    def display_grid(self, start_cell=None, goal_cell=None):
        """
        Credit: Quitterie
        """
        map_grid = self.grid
        cmap = ListedColormap(['white', 'black', 'red', 'green', 'blue'])
        map_display = np.zeros_like(map_grid, dtype=object)

        # Colors
        map_display[map_grid == -1] = 'red'  # Obstacle
        map_display[map_grid == 0] = 'white'   # Free Space

        if start_cell is not None:
            map_display[start_cell] = 'blue'
        if goal_cell is not None:
            map_display[goal_cell] = 'green'

        # Convert color names to numbers
        color_mapping = {'white': 0, 'black': 1, 'red': 2, 'blue': 3,
                        'green': 4}
        map_numeric_display = np.vectorize(color_mapping.get)(map_display)

        # Show map 
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(map_numeric_display, cmap=cmap)

        # Set tick positions (in cell indices)
        x_positions = np.arange(0, 201, 20)
        y_positions = np.arange(0, 201, 20)

        # Convert cell indices to cm labels
        x_labels = [int(utils.cell_to_cm(col)) for col in x_positions]
        y_labels = [int(utils.arena_height_cm - utils.cell_to_cm(row)) for row in y_positions]
        plt.xticks(x_positions, x_labels)
        plt.yticks(y_positions, y_labels)
        ax.set_xlabel('X Dimension (cm)')
        ax.set_ylabel('Y Dimension (cm)')
        
        # Grid lines
        ax.set_xticks(np.arange(-0.5, map_grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, map_grid.shape[0], 1), minor=True)
        ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.15)

        plt.show()

    # ============================================================
    #  AXIS CONVERSION FUNCTIONS - use utils module
    # ============================================================
    # Use utils.real_to_grid() and utils.grid_to_real() instead

    def getArenaCornerPixelsAndRealArenaSize(self, image):
        """
        Detects the two arena ArUco markers in image:
            ID 1 = Bottom-left (BL) marker
            ID 2 = Top-right (TR) marker

        Returns:
            pixel_corners : pixel corners in form [BL, BR, TR, TL]
            arena_width_m : arena width in world frame (m)
            arena_height_m : arena height in world frame (m)
        """

        # Dictionary for aruco markers
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.arenaMarkerDict)
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

        # Convert to gray scale for better detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            print("No ArUco markers detected")
            return None, None, None

        ids = ids.flatten()

        # Must detect both markers
        if not (1 in ids and 2 in ids):
            print(f"Missing arena markers: found ID {ids}, expected IDs 1 and 2")
            return None, None, None

        # Find corners of markers
        idx_bl = list(ids).index(1) # Bottom left arena marker
        idx_tr = list(ids).index(2) # Top right arena marker

        corners_bl = corners[idx_bl][0]  # 4 corner points (TL,TR,BR,BL)
        corners_tr = corners[idx_tr][0]

        # Pixel center of BL marker
        bl_px = (int(corners_bl[:,0].mean()), int(corners_bl[:,1].mean())) # Mean x points of all rows and mean y points of the all rows of the corners
        # Pixel center of TR marker
        tr_px = (int(corners_tr[:,0].mean()), int(corners_tr[:,1].mean()))

        # Build rectangular arena in pixels:
        x1, y1 = bl_px
        x2, y2 = tr_px
        br_px = (x2, y1)
        tl_px = (x1, y2)

        pixel_corners = [bl_px, br_px, tr_px, tl_px]

        # Compute meters per pixel

        # Use BL marker (ID 1) to find the marker width in pixels (use top-left to top-right edge)
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

        return pixel_corners, arena_width_m, arena_height_m
    
    def findGoalPos(self, image):
        """
        Detects the goal ArUco marker in image:
            ID 3

        Returns:
            (X, Y) : position of goal in world frame (m)
            (goal_cam_x, goal_cam_y) : position of goal in camera frame (pixels)
            goal_marker_corners : pixel corners of the ArUco marker in form [BL, BR, TR, TL]
        """

        # Get goal position in camera frame
        goal_cam_x, goal_cam_y, goal_marker_corners = self.getGoalPosCameraFrame(image)

        if (goal_cam_x is None or goal_cam_y is None or goal_marker_corners is None):
            print("Couldn't locate goal position so exiting find goal function")
            return None, None, None

        # Convert to global coordinates
        X, Y = self.cameraToGlobal(goal_cam_x, goal_cam_y)

        return (X, Y), (goal_cam_x, goal_cam_y), goal_marker_corners

    
    def getRobotPose(self, image):
        """
        Detects the robot ArUco marker in image:
            ID 0

        Returns:
            [X, Y] : position of robot in world frame (m)
            robot_heading_angle: angle that the front of the robot makes with the horizontal in rads. In range (-pi, pi]
        """
        # Get robot pose in camera frame
        robot_cam_x, robot_cam_y, robot_heading_angle, robot_marker_corners = self.getRobotPoseCameraFrame(image)

        if (robot_cam_x is None or robot_cam_y is None or robot_heading_angle is None or robot_marker_corners is None):
            print("Couldn't locate robot so exiting getRobotPose function")
            return None, None

        # Convert to global coordinates
        X, Y = self.cameraToGlobal(robot_cam_x, robot_cam_y)

        return [X, Y], robot_heading_angle
    
    def getRobotPoseAndVisualise(self, image, vis):
        """
        Detects the robot ArUco marker in image and visualises its position and heading direction on the vis instance:
            ID 0

        Returns:
            [X, Y] : position of robot in world frame (m)
            robot_heading_angle: angle that the front of the robot makes with the horizontal in rads. In range (-pi, pi]
        """
        # Get robot pose in camera frame
        robot_cam_x, robot_cam_y, robot_heading_angle, robot_marker_corners = self.getRobotPoseCameraFrame(image)

        if (robot_cam_x is None or robot_cam_y is None or robot_heading_angle is None or robot_marker_corners is None):
            print("Couldn't locate robot so exiting geRobotPoseAndVisualise function")
            return None, None 

        # Convert to global coordinates
        X, Y = self.cameraToGlobal(robot_cam_x, robot_cam_y)

        # Visualise
        self.visualiseRobotPose(vis, robot_marker_corners, robot_cam_x, robot_cam_y, robot_heading_angle, X, Y)

        return [X, Y], robot_heading_angle

        
    def getRobotPoseCameraFrame(self, image):
        """
        Detects the robot ArUco marker (ID 0) in the image and returns its pose in the camera frame.

        Returns:
            cx, cy : center pixel of marker
            heading_angle : heading angle in radians
            robotCorners : 4 corner points in order (TL, TR, BR, BL)
        """

        # Dictionary for aruco markers
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.robotMarkerDict) # Loads the ArUco dictionary of marker patterns that contains our printed aruco marker - tells OpenCV which patterns to look for
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters()) # tells openCv to look for patterns in our aruco_dict dict, and uses default parameters for things like: corner refinement, thresholding, adaptive window size, min marker size, error correction etc.

        # Convert to gray scale for better detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale (1 channel instead of 3) since colour adds no info for detecting black and white markers - open cv does automatically if you pass colour but this gives more control, is more explicit, openCv call faster
        corners, ids, _ = detector.detectMarkers(gray) # Searches the image for any markers from the given dict - _ is the contours that looked like markers but failed decoding (rejected)

        # No markers at all
        if ids is None:
            return None, None, None, None

        # Convert ids to a simple list - instead of something like array([2], [1], [0]), it gives [2, 1, 0]
        ids = ids.flatten()        

        # Check if ID 0 exists
        if 0 not in ids:
            return None, None, None, None

        # Find index of ID 0 - in [2, 1, 0], index would be 2
        idx = list(ids).index(0) # Convert to list so that you can call .index(0) - list(ids) in form [np.int32(2), np.int32(1), np.int32(0)]

        # Get its corners
        robotCorners = corners[idx][0]  # corners looks something like [array([[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]]), array([[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]]), ...]) so need [0] to get the actual corners at the idx (ID 0)
        tl, tr, br, bl = robotCorners

        # Find center pixel of the marker
        cx = int(robotCorners[:, 0].mean()) # Average of x values in all rows
        cy = int(robotCorners[:, 1].mean()) # Average of y values in all rows

        # Heading direction vector (TR - TL)
        dir_vec = tr - tl
        heading_angle = np.arctan2(-dir_vec[1], dir_vec[0])  # (radians) NOTE: do negative of y since camera y is top bottom not bottom top 

        return cx, cy, heading_angle, robotCorners
    
    def getGoalPosCameraFrame(self, image):
        """
        Detects the goal ArUco marker (ID 3) in the image and returns the position in camera frame.

        Returns:
            cx, cy: center pixel of marker
            goalCorners: 4 corner points of marker (TL, TR, BR, BL)
        """
        # Goal Aruco ID
        goalId = 3

        # Dictionary for aruco markers
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.goalMarkerDict)
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

        # Convert to gray scale for better detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        # No markers at all
        if ids is None:
            return None, None, None

        # Convert ids to a simple list
        ids = ids.flatten()

        # Check if goalID exists
        if goalId not in ids:
            return None, None, None

        # Find index of ID goalId
        idx = list(ids).index(goalId)

        # Get its corners
        goalCorners = corners[idx][0]

        # Find center pixel of the marker
        cx = int(goalCorners[:, 0].mean())
        cy = int(goalCorners[:, 1].mean())

        return cx, cy, goalCorners

    def computeHomography(self, arena_corners_pixels, arena_width_m, arena_height_m):
        """
        Computes the 2D homography matrix H that maps pixel coordinates (u, v)
        to real-world coordinates (X, Y) in meters.

        The transformation is a planar projective transform (2D homography),
        represented by a 3x3 matrix H. Points are converted using homogeneous 
        coordinates:
            [X']   [h11 h12 h13]   [u]
            [Y'] = [h21 h22 h23] * [v]
            [W']   [h31 h32 h33]   [1]

        Then the true real-world coordinates are obtained by dividing by W':
            X = X' / W'
            Y = Y' / W'

        Returns:
            H : the homography matrix that maps pixel coordinates to world coordinates.
        """

        # Real-world coordinates of the arena corners (meters), defining the
        # world coordinate system. Bottom-left = (0,0), top-right = (width,height).
        world_corners = np.array([
            [0, 0],                          # Bottom-left
            [arena_width_m, 0],              # Bottom-right
            [arena_width_m, arena_height_m], # Top-right
            [0, arena_height_m]              # Top-left
        ], dtype=float)

        # Convert inputs to the shape expected by cv2.findHomography
        img_pts = np.array(arena_corners_pixels, dtype=float) # shape: (4, 2)
        world_pts = world_corners.reshape(-1, 1, 2) # shape: (4, 1, 2)

        # --- 2D Homography Calculation ---
        # cv2.findHomography computes the 3×3 matrix H that best satisfies:
        #
        #   [X, Y, 1]^T  =  H * [u, v, 1]^T
        #
        # This is a planar projective transformation (2D homography),
        # requiring homogeneous coordinates (the extra "1").
        H, _ = cv2.findHomography(img_pts, world_pts)

        return H


    def pixelToGlobal(self, H, pixel_point):
        """
        Convert a pixel point (u,v) into global (X,Y) using homography H.

        Returns:
            X, Y: position of point in global frame
        """
        u, v = pixel_point
        pt = np.array([u, v, 1.0])
        world = H @ pt

        # Recall that:
        #    [X']   [h11 h12 h13]   [u]
        #    [Y'] = [h21 h22 h23] * [v]
        #    [W']   [h31 h32 h33]   [1]
        #
        # So, since homography uses homogeneous coordinates which are scale invariant, we must divide X' and Y' by W' to get actual world coordinates
        world /= world[2]  

        X = float(world[0])
        Y = float(world[1])

        return X, Y
    
    def globalToPixel(self, H, global_point):
        """
        Convert a global point (X, Y) into camera pixel coordinates (u, v) using the inverse of homography H.

        Returns:
            u, v: position of point in camera frame
        """
        X, Y = global_point

        # We need H⁻¹ to go from world to pixel
        H_inv = np.linalg.inv(H)

        pt = np.array([X, Y, 1.0])
        pixel = H_inv @ pt
        pixel /= pixel[2]   # Normalize homogeneous coordinates

        u = float(pixel[0])
        v = float(pixel[1])

        return u, v

    def cameraToGlobal(self, marker_cam_x, marker_cam_y):
        """
        Convert a pixel point (u,v) into global coordinates (X,Y).

        Returns:
            X, Y: position of point in global frame
        """
        H = self.computeHomography(self.arena_corners_pixels, self.arena_width_m, self.arena_height_m)
        return self.pixelToGlobal(H, (marker_cam_x, marker_cam_y))
    
    def globalToCamera(self, gloabl_x, global_y):
        """
        Convert a global point (X, Y) into camera pixel coordinates (u, v).

        Returns:
            u, v: position of point in camera frame
        """
        H = self.computeHomography(self.arena_corners_pixels, self.arena_width_m, self.arena_height_m)
        return self.globalToPixel(H, (gloabl_x, global_y))
    
    def convertPolygonsToWorld(self, polygons):
        """
        Convert a list of pixel-based polygons into world coordinates.

        Returns:
            world_polygons: a list of polygons whose vertices are in world coordinates (meters).
        """
        # Compute homography
        H = self.computeHomography(self.arena_corners_pixels, self.arena_width_m, self.arena_height_m)

        world_polygons = []

        for poly in polygons:
            world_poly = []
            for (u, v) in poly:
                X, Y = self.pixelToGlobal(H, (u, v))
                world_poly.append([X, Y])

            # Convert each polygon separately - (N,1,2)
            poly_np = np.array(world_poly, dtype=np.float32).reshape(-1, 1, 2)
            world_polygons.append(poly_np)


        return world_polygons

    def locateObstaclesRed(self, image):
        """
        Detects red blobs in an image, and approximates their shapes as polygons.

        Returns:
            polygons: a list of polygons in the camera frame (each polygon is a list of (u,v) points).
        """

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red spans the start and end of the hue circle so need 2 upper and lower bounds
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
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # RETR_EXTERNAL: retrieve only the outermost contours, CHAIN_APPROX_NONE: no compression

        polygons = []

        for cnt in contours:
            # Filter out small artifacts
            if cv2.contourArea(cnt) < 100: # Computes area enclosed by the contours in pixels - rejects anything less than 100 pixels to eliminate small noise
                continue

            # Approximate to polygon
            epsilon = 0.001 * cv2.arcLength(cnt, True) # Use 0.1% of the contour perimiter as the tolerance for simplification (smaller epsilon = more detailed)
            poly = cv2.approxPolyDP(cnt, epsilon, True) # Uses the Ramer–Douglas–Peucker algorithm to produce a polygon with fewer vertices while approximating the original shape

            # Convert polygon format for output
            polygon_points = [(int(p[0][0]), int(p[0][1])) for p in poly]
            polygons.append(polygon_points)

        return polygons

    def visualiseArena(self, vis, arena_corners_pixels):
        """
        Draws the corners of the arena onto the vis frame.
        """
        # Draw arena outline (green polygon)
        pts = np.array(arena_corners_pixels, dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

        # Draw arena corner names
        labels = ["BL", "BR", "TR", "TL"]
        for (x, y), name in zip(arena_corners_pixels, labels):
            cv2.putText(vis, name, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def visualiseRobotPose(self, vis, robot_marker_corners, robot_center_cam_x, robot_center_cam_y, robot_heading_angle, robot_center_world_x, robot_center_world_y):
        """
        Draws the pose of the robot onto the vis frame, including a marker bounding box, the robot’s center position, and the robot's heading direction as an arrow.
        """
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
        
    def visualiseGoalPos(self, vis, goal_marker_corners, goal_center_cam_x, goal_center_cam_y, goal_center_world_x, goal_center_world_y):
        """
        Draws a bounding box around the goal marker and highlights the center position in the vis frame.
        """
        # Draw ArUco marker location
        int_corners = goal_marker_corners.astype(np.int32)
        cv2.polylines(vis, [int_corners], True, (255, 0, 0), 3)

        # Draw center point
        cv2.circle(vis, (goal_center_cam_x, goal_center_cam_y), 6, (0, 0, 255), -1)

        # Label with global coordinate
        coord_text = f"({goal_center_world_x:.5f}m, {goal_center_world_y:.5f}m)"
        cv2.putText(vis, coord_text, (goal_center_cam_x + 15, goal_center_cam_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


    def visualiseObstacles(self, vis, polygons):
        """
        Draws each obstacle polygon in the vis frame.
        """
        # Outline obstacles
        for poly in polygons:
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    def visualiseGlobalPoint(self, vis, point_x, point_y, colour=(0, 255, 0)): # BGR format - green colour at default
        """
        Draws a point in the world frame (x, y) in meters onto the vis image.
        """
        # Convert global position to camera pixels
        camera_x, camera_y = self.globalToCamera(point_x, point_y)

        # If the point is invalid or outside image, stop
        if camera_x is None or camera_y is None:
            print("Could not get camera coords")
            return
                
        # Draw the point on the image
        cv2.circle(vis, (int(camera_x), int(camera_y)), 2, colour, -1)

    def visualiseGlobalPath(self, vis, path_waypoints, colour=(0, 255, 0), max_points=200):
        """
        Draw a path of global points onto the camera frame. Takes cm.
        
        vis : frame to draw on
        path_waypoints : list of (x, y) global coordinates in cm
        colour : BGR colour for points and lines
        max_points : max number of points to plot (down-samples if needed)
        """

        if not path_waypoints or len(path_waypoints) < 1:
            return

        # Apply sampling if path is too long
        total_pts = len(path_waypoints)

        if total_pts > max_points:
            step = total_pts / max_points
            sampled = []

            for i in range(max_points):
                idx = int(i * step)
                sampled.append(path_waypoints[idx])

            # Ensure last point is exactly included
            if sampled[-1] != path_waypoints[-1]:
                sampled.append(path_waypoints[-1])

            path_waypoints = sampled

        # Convert sampled waypoints to pixel points
        pixel_points = []

        for [x, y] in path_waypoints:
            # convert cm to m
            x, y = x / 100.0, y / 100.0

            px, py = self.globalToCamera(x, y)
            if px is None or py is None:
                continue # skip invalid points

            pixel_points.append((int(px), int(py)))

            # Draw waypoint as a circle
            self.visualiseGlobalPoint(vis, x, y, colour)

        # Draw the connecting path lines
        for i in range(len(pixel_points) - 1):
            cv2.line(vis, pixel_points[i], pixel_points[i + 1], colour, 2)

    def visualiseGlobalPoints(self, vis, points, colour=(0, 255, 0)):  # BGR format - green colour at default
        """
        Draw multiple global points onto the camera frame.
        
        vis              : frame to draw on
        points           : list of (x, y) or (x, y, colour)
        default_colour   : BGR fallback colour if point has none
        """
        for p in points:
            # If user provided colour per point
            if len(p) == 3:
                x, y, colour = p
            else:
                x, y = p

            # Call your existing function
            self.visualiseGlobalPoint(vis, x, y, colour)
        


if __name__ == "__main__":
    """
    A demo usage of the vision class
    """
    visionInstance = Vision()
    print("arena width and height : ", visionInstance.arena_width_m, visionInstance.arena_height_m)

    if not visionInstance.cap.isOpened():
        print("Error: Could not access the webcam.")
        exit()

    # Live robot pose visualisation
    while True:
        ret, frame = visionInstance.cap.read()
        if not ret:
            print("Camera failed to capture the frame.")
            break

        vis = frame.copy()

        # Always redraw arena outline
        visionInstance.visualiseArena(vis, visionInstance.arena_corners_pixels)

        # Always detect and draw robot pose
        position, robot_heading_angle = visionInstance.getRobotPoseAndVisualise(frame, vis)
        if (position is None or robot_heading_angle is None):
            continue
        [X, Y] = position
        # print(f"X: {X:.5f}, Y: {Y:.5f}, Direction: {robot_heading_angle:.5f}")

        # Visualise a random path starting from the robot's position
        path = [[X*100, Y*100], [40, 40], [50, 60], [70, 70]]
        visionInstance.visualiseGlobalPath(vis, path)

        # Show the live window
        cv2.imshow("Live Robot Pose", vis)

        # Exit on q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    visionInstance.cap.release()
    cv2.destroyAllWindows()
