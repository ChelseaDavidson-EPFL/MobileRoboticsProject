import cv2
import numpy as np
import cv2.aruco as aruco

def liveFeed(): 
    cap = cv2.VideoCapture(0) # Getting the webcame feed

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        exit()

    print("Webcam accessed successfully!")

    # Create a named window once
    cv2.namedWindow("Live Feed")

    while True:
        # Reading the webcame image
        ret, frame = cap.read()

        if not ret:
            print("Error: Camera failed to capture the frame.")
            break

        cv2.imshow("Live Feed", frame) # Sends pixel data to the window buffer but doesn't actuall update it

        # waitKey() updates the Live Feed window, waitKey(1) waits 1ms for a key to be pressed and 0xFF keep only the lowest 8 bits so pressing q matches asci 'q'
        key = cv2.waitKey(1) & 0xFF

        # let's you quit by pressing q
        if key == ord('q'):  
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


def liveFeedWithGridWhite():
    cap = cv2.VideoCapture(0)  # Webcam feed

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        exit()

    print("Webcam accessed successfully!")
    cv2.namedWindow("Live Feed")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Camera failed to capture the frame.")
            break

        # --- Detect white squares ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define white color range (low saturation, high value)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 40, 255])

        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Optional: remove noise
        mask = cv2.medianBlur(mask, 5)

        # Find contours of the white squares
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        white_centers = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # filter out small noise
                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + w // 2
                cy = y + h // 2
                white_centers.append((cx, cy))
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # mark center

        # --- Create 10x10 grid if 4 white squares detected ---
        if len(white_centers) == 4:
            # Sort centers to get corners in consistent order: top-left, top-right, bottom-left, bottom-right
            white_centers = sorted(white_centers, key=lambda c: (c[1], c[0]))  # sort by y, then x
            top = sorted(white_centers[:2], key=lambda c: c[0])
            bottom = sorted(white_centers[2:], key=lambda c: c[0])
            tl, tr = top
            bl, br = bottom

            # Draw horizontal lines
            for i in range(11):
                alpha = i / 10
                start_x = int(tl[0] * (1 - alpha) + bl[0] * alpha)
                start_y = int(tl[1] * (1 - alpha) + bl[1] * alpha)
                end_x = int(tr[0] * (1 - alpha) + br[0] * alpha)
                end_y = int(tr[1] * (1 - alpha) + br[1] * alpha)
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 1)

            # Draw vertical lines
            for j in range(11):
                beta = j / 10
                start_x = int(tl[0] * (1 - beta) + tr[0] * beta)
                start_y = int(tl[1] * (1 - beta) + tr[1] * beta)
                end_x = int(bl[0] * (1 - beta) + br[0] * beta)
                end_y = int(bl[1] * (1 - beta) + br[1] * beta)
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 1)

        cv2.imshow("Live Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def liveFeedWithGridRed():
    cap = cv2.VideoCapture(0)  # Webcam feed

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        exit()

    print("Webcam accessed successfully!")
    cv2.namedWindow("Live Feed")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Camera failed to capture the frame.")
            break

        # --- Detect red squares ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define red color range (two ranges because red wraps around HSV hue)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2

        # Optional: remove noise
        mask = cv2.medianBlur(mask, 5)

        # Find contours of the red squares
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_centers = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # filter out small noise
                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + w // 2
                cy = y + h // 2
                red_centers.append((cx, cy))
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # mark center

        # --- Create 10x10 grid if 4 red squares detected ---
        if len(red_centers) == 4:
            # Sort centers to get corners in consistent order: top-left, top-right, bottom-left, bottom-right
            red_centers = sorted(red_centers, key=lambda c: (c[1], c[0]))  # sort by y, then x
            top = sorted(red_centers[:2], key=lambda c: c[0])
            bottom = sorted(red_centers[2:], key=lambda c: c[0])
            tl, tr = top
            bl, br = bottom

            # Draw horizontal lines
            for i in range(11):
                alpha = i / 10
                start_x = int(tl[0] * (1 - alpha) + bl[0] * alpha)
                start_y = int(tl[1] * (1 - alpha) + bl[1] * alpha)
                end_x = int(tr[0] * (1 - alpha) + br[0] * alpha)
                end_y = int(tr[1] * (1 - alpha) + br[1] * alpha)
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 1)

            # Draw vertical lines
            for j in range(11):
                beta = j / 10
                start_x = int(tl[0] * (1 - beta) + tr[0] * beta)
                start_y = int(tl[1] * (1 - beta) + tr[1] * beta)
                end_x = int(bl[0] * (1 - beta) + br[0] * beta)
                end_y = int(bl[1] * (1 - beta) + br[1] * beta)
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 1)

        cv2.imshow("Live Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def liveFeedWithAruco():
    cap = cv2.VideoCapture(0)  # Webcam feed

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        exit()

    print("Webcam accessed successfully!")
    cv2.namedWindow("Live Feed")

    # Use a predefined dictionary (4x4 with 50 unique markers)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    parameters = cv2.aruco.DetectorParameters()  # updated

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Camera failed to capture the frame.")
            break

        # Convert to grayscale for ArUco detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, rejected = detector.detectMarkers(gray)

        # If markers detected
        if ids is not None:
            for i, corner in enumerate(corners):
                # Draw the bounding box
                int_corners = corner.astype(np.int32)
                cv2.polylines(frame, int_corners, True, (0, 255, 0), 2)

                # Compute center of the marker
                c = corner[0]
                cx = int(c[:, 0].mean())
                cy = int(c[:, 1].mean())
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # Display the ID
                marker_id = ids[i][0]
                cv2.putText(frame, f"ID:{marker_id}", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 2)

                # Print marker location in pixels (optional)
                print(f"Marker {marker_id} at ({cx}, {cy})")

        cv2.imshow("Live Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def liveFeedRedGridWithAruco():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        exit()

    cv2.namedWindow("Live Feed")
    print("Webcam accessed successfully!")

    # ArUco dictionary and detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera failed to capture the frame.")
            break

        # --- Detect red squares ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + w // 2
                cy = y + h // 2
                red_centers.append((cx, cy))

        # --- Detect ArUco markers first ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        marker_info = []
        if ids is not None:
            for i, corner in enumerate(corners):
                # Marker center
                c = corner[0]
                cx = int(c[:, 0].mean())
                cy = int(c[:, 1].mean())
                marker_id = ids[i][0]
                marker_info.append((cx, cy, corner, marker_id))

        # --- Now overlay the red grid ---
        if len(red_centers) == 4:
            red_centers = sorted(red_centers, key=lambda c: (c[1], c[0]))
            top = sorted(red_centers[:2], key=lambda c: c[0])
            bottom = sorted(red_centers[2:], key=lambda c: c[0])
            tl, tr = top
            bl, br = bottom

            # Horizontal lines
            for i in range(11):
                alpha = i / 10
                start_x = int(tl[0] * (1 - alpha) + bl[0] * alpha)
                start_y = int(tl[1] * (1 - alpha) + bl[1] * alpha)
                end_x = int(tr[0] * (1 - alpha) + br[0] * alpha)
                end_y = int(tr[1] * (1 - alpha) + br[1] * alpha)
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 1)

            # Vertical lines
            for j in range(11):
                beta = j / 10
                start_x = int(tl[0] * (1 - beta) + tr[0] * beta)
                start_y = int(tl[1] * (1 - beta) + tr[1] * beta)
                end_x = int(bl[0] * (1 - beta) + br[0] * beta)
                end_y = int(bl[1] * (1 - beta) + br[1] * beta)
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 1)

        # --- Draw ArUco markers and map to grid ---
        for cx, cy, corner, marker_id in marker_info:
            int_corners = corner.astype(np.int32)
            cv2.polylines(frame, [int_corners], True, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID:{marker_id}", (cx+10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            # Map marker to grid cell if grid detected
            if len(red_centers) == 4:
                u = (cx - tl[0]) / (tr[0] - tl[0]) if tr[0] != tl[0] else 0
                v = (cy - tl[1]) / (bl[1] - tl[1]) if bl[1] != tl[1] else 0
                u = min(max(u, 0), 0.999)
                v = min(max(v, 0), 0.999)
                cell_x = int(u * 10)
                cell_y = int(v * 10)
                cv2.putText(frame, f"Cell:({cell_x},{cell_y})", (cx+10, cy+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                print(f"Marker {marker_id} in cell ({cell_x},{cell_y})")

        cv2.imshow("Live Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def liveFeedRedGridWithArucoConservative():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        exit()

    cv2.namedWindow("Live Feed")
    print("Webcam accessed successfully!")

    # ArUco dictionary and detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera failed to capture the frame.")
            break

        overlay = frame.copy()  # For drawing filled cells

        # --- Detect red squares ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + w // 2
                cy = y + h // 2
                red_centers.append((cx, cy))

        # --- Detect ArUco markers first ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        marker_info = []
        if ids is not None:
            for i, corner in enumerate(corners):
                # Marker center
                c = corner[0]
                cx = int(c[:, 0].mean())
                cy = int(c[:, 1].mean())
                marker_id = ids[i][0]
                marker_info.append((cx, cy, corner, marker_id))

        # --- Draw 10x10 grid and store cell coordinates ---
        grid_cells = []  # Will store (top-left, bottom-right) of each cell

        if len(red_centers) == 4:
            red_centers = sorted(red_centers, key=lambda c: (c[1], c[0]))
            top = sorted(red_centers[:2], key=lambda c: c[0])
            bottom = sorted(red_centers[2:], key=lambda c: c[0])
            tl, tr = top
            bl, br = bottom

            # Precompute cell corners
            for i in range(10):
                alpha1 = i / 10
                alpha2 = (i + 1) / 10
                start_row_tl = (int(tl[0] * (1 - alpha1) + bl[0] * alpha1),
                                int(tl[1] * (1 - alpha1) + bl[1] * alpha1))
                end_row_tl = (int(tl[0] * (1 - alpha2) + bl[0] * alpha2),
                              int(tl[1] * (1 - alpha2) + bl[1] * alpha2))

                start_row_tr = (int(tr[0] * (1 - alpha1) + br[0] * alpha1),
                                int(tr[1] * (1 - alpha1) + br[1] * alpha1))
                end_row_tr = (int(tr[0] * (1 - alpha2) + br[0] * alpha2),
                              int(tr[1] * (1 - alpha2) + br[1] * alpha2))

                # For each row, create 10 vertical cells
                for j in range(10):
                    beta1 = j / 10
                    beta2 = (j + 1) / 10
                    cell_tl = (int(start_row_tl[0] * (1 - beta1) + start_row_tr[0] * beta1),
                               int(start_row_tl[1] * (1 - beta1) + start_row_tr[1] * beta1))
                    cell_br = (int(end_row_tl[0] * (1 - beta2) + end_row_tr[0] * beta2),
                               int(end_row_tl[1] * (1 - beta2) + end_row_tr[1] * beta2))
                    grid_cells.append((cell_tl, cell_br))

            # Draw grid lines on frame (optional)
            for cell_tl, cell_br in grid_cells:
                cv2.rectangle(frame, cell_tl, cell_br, (255, 0, 0), 1)

        # --- Draw ArUco markers and mark overlapping cells ---
        for cx, cy, corner, marker_id in marker_info:
            int_corners = corner.astype(np.int32)
            cv2.polylines(frame, [int_corners], True, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID:{marker_id}", (cx+10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            if grid_cells:
                # Marker bounding box
                m_tl = (int(corner[0][:,0].min()), int(corner[0][:,1].min()))
                m_br = (int(corner[0][:,0].max()), int(corner[0][:,1].max()))

                # Check overlap with all grid cells
                for cell_tl, cell_br in grid_cells:
                    overlap_x = max(0, min(m_br[0], cell_br[0]) - max(m_tl[0], cell_tl[0]))
                    overlap_y = max(0, min(m_br[1], cell_br[1]) - max(m_tl[1], cell_tl[1]))
                    if overlap_x > 0 and overlap_y > 0:
                        # Fill overlapping cell in red (semi-transparent)
                        cv2.rectangle(overlay, cell_tl, cell_br, (0,0,255), -1)

        # Combine overlay with original frame
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        cv2.imshow("Live Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def liveFeedTwoWindows():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        exit()

    cv2.namedWindow("Detection Feed")
    cv2.namedWindow("Overlay Feed")
    print("Webcam accessed successfully!")

    # ArUco dictionary and detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera failed to capture the frame.")
            break

        # --- Use a copy for detection (no overlays) ---
        detection_frame = frame.copy()

        # --- Detect red squares ---
        hsv = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2HSV)
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
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + w // 2
                cy = y + h // 2
                red_centers.append((cx, cy))

        # --- Detect ArUco markers ---
        gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        marker_info = []
        if ids is not None:
            for i, corner in enumerate(corners):
                c = corner[0]
                cx = int(c[:, 0].mean())
                cy = int(c[:, 1].mean())
                marker_id = ids[i][0]
                marker_info.append((cx, cy, corner, marker_id))

        # --- Prepare overlay frame ---
        overlay_frame = frame.copy()
        grid_cells = []

        if len(red_centers) == 4:
            red_centers = sorted(red_centers, key=lambda c: (c[1], c[0]))
            top = sorted(red_centers[:2], key=lambda c: c[0])
            bottom = sorted(red_centers[2:], key=lambda c: c[0])
            tl, tr = top
            bl, br = bottom

            # Precompute cell corners
            for i in range(10):
                alpha1 = i / 10
                alpha2 = (i + 1) / 10
                start_row_tl = (int(tl[0] * (1 - alpha1) + bl[0] * alpha1),
                                int(tl[1] * (1 - alpha1) + bl[1] * alpha1))
                end_row_tl = (int(tl[0] * (1 - alpha2) + bl[0] * alpha2),
                              int(tl[1] * (1 - alpha2) + bl[1] * alpha2))

                start_row_tr = (int(tr[0] * (1 - alpha1) + br[0] * alpha1),
                                int(tr[1] * (1 - alpha1) + br[1] * alpha1))
                end_row_tr = (int(tr[0] * (1 - alpha2) + br[0] * alpha2),
                              int(tr[1] * (1 - alpha2) + br[1] * alpha2))

                for j in range(10):
                    beta1 = j / 10
                    beta2 = (j + 1) / 10
                    cell_tl = (int(start_row_tl[0] * (1 - beta1) + start_row_tr[0] * beta1),
                               int(start_row_tl[1] * (1 - beta1) + start_row_tr[1] * beta1))
                    cell_br = (int(end_row_tl[0] * (1 - beta2) + end_row_tr[0] * beta2),
                               int(end_row_tl[1] * (1 - beta2) + end_row_tr[1] * beta2))
                    grid_cells.append((cell_tl, cell_br))

                    # Draw grid lines (optional)
                    cv2.rectangle(overlay_frame, cell_tl, cell_br, (255, 0, 0), 1)

        # --- Draw markers and fill overlapping cells ---
        for cx, cy, corner, marker_id in marker_info:
            int_corners = corner.astype(np.int32)
            cv2.polylines(overlay_frame, [int_corners], True, (0, 255, 0), 2)
            cv2.circle(overlay_frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(overlay_frame, f"ID:{marker_id}", (cx+10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            if grid_cells:
                m_tl = (int(corner[0][:,0].min()), int(corner[0][:,1].min()))
                m_br = (int(corner[0][:,0].max()), int(corner[0][:,1].max()))

                for cell_tl, cell_br in grid_cells:
                    overlap_x = max(0, min(m_br[0], cell_br[0]) - max(m_tl[0], cell_tl[0]))
                    overlap_y = max(0, min(m_br[1], cell_br[1]) - max(m_tl[1], cell_tl[1]))
                    if overlap_x > 0 and overlap_y > 0:
                        # Fill overlapping cell in red (semi-transparent)
                        cv2.rectangle(overlay_frame, cell_tl, cell_br, (0,0,255), -1)

        # Combine overlay transparency
        overlay_frame = cv2.addWeighted(overlay_frame, 0.6, frame, 0.4, 0)

        # --- Show windows ---
        cv2.imshow("Detection Feed", detection_frame)  # Used for detection only
        cv2.imshow("Overlay Feed", overlay_frame)      # Shows grid + red filled cells

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def liveFeedTwoWindowsLargeGrid(grid_size):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        exit()

    cv2.namedWindow("Detection Feed")
    cv2.namedWindow("Overlay Feed")
    print("Webcam accessed successfully!")

    # ArUco dictionary and detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera failed to capture the frame.")
            break

        # --- Use a copy for detection ---
        detection_frame = frame.copy()

        # --- Detect red squares ---
        hsv = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2HSV)
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
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + w // 2
                cy = y + h // 2
                red_centers.append((cx, cy))

        # --- Detect ArUco markers ---
        gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        marker_info = []
        if ids is not None:
            for i, corner in enumerate(corners):
                c = corner[0]
                cx = int(c[:, 0].mean())
                cy = int(c[:, 1].mean())
                marker_id = ids[i][0]
                marker_info.append((cx, cy, corner, marker_id))

        # --- Prepare overlay frame ---
        overlay_frame = frame.copy()
        grid_cells = []

        if len(red_centers) == 4:
            # Sort corners: top-left, top-right, bottom-left, bottom-right
            red_centers = sorted(red_centers, key=lambda c: (c[1], c[0]))
            top = sorted(red_centers[:2], key=lambda c: c[0])
            bottom = sorted(red_centers[2:], key=lambda c: c[0])
            tl, tr = top
            bl, br = bottom

            # Precompute cell corners based on grid_size
            for i in range(grid_size):
                alpha1 = i / grid_size
                alpha2 = (i + 1) / grid_size
                start_row_tl = (int(tl[0] * (1 - alpha1) + bl[0] * alpha1),
                                int(tl[1] * (1 - alpha1) + bl[1] * alpha1))
                end_row_tl = (int(tl[0] * (1 - alpha2) + bl[0] * alpha2),
                              int(tl[1] * (1 - alpha2) + bl[1] * alpha2))

                start_row_tr = (int(tr[0] * (1 - alpha1) + br[0] * alpha1),
                                int(tr[1] * (1 - alpha1) + br[1] * alpha1))
                end_row_tr = (int(tr[0] * (1 - alpha2) + br[0] * alpha2),
                              int(tr[1] * (1 - alpha2) + br[1] * alpha2))

                for j in range(grid_size):
                    beta1 = j / grid_size
                    beta2 = (j + 1) / grid_size
                    cell_tl = (int(start_row_tl[0] * (1 - beta1) + start_row_tr[0] * beta1),
                               int(start_row_tl[1] * (1 - beta1) + start_row_tr[1] * beta1))
                    cell_br = (int(end_row_tl[0] * (1 - beta2) + end_row_tr[0] * beta2),
                               int(end_row_tl[1] * (1 - beta2) + end_row_tr[1] * beta2))
                    grid_cells.append((cell_tl, cell_br))

                    # Draw grid lines on overlay (optional)
                    cv2.rectangle(overlay_frame, cell_tl, cell_br, (255, 0, 0), 1)

        # --- Draw ArUco markers and fill overlapping cells ---
        for cx, cy, corner, marker_id in marker_info:
            int_corners = corner.astype(np.int32)
            cv2.polylines(overlay_frame, [int_corners], True, (0, 255, 0), 2)
            cv2.circle(overlay_frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(overlay_frame, f"ID:{marker_id}", (cx+10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            if grid_cells:
                # Marker bounding box
                m_tl = (int(corner[0][:,0].min()), int(corner[0][:,1].min()))
                m_br = (int(corner[0][:,0].max()), int(corner[0][:,1].max()))

                for cell_tl, cell_br in grid_cells:
                    overlap_x = max(0, min(m_br[0], cell_br[0]) - max(m_tl[0], cell_tl[0]))
                    overlap_y = max(0, min(m_br[1], cell_br[1]) - max(m_tl[1], cell_tl[1]))
                    if overlap_x > 0 and overlap_y > 0:
                        # Fill overlapping cell in red
                        cv2.rectangle(overlay_frame, cell_tl, cell_br, (0,0,255), -1)

        # Combine overlay transparency
        overlay_frame = cv2.addWeighted(overlay_frame, 0.6, frame, 0.4, 0)

        # --- Show windows ---
        cv2.imshow("Detection Feed", detection_frame)  # For detection only
        cv2.imshow("Overlay Feed", overlay_frame)      # Shows grid + red filled cells

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    liveFeedRedGridWithAruco()