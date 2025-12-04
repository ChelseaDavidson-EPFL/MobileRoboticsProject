from thymio import Thymio
import numpy as np

DISTANCE_WHEELS = 9.5 #in cm

# ----------------------------------------------------
# Extended Kalman Filter prediction + correction step
# x = [x, y, theta, velocity]
# ----------------------------------------------------

def kallman(x_est_prev, P_est_prev, v, omega, Q, Ts, pos_meas, R):     #x = [x, y, theta, velocity]

    # -------------------------------
    # PREDICTION STEP (motion model)
    # -------------------------------
    # Nonlinear state update based on robot kinematics
    x_next = x_est_prev[0]+v*np.cos(x_est_prev[2])*Ts
    y_next = x_est_prev[1]+v*np.sin(x_est_prev[2])*Ts
    theta_next = x_est_prev[2]+omega*Ts
    v_next = v # velocity directly computed at each step by filter_pos() function

    # Predicted (a priori) state
    x_est_a_priori = np.array([x_next, y_next, theta_next, v_next])

    # Jacobian of the motion model
    F = np.array([[1, 0, -v*np.sin(x_est_prev[2])*Ts,  np.cos(x_est_prev[2])*Ts],
                  [0, 1,  v*np.cos(x_est_prev[2])*Ts,  np.sin(x_est_prev[2])*Ts],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    # Predicted covariance
    P_est_a_priori = np.dot(F, np.dot(P_est_prev, F.T)) + Q
 
    # -------------------------------------------------
    # If NO measurement (=camera hidden): return predicted state only
    # -------------------------------------------------
    if pos_meas[0] is None:
        # Normalize angular innovation to [-pi, pi]
        if x_est_a_priori[2]>np.pi:
            x_est_a_priori[2] -= 2*np.pi
        if x_est_a_priori[2]<-np.pi:
            x_est_a_priori[2] += 2*np.pi

        return x_est_a_priori, P_est_a_priori

    
    # --------------------------
    # CORRECTION STEP
    # --------------------------
    
    # Measurement vector (x, y, theta), data from camera
    y = np.array([pos_meas[0], pos_meas[1], pos_meas[2]])

    # Measurement matrix to extract x, y, theta from state vector
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0]])

    # Innovation (measurement - predicted measurement)
    i = y-np.dot(H, x_est_a_priori)
        
    # Normalize angular innovation to [-pi, pi]
    while i[2] > np.pi:
        i[2] -= 2*np.pi
    while i[2] < -np.pi:
        i[2] += 2*np.pi
        
    # Innovation covariance
    S = np.dot(H, np.dot(P_est_a_priori, H.T)) + R
    # Kalman gain
    K = np.dot(P_est_a_priori, np.dot(H.T, np.linalg.inv(S)))
        
    # Updated state estimate and covariance
    x_est = x_est_a_priori + np.dot(K, i)
    P_est = P_est_a_priori - np.dot(K, np.dot(H, P_est_a_priori))
        
    # Normalize final theta
    if x_est[2]>np.pi:
        x_est[2] -= 2*np.pi
    if x_est[2]<-np.pi:
        x_est[2] += 2*np.pi

    return x_est, P_est


# ---------------------------------------------------
# Initialize EKF state and covariance
# Call once when starting, in pmain before the loop
# ---------------------------------------------------

def init_filter(q_x, q_y, q_theta, q_v, r_x, r_y, r_theta, initial_pos=None, initial_orient=None): #initial_pos = [x, y, orient]
    if initial_pos is None or initial_orient is None:
        x_est = np.array([0., 0., 0., 0.])
    else:
        x_est = np.array([initial_pos[0], initial_pos[1], initial_orient, 0.])
    
    # Very large initial uncertainty
    P_est = 1000 * np.eye(4)
    # Process noise covariance
    Q=np.array([[q_x, 0, 0, 0],
                [0, q_y, 0, 0],
                [0, 0, q_theta, 0],
                [0, 0, 0, q_v]])
    # Measurement noise covariance
    R=np.array([[r_x, 0, 0],
                [0, r_y, 0],
                [0, 0, r_theta]])  # Use same variance for theta measurement

    return x_est, P_est, Q, R


# --------------------------------------------------------------
# EKF update wrapper used inside main loop of the robot
# Called at each loop of the main, with or without camera data
# --------------------------------------------------------------

def filter_pos(thym: Thymio, pos_on_img, x_est, P_est, Q, R, Ts, RATIO_SPEED):

    # Convert motor speeds into translational and angular velocity
    motor_speed=thym.motor_speeds
    v = (motor_speed[0]+motor_speed[1])/(2*RATIO_SPEED)
    omega = (motor_speed[1]-motor_speed[0])/(RATIO_SPEED*DISTANCE_WHEELS)
    
    # Perform EKF update
    new_x_est, new_P_est = kallman(x_est, P_est, v, omega, Q, Ts, pos_on_img, R)

    return new_x_est, new_P_est