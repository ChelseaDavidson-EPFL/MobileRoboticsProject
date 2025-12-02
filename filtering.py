from thymio import Thymio
import numpy as np

DISTANCE_WHEELS = 9.5 #in cm

def kallman(x_est_prev, P_est_prev, v, omega, Q, Ts, pos_meas, R):     #x = [x, y, theta, velocity]
    x_next = x_est_prev[0]+v*np.cos(x_est_prev[2])*Ts
    y_next = x_est_prev[1]+v*np.sin(x_est_prev[2])*Ts
    theta_next = x_est_prev[2]+omega*Ts
    v_next = v
    x_est_a_priori = np.array([x_next, y_next, theta_next, v_next])

    #jacobian (derivatives of states)
    F = np.array([[1, 0, -v*np.sin(x_est_prev[2])*Ts,  np.cos(x_est_prev[2])*Ts],
                  [0, 1,  v*np.cos(x_est_prev[2])*Ts,  np.sin(x_est_prev[2])*Ts],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    P_est_a_priori = np.dot(F, np.dot(P_est_prev, F.T)) + Q
 
    if pos_meas[0] is None:
        if x_est_a_priori[2]>np.pi:
            x_est_a_priori[2] -= 2*np.pi
        if x_est_a_priori[2]<-np.pi:
            x_est_a_priori[2] += 2*np.pi
        return x_est_a_priori, P_est_a_priori

    else:
        # Measurement includes x, y, and theta
        y = np.array([pos_meas[0], pos_meas[1], pos_meas[2]])

        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0]])

        i = y-np.dot(H, x_est_a_priori)
        
        # Normalize angle innovation to [-pi, pi]
        while i[2] > np.pi:
            i[2] -= 2*np.pi
        while i[2] < -np.pi:
            i[2] += 2*np.pi
        
        S = np.dot(H, np.dot(P_est_a_priori, H.T)) + R
        K = np.dot(P_est_a_priori, np.dot(H.T, np.linalg.inv(S)))

        # print(f"P_diag: [{P_est_a_priori[0,0]:.2e}, {P_est_a_priori[1,1]:.2e}, {P_est_a_priori[2,2]:.2e}]")
        # print(f"K_xy: {K[:2,:2]}, K_theta: {K[2,2]:.3f}")
        # print(f"Innovation: x={i[0]:.2f}, y={i[1]:.2f}, theta={i[2]:.3f}")
        
        x_est = x_est_a_priori + np.dot(K, i)
        P_est = P_est_a_priori - np.dot(K, np.dot(H, P_est_a_priori))
        #P_est = (np.eye(4) - K @ H) @ P_est_a_priori @ (np.eye(4) - K @ H).T + K @ R @ K.T
        
        if x_est[2]>np.pi:
            x_est[2] -= 2*np.pi
        if x_est[2]<-np.pi:
            x_est[2] += 2*np.pi
        return x_est, P_est


############################################################
#                                                          #
# Call at the begining or when restarting all if kidnapped #
#                                                          #
############################################################

def init_filter(q_x, q_y, q_theta, q_v, r_x, r_y, initial_pos=None, initial_orient=None): #initial_pos = [x, y, orient]
    if initial_pos is None or initial_orient is None:
        x_est = np.array([0., 0., 0., 0.])
    else:
        x_est = np.array([initial_pos[0], initial_pos[1], initial_orient, 0.])
    P_est = 1000 * np.eye(4)
    Q=np.array([[q_x, 0, 0, 0],
                [0, q_y, 0, 0],
                [0, 0, q_theta, 0],
                [0, 0, 0, q_v]])
    R=np.array([[r_x, 0, 0],
                [0, r_y, 0],
                [0, 0, q_theta]])  # Use same variance for theta measurement

    return x_est, P_est, Q, R


#########################################################
#                                                       #
# Call at each new position measured in the moving loop #
#                                                       #
#########################################################

def filter_pos(thym: Thymio, pos_on_img, x_est, P_est, Q, R, Ts, RATIO_SPEED):
    #thym.pos = pos_on_img
    #thym.orient = orient_on_img
    motor_speed=thym.motor_speeds
    v = (motor_speed[0]+motor_speed[1])/(2*RATIO_SPEED)
    omega = (motor_speed[1]-motor_speed[0])/(RATIO_SPEED*DISTANCE_WHEELS)
    new_x_est, new_P_est = kallman(x_est, P_est, v, omega, Q, Ts, pos_on_img, R)

    # print(f"Motor speeds: L={motor_speed[0]}, R={motor_speed[1]}, v={v:.2f} cm/s, omega={omega:.2f} rad/s")
    return new_x_est, new_P_est



# 1. Initialize filter
#x_est, P_est, Q, R = init_filter(q_x, q_y, q_theta, q_v, r_x, r_y, initial_pos=None)
#Ts = cst

# 2. Inside your robot loop
#while bot_is_running:
    #pos_on_img, orient_on_img = pos_measured_camera

    # 3. Update Kalman filter
    #x_est, P_est = filter_pos(thym, pos_on_img, orient_on_img, x_est, P_est, Q, R, Ts)

    # 4. Use filtered position
    #filtered_pos = x_est[:2]
    #filtered_theta = x_est[2]

    # Continue with other robot logic
    #move_robot(filtered_pos)