from thymio import Thymio
import numpy as np

distance_wheels = 95 #in mm

def filter_pos(thym: Thymio, pos_on_img, orient_on_img):
    thym.pos = pos_on_img
    thym.orient = orient_on_img
    return 
    

def get_data(thym, ratio_speed):
    pos = Thymio.get(thym, pos)
    motor_speed = Thymio.get(thym, motor_speed)
    v = (motor_speed[0]+motor_speed[1])/2
    omega = (motor_speed[0]-motor_speed[1])/(ratio_speed*distance_wheels)
    return pos, v, omega

def kallman(x_est_prev, P_est_prev, v, omega, Q, Ts, pos_meas, R):     #x = [x, y, theta, velocity]
    x_next = x_est_prev[0]+v*np.cos(x_est_prev[2])*Ts
    y_next = x_est_prev[1]+v*np.sin(x_est_prev[2])*Ts
    theta_next = x_est_prev[2]+omega*Ts
    v_next = v

    x_est_a_priori = np.array([x_next, y_next, theta_next, v_next])

    #jacobian (derivatives of states)
    F = np.array([[1, 0, -v*np.sin(x_est_a_priori[2])*Ts,  np.cos(x_est_a_priori[2])*Ts],
                  [0, 1,  v*np.cos(x_est_a_priori[2])*Ts,  np.sin(x_est_a_priori[2])*Ts],
                  [0, 0, 1, 0],
                  [0, 0, 0, 0]])

    P_est_a_priori = np.dot(F, np.dot(P_est_prev, F.T)) + Q
 
    y = np.array([pos_meas[0], pos_meas[1]])

    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])

    i = y-np.dot(H, x_est_a_priori)
    S = np.dot(H, np.dot(P_est_a_priori, H.T)) + R
    K = np.dot(P_est_a_priori, np.dot(H.T, np.linalg.inv(S)))

    x_est = x_est_a_priori + np.dot(K, i)
    P_est = P_est_a_priori - np.dot(K, np.dot(H, P_est_a_priori))
    return x_est, P_est

def use_kallman(thym, state, ratio_speed):
    x_est = [np.array([[0], [0], [0], [0]])]
    P_est = [1000 * np.ones(4)]
    R = np.ones(2) #prout
    Ts = 10 # dans le dossier général ? sinon s'accorder pour utiliser le même partout
    Q = np.ones(4) #prout

    while state: #faut vraiment trouver autre chose -> while state: with state=0 stop and state=1 start ?

        last_pos, v, omega = get_data(thym, ratio_speed)
        new_x_est, new_P_est = kallman(x_est[-1], P_est[-1], v, omega, Q, Ts, last_pos, R)
        x_est.append(new_x_est)
        P_est.append(new_P_est)