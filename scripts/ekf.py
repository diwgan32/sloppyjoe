import numpy as np
from numpy import sin, cos
import scipy.linalg    # you may find scipy.linalg.block_diag useful
from ExtractLines import ExtractLines, normalize_line_parameters, angle_difference
from maze_sim_parameters import LineExtractionParams, NoiseParams, MapParams

class EKF(object):

    def __init__(self, x0, P0, Q):
        self.x = x0    # Gaussian belief mean
        self.P = P0    # Gaussian belief covariance
        self.Q = Q     # Gaussian control noise covariance (corresponding to dt = 1 second)

    # Updates belief state given a discrete control step (Gaussianity preserved by linearizing dynamics)
    # INPUT:  (u, dt)
    #       u - zero-order hold control input
    #      dt - length of discrete time step
    # OUTPUT: none (internal belief state (self.x, self.P) should be updated)
    def transition_update(self, u, dt):
        g, Gx, Gu = self.transition_model(u, dt)
        P1 = self.P
        Q1 = self.Q
        x2 = g
        P2 = np.matmul(np.matmul(Gx,P1),np.transpose(Gx)) + dt*np.matmul(np.matmul(Gu,Q1), np.transpose(Gu))
        self.P = P2
        self.x = x2
        #### TODO ####
        # update self.x, self.P
        ##############

    # Propagates exact (nonlinear) state dynamics; also returns associated Jacobians for EKF linearization
    # INPUT:  (u, dt)
    #       u - zero-order hold control input
    #      dt - length of discrete time step
    # OUTPUT: (g, Gx, Gu)
    #      g  - result of belief mean self.x propagated according to the system dynamics with control u for dt seconds
    #      Gx - Jacobian of g with respect to the belief mean self.x
    #      Gu - Jacobian of g with respect to the control u
    def transition_model(self, u, dt):
        raise NotImplementedError("transition_model must be overriden by a subclass of EKF")

    # Updates belief state according to a given measurement (with associated uncertainty)
    # INPUT:  (rawZ, rawR)
    #    rawZ - raw measurement mean
    #    rawR - raw measurement uncertainty
    # OUTPUT: none (internal belief state (self.x, self.P) should be updated)
    def measurement_update(self, rawZ, rawR):
        z, R, H = self.measurement_model(rawZ, rawR)
        if z is None:    # don't update if measurement is invalid (e.g., no line matches for line-based EKF localization)
            return

        #### TODO ####
        # update self.x, self.P
        ##############

        sigma = np.matmul(H,np.matmul(self.P, np.transpose(H))) + R
        K = np.matmul(self.P, np.matmul(np.transpose(H),np.linalg.inv(sigma)))

        self.x = self.x + np.transpose(np.matmul(K,z))
        self.x = self.x[0]

        self.P = self.P - np.matmul(K, np.matmul(sigma, np.transpose(K)))


    # Converts raw measurement into the relevant Gaussian form (e.g., a dimensionality reduction);
    # also returns associated Jacobian for EKF linearization
    # INPUT:  (rawZ, rawR)
    #    rawZ - raw measurement mean
    #    rawR - raw measurement uncertainty
    # OUTPUT: (z, R, H)
    #       z - measurement mean (for simple measurement models this may = rawZ)
    #       R - measurement covariance (for simple measurement models this may = rawR)
    #       H - Jacobian of z with respect to the belief mean self.x
    def measurement_model(self, rawZ, rawR):
        raise NotImplementedError("measurement_model must be overriden by a subclass of EKF")


class Localization_EKF(EKF):

    def __init__(self, x0, P0, Q, map_lines, tf_base_to_camera, g):
        self.map_lines = map_lines                    # 2xJ matrix containing (alpha, r) for each of J map lines
        self.tf_base_to_camera = tf_base_to_camera    # (x, y, theta) transform from the robot base to the camera frame
        self.g = g                                    # validation gate
        super(self.__class__, self).__init__(x0, P0, Q)

    # Unicycle dynamics (Turtlebot 2)
    def transition_model(self, u, dt):
        v, om = u
        x, y, th = self.x

        #### TODO ####
        # compute g, Gx, Gu
        ##############
        #Calculate g from u and m
        th2 = om*dt + th
        x2 = 0.0
        y2 = 0.0
        Gx = np.zeros((3,3))
        Gu = np.zeros((3,2))
        if(np.absolute(om) < 0.00001):
            x2 = v*cos(th)*dt + x
            y2 = v*sin(th)*dt + y
            Gx[0,:] = [1.0, 0.0, -v*np.sin(th2)*dt]
            Gx[1,:] = [0.0, 1.0, v*np.cos(th2)*dt]
            Gx[2,:] = [0.0, 0.0, 1.0]
            Gu[0,:] = [np.cos(th2)*dt, 0.0]
            Gu[1,:] = [np.sin(th2)*dt, 0.0]
            Gu[2,:] = [0.0, dt]
        else:
            x2 = v*(np.sin(th2) - np.sin(th))/om + x
            y2 = -v*(np.cos(th2) - np.cos(th))/om + y
            Gx[0,:] = [1.0, 0.0, v/om*(np.cos(th2) - np.cos(th))]
            Gx[1,:] = [0.0, 1.0, v/om*(np.sin(th2) - np.sin(th))]
            Gx[2,:] = [0.0, 0.0, 1.0]
            Gu[0,:] = [(np.sin(th2) - np.sin(th))/om, v*np.cos(th2)*dt/om - v*np.sin(th2)/(om*om) + v*np.sin(th)/(om*om)]
            Gu[1,:] = [-(np.cos(th2) - np.cos(th))/om, v*np.sin(th2)*dt/om + v*cos(th2)/(om*om) - v*cos(th)/(om*om)]
            Gu[2,:] = [0.0, dt]
        
        g = np.array([x2, y2, th2])
        return g, Gx, Gu

    # Given a single map line m in the world frame, outputs the line parameters in the scanner frame so it can
    # be associated with the lines extracted from the scanner measurements
    # INPUT:  m = (alpha, r)
    #       m - line parameters in the world frame
    # OUTPUT: (h, Hx)
    #       h - line parameters in the scanner (camera) frame
    #      Hx - Jacobian of h with respect to the the belief mean self.x
    def map_line_to_predicted_measurement(self, m):
        alpha, r = m


        #### TODO ####
        # compute h, Hx
        ##############
        x_cart, y_cart, th_cart = self.x
        dx, dy, dth = self.tf_base_to_camera

        th_cam = dth + th_cart


        x_cam = x_cart + dx*np.cos(th_cart) - dy*np.sin(th_cart)
        y_cam = y_cart + dx*np.sin(th_cart) + dy*np.cos(th_cart)
        


        r2_val = (np.cos(alpha)*x_cam + np.sin(alpha)*y_cam - r)/np.sqrt(np.square(np.cos(alpha)) + np.square(np.sin(alpha)))
        r2 = np.absolute(r2_val)
        r2_sign = np.sign(r2_val)

        alpha2 = 0
        a = 0
        b = 0
        if(r2_sign == 1):
            alpha2 = alpha - th_cam + np.pi
            a = np.cos(alpha)
            b = np.sin(alpha)
        else:
            alpha2 = alpha - th_cam
            a = -np.cos(alpha)
            b = -np.sin(alpha)
        

        h = (alpha2, r2)
        Hx = np.zeros((2,3))

        
        Hx[:,0] = [0.0, a]
        Hx[:,1] = [0.0, b]
        #Hx[:,0] = [0.0, -np.cos(alpha)]
        #Hx[:,1] = [0.0, -np.sin(alpha)]
        partial_theta = a*(-dx*np.sin(th_cart) - dy*np.cos(th_cart)) + b*(dx*np.cos(th_cart) - dy*np.sin(th_cart))

        Hx[:,2] = [-1.0, partial_theta]


        flipped, h = normalize_line_parameters(h)
        if flipped:
            Hx[1,:] = -Hx[1,:]

        return h, Hx

    # Given lines extracted from the scanner data, tries to associate to each one the closest map entry
    # measured by Mahalanobis distance
    # INPUT:  (rawZ, rawR)
    #    rawZ - 2xI matrix containing (alpha, r) for each of I lines extracted from the scanner data (in scanner frame)
    #    rawR - list of I 2x2 covariance matrices corresponding to each (alpha, r) column of rawZ
    # OUTPUT: (v_list, R_list, H_list)
    #  v_list - list of at most I innovation vectors (predicted map measurement - scanner measurement)
    #  R_list - list of len(v_list) covariance matrices of the innovation vectors (from scanner uncertainty)
    #  H_list - list of len(v_list) Jacobians of the innovation vectors with respect to the belief mean self.x
    def associate_measurements(self, rawZ, rawR):

        #### TODO ####
        # compute v_list, R_list, H_list
        ##############
        m = self.map_lines
        
        v_list = np.zeros((rawZ.shape[1],2))
        R_list = np.zeros((rawZ.shape[1], 2, 2))
        H_list = np.zeros((rawZ.shape[1], 2, 3))

        del_list = []
        
        for k in range(0,rawZ.shape[1]):
            v_k = np.zeros((2,1))
            min_d = self.g*self.g + 1000
            R_k = np.zeros((2, 2))
            H_k = np.zeros((2,2))

            z_k = rawZ[:,k]
            for i in range(0,m.shape[1]):
                m_i = np.array([m[0,i], m[1,i]])

                (h, Hx) = self.map_line_to_predicted_measurement(m_i)
                v = z_k - h
            
                S = np.matmul(Hx, np.matmul(self.P, np.transpose(Hx))) + rawR[k]
                #print S
                d = np.matmul(np.transpose(v),np.matmul(np.linalg.inv(S), v))
                if(d < min_d):
                    min_d = d
                    v_k = v
                    R_k = rawR[k]
                    H_k = Hx

            if(min_d < self.g*self.g):

                v_list[k,:] = v_k
                R_list[k,:,:] = R_k
                H_list[k,:,:] = H_k
            else:
                del_list.append(k)

        #print v_list
        #print del_list
        


        for k in range(0,len(del_list)):
            n = len(del_list) - k - 1
            idx = del_list[n]

            v_list = np.delete(v_list, (idx), axis=0)
            R_list = np.delete(R_list, (idx), axis=0)
            H_list = np.delete(H_list, (idx), axis=0)



        
        v = []
        R = []
        H = []
        for k in range(0, v_list.shape[0]):

            v.append(v_list[k,:])
            R.append(R_list[k,:])
            H.append(H_list[k,:])

        v_list = v
        R_list = R
        H_list = H



        return v_list, R_list, H_list

    # Assemble one joint measurement, covariance, and Jacobian from the individual values corresponding to each
    # matched line feature
    def measurement_model(self, rawZ, rawR):
        v_list, R_list, H_list = self.associate_measurements(rawZ, rawR)
        if not v_list:
            print "Scanner sees", rawZ.shape[1], "line(s) but can't associate them with any map entries"
            return None, None, None

        #### TODO ####
        # compute z, R, H
        ##############

        z = np.zeros((2*len(v_list),1))
        R = R_list[0]
        H = np.zeros((2*len(v_list),3))

        for k in range(0, len(v_list)):
            z[2*k:2*k+2] = v_list[k].reshape(2,1)
            if(k == 0):
                pass
            else:
                R = scipy.linalg.block_diag(R, R_list[k])
            H[2*k:2*k+2,:] = H_list[k]


        return z, R, H


class SLAM_EKF(EKF):

    def __init__(self, x0, P0, Q, tf_base_to_camera, g):
        self.tf_base_to_camera = tf_base_to_camera    # (x, y, theta) transform from the robot base to the camera frame
        self.g = g                                    # validation gate
        super(self.__class__, self).__init__(x0, P0, Q)

    # Combined Turtlebot + map dynamics
    # Adapt this method from Localization_EKF.transition_model.
    def transition_model(self, u, dt):
        v, om = u
        x, y, th = self.x[:3]

        #### TODO ####
        # compute g, Gx, Gu (some shape hints below)
        # g = np.copy(self.x)
        # Gx = np.eye(self.x.size)
        # Gu = np.zeros((self.x.size, 2))
        ##############
        g = np.copy(self.x)
        Gx = np.eye(self.x.size)
        Gu = np.zeros((self.x.size,2))
        th2 = om*dt + th
        x2 = 0.0
        y2 = 0.0
        if(np.absolute(om) < 0.00001):
            x2 = v*cos(th)*dt + x
            y2 = v*sin(th)*dt + y
            Gx[0,2] = -v*np.sin(th2)*dt
            Gx[1,2] = v*np.cos(th2)*dt
            Gu[0,0] = np.cos(th2)*dt
            Gu[1,0] = np.sin(th2)*dt
            Gu[2,1] = dt
        else:
            x2 = v*(np.sin(th2) - np.sin(th))/om + x
            y2 = -v*(np.cos(th2) - np.cos(th))/om + y
            
            Gx[0,2] = v/om*(np.cos(th2) - np.cos(th))
            Gx[1,2] = v/om*(np.sin(th2) - np.sin(th))
            #print np.size([(np.sin(th2) - np.sin(th))/om, v*np.cos(th2)*dt/om - v*np.sin(th2)/(om*om) + v*np.sin(th)/(om*om)])
            Gu[0,0] = (np.sin(th2) - np.sin(th))/om
            Gu[0,1] =  v*np.cos(th2)*dt/om - v*np.sin(th2)/(om*om) + v*np.sin(th)/(om*om)
            Gu[1,0] = -(np.cos(th2) - np.cos(th))/om

            Gu[1,1] = v*np.sin(th2)*dt/om + v*cos(th2)/(om*om) - v*cos(th)/(om*om)
            Gu[2,1] = dt
        

        g[0:3] = [x2, y2, th2]





        return g, Gx, Gu

    # Combined Turtlebot + map measurement model
    # Adapt this method from Localization_EKF.measurement_model.
    #
    # The ingredients for this model should look very similar to those for Localization_EKF.
    # In particular, essentially the only thing that needs to change is the computation
    # of Hx in map_line_to_predicted_measurement and how that method is called in
    # associate_measurements (i.e., instead of getting world-frame line parameters from
    # self.map_lines, you must extract them from the state self.x)
    def measurement_model(self, rawZ, rawR):
        v_list, R_list, H_list = self.associate_measurements(rawZ, rawR)
        if not v_list:
            print "Scanner sees", rawZ.shape[1], "line(s) but can't associate them with any map entries"
            return None, None, None

        #### TODO ####
        # compute z, R, H (should be identical to Localization_EKF.measurement_model above)
        ##############
        z = np.zeros((2*len(v_list),1))
        R = R_list[0]
        H = np.zeros((2*len(v_list),self.x.size))

        for k in range(0, len(v_list)):
            z[2*k:2*k+2] = v_list[k].reshape(2,1)
            if(k == 0):
                pass
            else:
                R = scipy.linalg.block_diag(R, R_list[k])
            H[2*k:2*k+2,:] = H_list[k]

        return z, R, H

    # Adapt this method from Localization_EKF.map_line_to_predicted_measurement.
    #
    # Note that instead of the actual parameters m = (alpha, r) we pass in the map line index j
    # so that we know which components of the Jacobian to fill in.
    def map_line_to_predicted_measurement(self, j):
        alpha, r = self.x[(3+2*j):(3+2*j+2)]    # j is zero-indexed! (yeah yeah I know this doesn't match the pset writeup)

        #### TODO ####
        # compute h, Hx (you may find the skeleton for computing Hx below useful)
        x_cart, y_cart, th_cart = self.x[:3]
        dx, dy, dth = self.tf_base_to_camera

        th_cam = dth + th_cart


        x_cam = x_cart + dx*np.cos(th_cart) - dy*np.sin(th_cart)
        y_cam = y_cart + dx*np.sin(th_cart) + dy*np.cos(th_cart)
        


        r2_val = (np.cos(alpha)*x_cam + np.sin(alpha)*y_cam - r)/np.sqrt(np.square(np.cos(alpha)) + np.square(np.sin(alpha)))
        r2 = np.absolute(r2_val)
        r2_sign = np.sign(r2_val)

        alpha2 = 0
        a = 0
        b = 0
        if(r2_sign == 1):
            alpha2 = alpha - th_cam + np.pi
            a = np.cos(alpha)
            b = np.sin(alpha)
        else:
            alpha2 = alpha - th_cam
            a = -np.cos(alpha)
            b = -np.sin(alpha)
        

        h = (alpha2, r2)



        Hx = np.zeros((2,3))

        Hx = np.zeros((2,self.x.size))
        Hx[:,0] = [0.0, a]
        Hx[:,1] = [0.0, b]
        partial_theta = a*(-dx*np.sin(th_cart) - dy*np.cos(th_cart)) + b*(dx*np.cos(th_cart) - dy*np.sin(th_cart))

        Hx[:,2] = [-1.0, partial_theta]
        # First two map lines are assumed fixed so we don't want to propagate any measurement correction to them
        if j > 1:
            Hx[0, 3+2*j] = 1.0
            Hx[1, 3+2*j] = 0.0
            Hx[0, 3+2*j+1] = 0.0
            Hx[1, 3+2*j+1] = 1.0

        ##############

        flipped, h = normalize_line_parameters(h)
        if flipped:
            Hx[1,:] = -Hx[1,:]

        return h, Hx

    # Adapt this method from Localization_EKF.associate_measurements.
    def associate_measurements(self, rawZ, rawR):

        #### TODO ####
        # compute v_list, R_list, H_list
        ##############
        m = self.x[3:]
        
        v_list = np.zeros((rawZ.shape[1],2))
        R_list = np.zeros((rawZ.shape[1], 2, 2))
        H_list = np.zeros((rawZ.shape[1], 2, self.x.size))

        del_list = []
        
        for k in range(0,rawZ.shape[1]):
            v_k = np.zeros((2,1))
            min_d = self.g*self.g + 1000
            R_k = np.zeros((2, 2))
            H_k = np.zeros((2,self.x.size))

            z_k = rawZ[:,k]
            for i in range(0,m.size/2):
                m_i = np.array([m[2*i], m[2*i + 1]])

                (h, Hx) = self.map_line_to_predicted_measurement(i)
                v = z_k - h
            
                S = np.matmul(Hx, np.matmul(self.P, np.transpose(Hx))) + rawR[k]
                #print S
                d = np.matmul(np.transpose(v),np.matmul(np.linalg.inv(S), v))
                if(d < min_d):
                    min_d = d
                    v_k = v
                    R_k = rawR[k]
                    H_k = Hx

            if(min_d < self.g*self.g):

                v_list[k,:] = v_k
                R_list[k,:,:] = R_k
                H_list[k,:,:] = H_k
            else:
                del_list.append(k)

        #print v_list
        #print del_list
        


        for k in range(0,len(del_list)):
            n = len(del_list) - k - 1
            idx = del_list[n]

            v_list = np.delete(v_list, (idx), axis=0)
            R_list = np.delete(R_list, (idx), axis=0)
            H_list = np.delete(H_list, (idx), axis=0)



        
        v = []
        R = []
        H = []
        for k in range(0, v_list.shape[0]):

            v.append(v_list[k,:])
            R.append(R_list[k,:])
            H.append(H_list[k,:])

        v_list = v
        R_list = R
        H_list = H

        return v_list, R_list, H_list
