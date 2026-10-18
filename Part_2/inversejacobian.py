import math
import numpy as np
import general_robotics_toolbox as rox


def jacobian_inverse(robot, q0, Rd, Pd, Nmax, alpha, tol):

    # Initialize
    n = len(q0)
    q = np.zeros((n, Nmax + 1))
    q[:, 0] = q0

    iternum = 0

    # forward kinematics at initial configuration
    H = rox.fwdkin(robot, q0)
    R = H.R
    P = np.array(H.p).reshape(3, 1)

    # initial position and orientation errors
    eP = 0.5 * np.linalg.norm(P - Pd)**2

    ER = R @ Rd.T
    eR = np.array(rox.R2rpy(ER)).reshape(3, 1)

    # Combined error vector: [roll, pitch, yaw, x, y, z]
    dX = np.vstack((eR, P - Pd))

    # jacobian at the initial configuration
    J = rox.robotjacobian(robot, q0)

    # Iterate while any error exceeds its tolerance
    while (np.abs(dX) > tol).any():
        # stops if maximum number of iterations is reached
        if iternum >= Nmax:
            break
        # pseudo inverse
        Uq = np.linalg.pinv(J) @ dX
        # Update the joint configuration
        q[:, iternum + 1] = (
            q[:, iternum] - alpha * Uq.flatten()
        )
        iternum += 1

        # Recalculate forward kinematics
        H = rox.fwdkin(robot, q[:, iternum])
        R = H.R
        P = np.array(H.p).reshape(3, 1)

        # Recalculate position and orientation errors
        eP = 0.5 * np.linalg.norm(P - Pd)**2

        ER = R @ Rd.T
        eR = np.array(rox.R2rpy(ER)).reshape(3, 1)

        dX = np.vstack((eR, P - Pd))

        # Recalculate the Jacobian at the new configuration
        J = rox.robotjacobian(robot, q[:, iternum])

    # Return the final joint configuration
    return q[:, iternum]


def main():

    # desired orientation
    Rd = np.array([
        [-0.5856, -0.5256, -0.6171],
        [-0.4575,  0.8428, -0.2836],
        [ 0.6691,  0.1163, -0.734 ]
    ])

    # desired position
    Pd = np.array([
        [0.1405],
        [0.1098],
        [0.1014]
    ])

    # cinvert initial guess of joint angles to radians
    q0 = np.array([38, 90, 0, 48, 9]) * math.pi / 180

    # Tolerances: [roll, pitch, yaw, x, y, z]
    tol = np.array([
        [0.02],
        [0.02],
        [0.02],
        [0.001],
        [0.001],
        [0.001]
    ])

    Nmax = 200
    alpha = 0.1

    # Define joint lengths in meters
    l0 = 0.061 # base to servo 1
    l1 = 0.0435 # servo 1 to servo 2
    l2 = 0.08285 # servo 2 to servo 3
    l3 = 0.08285 # servo 3 to servo 4
    l4 = 0.07385 # servo 4 to servo 5
    l5 = 0.05457 # servo 5 to gripper

    # Define unit vectors
    ex = np.array([1, 0, 0])
    ey = np.array([0, 1, 0])
    ez = np.array([0, 0, 1])

    # Define position vectors from frame i-1 to frame i
    P01 = (l0+l1)*ez # translation between base frame and 1 frame in base frame
    P12 = np.zeros(3,) # translation between 1 and 2 frame in 1 frame
    P23 = l2*ex # translation between 2 and 3 frame in 2 frame
    P34 = -l3*ez # translation between 3 and 4 frame in 3 frame
    P45 = np.zeros(3,) # translation between 4 and 5 frame in 4 frame
    P5T = -(l4+l5)*ex # translation between 5 and tool frame in 5 frames

    # joint axes
    H = np.array([
        ez,
        -ey,
        -ey,
        -ey,
        -ex
    ]).T

    # position vectors
    P = np.array([P01,
        P12,
        P23,
        P34,
        P45,
        P5T
    ]).T

    # All joints are revolute
    joint_type = [0, 0, 0, 0, 0]

    # Create the robot object
    robot = rox.Robot(H, P, joint_type)

    # inverse jacbobian
    q = jacobian_inverse(
        robot, q0, Rd, Pd, Nmax, alpha, tol
    )

    # Convert solution to degrees
    q = q * 180 / math.pi

    np.set_printoptions(suppress=True)
    print(np.round(q, 2))

if __name__ == "__main__":
    main()