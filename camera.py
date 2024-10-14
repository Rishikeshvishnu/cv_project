import numpy as np
from misc import (
    compute_reprojection_error, compute_reprojection_errors,
    compute_reprojection_residuals, msac, num_in_front
)

# Takes a list of cameras P2s, and returns the one with the largest number of points
# in front of the cameras P1 = (I 0) and P2
def find_best_camera(P2s, x1s, x2s):
    P1 = np.column_stack([np.identity(3), np.zeros((3,1))])

    mostPtsInfront = 0
    P2 = P2s[0]
    for i in range(0, len(P2s)):
        ptsInfront = num_in_front(P1, P2s[i], x1s, x2s)

        if ptsInfront > mostPtsInfront:
            mostPtsInfront = ptsInfront
            P2 = P2s[i]

    return P2

# Given a rotation matrix R, two 3D points and their corresponding image points on
# camera P=(R | t), compute t using DLT.
def estimate_t_DLT(R, X1, X2, x1, x2):
    M = np.array([
        [0, X1[3] * x1[2], -X1[3] * x1[1], (R[1,:] @ X1[:3]) * x1[2] - (R[2,:] @ X1[:3]) * x1[1]],
        [-X1[3] * x1[2], 0, X1[3] * x1[0], (R[2,:] @ X1[:3]) * x1[0] - (R[0,:] @ X1[:3]) * x1[2]],
        [0, X2[3] * x2[2], -X2[3] * x2[1], (R[1,:] @ X2[:3]) * x2[2] - (R[2,:] @ X2[:3]) * x2[1]],
        [-X2[3] * x2[2], 0, X2[3] * x2[0], (R[2,:] @ X2[:3]) * x2[0] - (R[0,:] @ X2[:3]) * x2[2]]
    ])
    (_,_,VT) = np.linalg.svd(M)
    V = np.transpose(VT)
    t = V[:,-1]
    t = (t / t[-1])[:3]

    return t

# Estimate t for a camera matrix P=K(R | t) from n 3d points Xs (4xn),
# and their corresponding calibrated image points xs (3xn), using MSAC.
def estimate_t_robust(R, Xs, xs, threshold, alpha=0.99):
    def estimate_fn(rand_indices):
        i1, i2 = rand_indices
        return [estimate_t_DLT(R, Xs[:,i1], Xs[:,i2], xs[:,i1], xs[:,i2])]

    def err_fn(t, indices):
        P = np.column_stack([R, t])
        return compute_reprojection_errors(P, Xs[:,indices], xs[:,indices])

    n = Xs.shape[1]
    print("Running MSAC to estimate t n = ", n)
    return msac(estimate_fn, err_fn, n, 2, 1, 0.01, threshold, alpha)

# Compute the reprojection error residuals and the Jacobian with respect to P
def linearize_reprojection_error_P(P, Xs, xs):
    n = Xs.shape[1]
    r = compute_reprojection_residuals(P, Xs, xs)
    r = np.reshape(r, (2 * n, 1))

    J = []
    for j in range(0, n):
        X = Xs[:,j]
        m1 = 1.0 / (P[2,:] @ X)
        m2 = (P[0,:] @ X) / ((P[2,:] @ X)**2.0)
        m3 = (P[1,:] @ X) / ((P[2,:] @ X)**2.0)
        J.append([
            -X[0] * m1, -X[1] * m1, -X[2] * m1, -X[3] * m1,
            0, 0, 0, 0,
            X[0] * m2, X[1] * m2, X[2] * m2, X[3] * m2
        ])
        J.append([
            0, 0, 0, 0,
            -X[0] * m1, -X[1] * m1, -X[2] * m1, -X[3] * m1,
            X[0] * m3, X[1] * m3, X[2] * m3, X[3] * m3
        ])
    J = np.array(J)
    return (r, J)

def compute_update(r, J, mu):
    return - np.linalg.inv(J.T @ J + mu * np.identity(J.shape[1])) @ J.T @ r

ref = 0
nref = 0
def refine_P(P, Xs, xs, T=10):
    mu = 1.0
    nref = 0
    ref = 0

    prev_err = compute_reprojection_error(P, Xs, xs)
    for _ in range(0, T):
        r, J = linearize_reprojection_error_P(P, Xs, xs)
        delta_P = compute_update(r, J, mu)
        new_P = P + np.reshape(delta_P, (3, 4))
        err = compute_reprojection_error(new_P, Xs, xs)

        if err < prev_err:
            P = new_P
            prev_err = err
            mu /= 2.0
            ref += 1
        else:
            mu *= 2.0
            nref += 1

    print("Refined:", ref, ". Not refined:", nref)
    return P
