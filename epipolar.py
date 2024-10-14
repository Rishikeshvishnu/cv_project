import numpy as np
import scipy as sp
import cv2 as cv
from multiprocessing.pool import ThreadPool
from camera import find_best_camera, num_in_front
from misc import msac

# Enforces the condition that the singular values of an essential
# matrix should be [1, 1, 0]
def enforce_essential(E):
    (U,_,VT) = np.linalg.svd(E)
    if np.linalg.det(U @ VT) < 0:
        VT = -VT
    return U @ np.diag([1,1,0]) @ VT

# Estimates the essential matrix from two 3xm matrices where each column is
# an image point in homogeneous coordinates such that x1s[i] and x2s[i] are
# the calibrated coordinates of point i in image 1 and 2 respectively.
def estimate_E_8_point(x1s, x2s):
    assert(np.shape(x1s) == np.shape(x2s))
    nPoints = np.shape(x1s)[1]

    M = np.empty((nPoints, 9))
    for i in range(0,nPoints):
        M[i,:] = [
            x2s[0,i] * x1s[0,i], x2s[0,i] * x1s[1,i], x2s[0,i] * x1s[2,i],
            x2s[1,i] * x1s[0,i], x2s[1,i] * x1s[1,i], x2s[1,i] * x1s[2,i],
            x2s[2,i] * x1s[0,i], x2s[2,i] * x1s[1,i], x2s[2,i] * x1s[2,i]
        ]

    (_,_,VT) = np.linalg.svd(M)
    V = VT.T
    v = V[:,-1]
    
    E = enforce_essential(np.reshape(v, (3,3)))
    return E

def estimate_E_homography(x1s, x2s):
    assert(x1s.shape == x2s.shape)

    nPoints = np.shape(x1s)[1]
    M = np.empty((2 * nPoints, 9))
    for i in range(0,nPoints):
        M[2*i + 0,:] = [
            0, 0, 0,
            -x2s[2,i] * x1s[0,i], -x2s[2,i] * x1s[1,i], -x2s[2,i] * x1s[2,i],
            x2s[1,i] * x1s[0,i], x2s[1,i] * x1s[1,i], x2s[1,i] * x1s[2,i]
        ]
        M[2*i + 1,:] = [
            x2s[2,i] * x1s[0,i], x2s[2,i] * x1s[1,i], x2s[2,i] * x1s[2,i], 
            0, 0, 0,
            -x2s[0,i] * x1s[0,i], -x2s[0,i] * x1s[1,i], -x2s[0,i] * x1s[2,i]
        ]

    (_,_,VT) = np.linalg.svd(M)
    V = VT.T
    v = V[:,-1]
    
    H = np.reshape(v, (3,3))

    I = np.eye(3)
    _, Rs, Ts, _ = cv.decomposeHomographyMat(H, I)
    Es = list(map(lambda i: np.cross(I, Ts[i].T) @ Rs[i], range(4)))
    return Es

def one_at(i, s):
    v = np.zeros(s)
    v[i] = 1.0
    return v

def estimate_E_5_point(x1s, x2s):
    assert(x1s.shape == (3,5) and x2s.shape == (3,5))

    # Use 5 point correspondences to solve the constraint
    # x2^T E x1 = 0
    M = np.empty((5, 9))
    for i in range(5):
        M[i,:] = [
            x2s[0,i] * x1s[0,i], x2s[0,i] * x1s[1,i], x2s[0,i] * x1s[2,i],
            x2s[1,i] * x1s[0,i], x2s[1,i] * x1s[1,i], x2s[1,i] * x1s[2,i],
            x2s[2,i] * x1s[0,i], x2s[2,i] * x1s[1,i], x2s[2,i] * x1s[2,i]
        ]
    ns = sp.linalg.null_space(M)
    Es = list(map(lambda i: ns[:,i].reshape((3,3)), range(4)))

    # Maps (i,j,k) to monomial index of alpha_i * alpha_j * alpha_k. Where
    # alpha_1 -> x, alpha_2 -> y, alpha_3 -> z, alpha_4 -> 1 and the monomials are indexed as such:
    #
    # Monomial: x^3 x^2y x^2z xy^2 xyz xz^2 y^3 y^2z yz^2 z^3 x^2 xy xz y^2 yz z^2 x  y  z  1
    # Index:     0   1    2    3    4   5    6   7    8    9   10 11 12 13  14  15 16 17 18 19
    monomial_index = np.array([
        [[0, 1, 2, 10],
         [1, 3, 4, 11],
         [2, 4, 5, 12],
         [10, 11, 12, 16]],
        [[1, 3, 4, 11],
         [3, 6, 7, 13],
         [4, 7, 8, 14],
         [11, 13, 14, 17]],
        [[2, 4, 5, 12],
         [4, 7, 8, 14],
         [5, 8, 9, 15],
         [12, 14, 15, 18]],
        [[10, 11, 12, 16],
         [11, 13, 14, 17],
         [12, 14, 15, 18],
         [16, 17, 18, 19]]
    ])

    # Set up the constraints 2E^TE - trace(EE^T)E = 0 and det(E) = 0 so that
    # C [x^3 x^2y ...]^T = 0
    C = np.empty((10,20))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                m_i = monomial_index[i,j,k]
                C[:9, m_i] = \
                    (2.0 * Es[i] @ Es[j].T @ Es[k] -
                     np.matrix.trace(Es[i] @ Es[j].T) * Es[k])\
                    .reshape((9,))
                C[9, m_i] = \
                    Es[i][0,0] * Es[j][1,1] * Es[k][2,2] + \
                    Es[i][0,1] * Es[j][1,2] * Es[k][2,0] + \
                    Es[i][0,2] * Es[j][1,0] * Es[k][2,1] - \
                    Es[i][0,0] * Es[j][1,2] * Es[k][2,1] - \
                    Es[i][0,1] * Es[j][1,0] * Es[k][2,2] - \
                    Es[i][0,2] * Es[j][1,1] * Es[k][2,0]

    # Re-write C in reduced row echelon.
    C = np.linalg.inv(C[:10,:10]) @ C

    # C = [I | C']. C [x^3 x^2y ... x^2 xy ..]^T = 0
    # So C' = -C[:,10:20] gives C[0,:] [x^2 xy ..]^T = X^3, C[1,:] [x^2 xy ..]^T = x^2y etc.
    C = - C[:,10:20]

    # Construct the action matrix Mx such that
    # x [x^2 xy ..]^T = Mx [x^2 xy ..]
    Mx = np.array([
        C[0,:], C[1,:], C[2,:], C[3,:], C[4,:], C[5,:],
        one_at(0,10), one_at(1,10), one_at(2,10), one_at(6, 10)
    ])
    _, vs = np.linalg.eig(Mx)

    # Construct the essential matrices
    E_res = []
    for i in range(10):
        x, y, z, w = vs.real[6,i], vs.real[7,i], vs.real[8,i], vs.real[9,i]
        x = x / w
        y = y / w
        z = z / w
        w = 1.0
        E = Es[0] * x + Es[1] * y + Es[2] * z + Es[3] * w

        E = enforce_essential(E)
        E_res.append(E)

    # Return all essential matrices to be evaluated for inliers
    return E_res

# Extract all the 4 possible P2 camera matrices from an essential matrix, where
# P1 = (I, 0)
def extract_P2s_from_E(E):
    (U,_,VT) = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    if np.linalg.det(U @ W @ VT) < 0.0:
        VT = -VT
    return [
        np.column_stack([U @ W @ VT, U[:,2]]),
        np.column_stack([U @ W @ VT, -U[:,2]]),
        np.column_stack([U @ W.T @ VT, U[:,2]]),
        np.column_stack([U @ W.T @ VT, -U[:,2]]),
    ]

# From an essential matrix and n point correspondences, compute the second camera matrix
# relative to P1 = (I,0).
def extract_P2_from_E(E, x1s, x2s):
    P2s = extract_P2s_from_E(E)
    return find_best_camera(P2s, x1s, x2s)

# Compute the squared distance between the image points and their
# corresponding epipolar lines.
def compute_epipolar_errors(F, x1s, x2s):
    l = F @ x1s
    l = l / np.sqrt(np.tile(l[0,:]**2 + l[1,:]**2, (3, 1)))
    errors = abs(np.sum(l * x2s, axis=0))
    return errors

def chierality_condition(E, x1s, x2s, threshold = 0.05):
    P2s = extract_P2s_from_E(E)

    n = x1s.shape[1]
    P1 = np.column_stack([np.identity(3), np.zeros((3,1))])
    in_front = list(map(lambda P2: num_in_front(P1, P2, x1s, x2s), P2s))

    n_zero = 0
    n_full = 0
    for P2 in P2s:
        in_front = num_in_front(P1, P2, x1s, x2s)
        if in_front <= threshold * n:
            n_zero += 1
        if in_front >= (1.0 - threshold) * n:
            n_full += 1

    return n_zero == 3 and n_full == 1

# Estimate the essential matrix E from two lists (3xn) of n calibrated
# image points from two images using MSAC. Also get the number of inliers when
# using this essential matrix. The inliers are the points where their
# epipolar error < threshold pixels.
def estimate_E_robust(x1s, x2s, threshold, alpha=0.9, use_5_point=False):
    def estimate_fn_5_pt(rand_indices):
        return estimate_E_5_point(x1s[:, rand_indices], x2s[:, rand_indices])

    def estimate_fn_8_pt(rand_indices):
        return [estimate_E_8_point(x1s[:, rand_indices], x2s[:, rand_indices])]

    def estimate_fn_H(rand_indices):
        return estimate_E_homography(x1s[:, rand_indices], x2s[:, rand_indices])

    def err_fn(E, indices):
        return (1/2) * (
            compute_epipolar_errors(E, x1s[:,indices], x2s[:,indices])**2 +
            compute_epipolar_errors(E.T, x2s[:,indices], x1s[:,indices])**2
        )

    def extra_cond_fn(E, inliers):
        return chierality_condition(E, x1s[:,inliers], x2s[:,inliers])
    
    n = x1s.shape[1]
    if use_5_point:
        print("Running MSAC for estimating E using 5 point method, n=",n)
        return msac(estimate_fn_5_pt, err_fn, n, 5, 1, 0.01, threshold * 2.0, alpha, extra_cond_fn)
    else:
        pool = ThreadPool(processes=2)

        print("Running MSAC for estimating E using 8 point method, n=",n)
        print("Running MSAC for estimating E using homography estimation, n=",n)
        res1 = pool.apply_async(msac, (
            estimate_fn_8_pt, err_fn, n, 8, 1, 0.01, threshold, alpha, extra_cond_fn
        ))
        res2 = pool.apply_async(msac, (
            estimate_fn_H, err_fn, n, 4, 1, 0.01, threshold, alpha, extra_cond_fn
        ))
        E1, inl1 = res1.get()
        E2, inl2 = res2.get()
        p_inl1, p_inl2 = len(inl1) / n, len(inl2) / n
        if len(inl1) > len(inl2):
            print("8 pt was best (", p_inl1, "vs", p_inl2, " inliers)")
            return E1, inl1
        else:
            print("H was best (", p_inl2, "vs", p_inl1, " inliers)")
            return E2, inl2
