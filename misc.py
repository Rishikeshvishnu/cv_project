import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Takes an nxm matrix where each column is a homogenous vector in P^(n-1).
# Normalizes the vectors so that the last row = [1,1,...]
def pflat(xs):
    return xs / xs[-1, :]

# From two camera matrices P1 and P2, and n point correspondences x1s, x2s, compute
# the number of these points that lie in front of both cameras when triangulated
# into 3D.
def num_in_front(P1, P2, x1s, x2s):
    assert(np.abs(np.linalg.det(P2[:3,:3]) - 1.0) < 0.001)

    X = pflat(triangulate_DLT(P1, P2, x1s, x2s))
    points1 = P1 @ X
    points2 = P2 @ X

    n_infront = 0
    for i in range(0, np.shape(points1)[1]):
        if points1[2,i] > 0 and points2[2,i] > 0:
            n_infront += 1
    
    return n_infront

# Triangulate a set of point correspondences into 3D points using direct linear
# transform.
#
# Takes two 3x4 camera matrices (P1, P2), and two 3xn sets of image points
# (x1s, x2s) such that x1s[i] and x2s[i] correspond to the ith point projected
# to camera 1 and 2 respectively.
def triangulate_DLT(P1, P2, x1s, x2s):
    nPoints = np.shape(x1s)[1]
    X = []
    for i in range(0,nPoints):
        M = np.array([
            x1s[0,i] * P1[2,:] - x1s[2,i] * P1[0,:],
            x1s[1,i] * P1[2,:] - x1s[2,i] * P1[1,:],
            x2s[0,i] * P2[2,:] - x2s[2,i] * P2[0,:],
            x2s[1,i] * P2[2,:] - x2s[2,i] * P2[1,:],
        ])
        (_,_,VT) = np.linalg.svd(M)
        V = np.transpose(VT)
        X.append(V[:,-1])

    return np.transpose(np.array(X))

def plotcams(P, l=4):
    c = np.zeros((4, len(P)))
    v = np.zeros((3, len(P)))

    for i in range(0, len(P)):
        c[:,i] = sp.linalg.null_space(P[i]).flatten()
        v[:,i] = P[i][2, 0:3].flatten()

    c = c / np.tile(c[3,:], (4, 1))

    plt.quiver(c[0,:], c[2,:], -c[1,:], v[0,:], v[2,:], -v[1,:], length=l, color='r')

# Compute something using the MSAC algorithm.
#   estimate_fn takes no arguments and returns an estimate for the solution
#   err_fn takes the estimate and computes the squared errors
#   n is the number of points in our model
#   n_samples is the number of samples used when estimating a solution
#   d is the number of points to use for the T(d,d) test
#   eps is the chance that the estimate is correct
#   threshold is the highest allowed error to be considered an inlier
#   alpha is the desired probability of computing a correct solution
# Returns the solution with the highest number of inliers, and the
# corresponding inliers
def msac(estimate_fn, err_fn, n, n_samples, d, eps, threshold, alpha, extra_cond = None):
    def compute_T(eps, alpha):
        return int(np.ceil(np.log(1-alpha) / np.log(1-eps**n_samples)))

    T = compute_T(eps, alpha)

    threshold2 = threshold**2
    (res, res_inliers, res_cost) = ((), [], np.inf)
    iterations = 0
    while iterations < T:
        # Estimate model candidate
        rand_indices = np.random.randint(0, n, (n_samples,))
        res_estimate = estimate_fn(rand_indices)

        iterations += 1
        for r_est in res_estimate:
            # Early bailout T(d,d) test
            rand_indices_d = np.random.randint(0, n, (d,))
            err_d = err_fn(r_est, rand_indices_d)
            if not all(map(lambda i: err_d[i] < threshold**2, range(d))):
                continue

            # Compute cost
            err = err_fn(r_est, slice(n))
            cost = sum(map(lambda e2: min(e2, threshold2), err))

            # Compute inliers
            inliers = list(filter(
                lambda i: err[i] < threshold2,
                range(0, n)))

            if extra_cond is not None:
                if extra_cond(r_est, inliers) == False:
                    continue

            # Update solution
            if cost < res_cost:
                if len(inliers) / n > eps:
                    eps = len(inliers) / n
                    T = compute_T(eps, alpha)
                    print("eps updated to", eps, ". T updated to", T)

                res, res_inliers, res_cost = r_est, inliers, cost

    return (res, res_inliers)

# For a camera P (3x4), n 3D points Xs (4xn), n image points xs (3xn), compute
# the residuals r (nx2) for the reprojection error.
def compute_reprojection_residuals(P, Xs, xs):
    r = []
    n = Xs.shape[1]
    for j in range(0, n):
        r.append([
            xs[0,j] - (P[0,:] @ Xs[:,j]) / (P[2,:] @ Xs[:,j]),
            xs[1,j] - (P[1,:] @ Xs[:,j]) / (P[2,:] @ Xs[:,j])
        ])
    return np.array(r)

# Compute the squared reprojection error for each point (nx1).
def compute_reprojection_errors(P, Xs, xs):
    r = compute_reprojection_residuals(P, Xs, xs)
    return np.sum(r**2, axis=1)

# Compute the total squared reprojection error (1x1)
def compute_reprojection_error(P, Xs, xs):
    return np.sum(compute_reprojection_errors(P, Xs, xs))

