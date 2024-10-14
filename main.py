import sys
import numpy as np
import matplotlib.pyplot as plt
from get_dataset_info import get_dataset_info
from misc import pflat, triangulate_DLT, compute_reprojection_errors, plotcams
from sift import sift_matches, sift_image_matches, sift_keypoints
from epipolar import estimate_E_robust, extract_P2_from_E
from camera import estimate_t_robust, refine_P

def compute_relative_orientations(kps, des, img_names, threshold):
    relative_rotations = []
    for i in range(0, len(img_names) - 1):
        m1, m2 = sift_image_matches(des, i, i + 1, siftcache)
        x1s, x2s = kps[i][:,m1], kps[i + 1][:,m2]

        E, inliers = estimate_E_robust(x1s, x2s, threshold)
        x1s, x2s = x1s[:,inliers], x2s[:,inliers]
        P = extract_P2_from_E(E, x1s, x2s)
        R = P[:3,:3]
        relative_rotations.append(R)

    return relative_rotations

def compute_rotation_matrices(kps, des, img_names, threshold):
    # Calculate relative orientations between sequential image pairs
    relative_rotations = compute_relative_orientations(kps, des, img_names, threshold)

    # Calculate absolute rotation matrices
    Rs = [np.identity(3)]
    for i in range(0, len(relative_rotations)):
        Rs.append(relative_rotations[i] @ Rs[i])

    return Rs

def construct_3D_points_from_image_pair(kps, des, img1, img2, threshold):
    # Get image point correspondences
    matches1, matches2 = sift_image_matches(des, img1, img2, siftcache)
    x1s, x2s = kps[img1][:,matches1], kps[img2][:,matches2]

    # Compute 3d points
    E, inliers = estimate_E_robust(x1s, x2s, threshold)
    x1s, x2s = x1s[:,inliers], x2s[:,inliers]
    matches1, matches2 = matches1[inliers], matches2[inliers]
    P1 = np.column_stack([np.identity(3), np.zeros((3,1))])
    P2 = extract_P2_from_E(E, x1s, x2s)
    Xs = pflat(triangulate_DLT(P1, P2, x1s, x2s))

    return (Xs, matches1, matches2)

def compute_cameras(kps, des, Rs, img_names, Xs_init, img_init, matches_init, threshold):
    Ps = []
    for i in range(0, len(img_names)):
        # Establish correspondences
        matchname = 'matches_' + str(i) + "_3D"
        matches1, matches2 = sift_matches(des[i], des[img_init][matches_init,:], matchname, siftcache)
        x1s = kps[i][:,matches1]
        Xs = Xs_init[:, matches2]

        # Estimate and refine T
        t, inliers = estimate_t_robust(Rs[i], Xs, x1s, threshold*3.0)
        P = np.column_stack([Rs[i], t])
        P = refine_P(P, Xs[:,inliers], x1s[:,inliers], T=10)

        Ps.append(P)

    return Ps

def construct_3D_points(kps, des, Ps, img1, img2, threshold):
    matches1, matches2 = sift_image_matches(des, img1, img2, siftcache)
    x1s, x2s = kps[img1][:,matches1], kps[img2][:,matches2]

    Xs = pflat(triangulate_DLT(Ps[img1], Ps[img2], x1s, x2s))
    c = np.sum(Xs[:3,:], axis=1) / Xs.shape[1]
    dists = np.linalg.norm(Xs[:3,:] - c[:,np.newaxis], axis=0)
    dist_threshold = 2.0 * np.percentile(dists, 90.0)
    errs = 0.5 * (compute_reprojection_errors(Ps[img1], Xs, x1s) + compute_reprojection_errors(Ps[img2], Xs, x2s))
    inliers = list(filter(
        lambda i: errs[i] < threshold**2.0 and dists[i] < dist_threshold,
        range(0, Xs.shape[1])))

    return Xs[:,inliers]

def main():
    np.random.seed(0)

    # Load dataset
    K, img_names, init_pair, pixel_threshold = get_dataset_info(dataset)
    threshold = pixel_threshold / K[0, 0]
    init_pair = [init_pair[0] - 1, init_pair[1] - 1]

    # Compute SIFT keypoints
    kps, des = sift_keypoints(img_names, K, siftcache)

    # Absolute rotations matrices
    Rs = compute_rotation_matrices(kps, des, img_names, threshold)

    # Construct initial 3D points from image pair
    Xs_init, matches1_init, _ = construct_3D_points_from_image_pair(kps, des, init_pair[0], init_pair[1], threshold)
    n_init = Xs_init.shape[1]

    # Convert initial 3D points to word coordinates
    c = np.sum(Xs_init[:3,:], axis=1) / n_init
    Xs_init[:3,:] = Rs[init_pair[0]].T @ (Xs_init[:3,:] - c[:,np.newaxis])

    # Calculate cameras
    Ps = compute_cameras(kps, des, Rs, img_names, Xs_init, init_pair[0], matches1_init, threshold)

    # Triangulate points
    Xss = []
    for i in range(0, len(img_names) - 1):
        Xss.append(construct_3D_points(kps, des, Ps, i, i + 1, threshold))

    # Plot
    plt.figure()
    ax = plt.gcf().add_subplot(projection='3d')
    for i, Xs in enumerate(Xss):
        ax.scatter(Xs[0,:], Xs[2,:], -Xs[1,:], s=1)
    plotcams(np.array(Ps), l=0.5)
    ax.axis('equal')

    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Needs exactly one argument: <dataset id>")
        exit()
    dataset = int(sys.argv[1])
    dataset_path = 'data/' + str(dataset) + '/'
    siftcache = None
    main()
