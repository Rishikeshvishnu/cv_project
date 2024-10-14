import numpy as np
import cv2 as cv

# Compute the normalized image keypoints and descriptors from an image, optionally
# using a cache file
def sift_keypoints(imgs, K, siftcache=None):
    Kinv = np.linalg.inv(K)

    (res_kps, res_des) = ([], [])
    for imgname in imgs:
        if siftcache != None and imgname in siftcache:
            cachedval = siftcache[imgname]
            kps, des = (cachedval['kps'][0][0], cachedval['des'][0][0])
        else:
            img = cv.imread(imgname, cv.IMREAD_GRAYSCALE)

            sift = cv.SIFT_create()
            kps, des = sift.detectAndCompute(img, None)
            n = len(kps)

            kps = list(map(lambda kp: kp.pt, kps))
            kps = Kinv @ np.row_stack([np.array(kps).T, np.ones((1,n))])
            des = np.array(des)

            if siftcache != None:
                siftcache[imgname] = {'kps': kps, 'des': des}

        res_kps.append(kps)
        res_des.append(des)

    return (res_kps, res_des)

# Compute matches between two sets of descriptors, optionally
# using a cache file
def sift_matches(desc1, desc2, matchname, siftcache=None):
    if siftcache != None and matchname in siftcache:
        return siftcache[matchname]

    # Match points
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    # Find good matches
    good1 = []
    good2 = []
    for m,n in matches:
        if m.distance < n.distance * 0.75:
            good1.append(m.queryIdx)
            good2.append(m.trainIdx)
    good1, good2 = np.array(good1), np.array(good2)

    if siftcache != None:
        siftcache[matchname] = (good1, good2)

    return (good1, good2)

# Compute matches between two images, optionally
# using a cache file
def sift_image_matches(desc, img1, img2, siftcache=None):
    matchname = 'matches_' + str(img1) + '_' + str(img2)
    return sift_matches(desc[img1], desc[img2], matchname, siftcache)
