import warnings

import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
from operator import itemgetter
from utils import Ransac_DLT_homography, apply_H_fixed_image_size, plot_img


def plot_contour(H, logo_img, dst_img, offset_x=0):
    """
    Draws a contour around the detected logo in the destination image using homography

    Parameters:
    -----------
    H : ndarray
        3x3 homography matrix
    logo_img : ndarray
        Source/logo image
    dst_img : ndarray
        Destination image where contour will be drawn
    offset_x : int, optional
        Horizontal offset for drawing

    Returns:
    --------
    ndarray
        Copy of destination image with contour drawn
    """
    # Create a copy of the destination image
    result_img = dst_img.copy()

    # Get image dimensions
    h, w = logo_img.shape[:2]
    h = h - 1; w = w - 1

    # Define corners in homogeneous coordinates
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1]
    ], dtype=np.float32)

    # Transform corners using homography
    transformed_corners = []
    for corner in corners:
        # Apply homography
        transformed = H @ corner
        # Convert back from homogeneous coordinates
        transformed = transformed / transformed[2]
        transformed_corners.append(transformed[:2])

    # Convert to integer coordinates for drawing
    transformed_corners = np.array(transformed_corners, dtype=np.int32)

    # Add offset if specified
    if offset_x != 0:
        transformed_corners[:, 0] += offset_x

    # Draw the contour lines
    for i in range(4):
        pt1 = tuple(transformed_corners[i])
        pt2 = tuple(transformed_corners[(i + 1) % 4])
        cv2.line(result_img, pt1, pt2, (0, 255, 0), 2)

    return result_img


def detect_logo(logo_img, dst_img, verbose=False, threshold=0.7, ransac_threshold=3, ransac_iterations=1000):
    """
    Detects the logo in the destination image using SIFT descriptors and RANSAC
    :param logo_img: Logo image
    :param dst_img: Destination image
    :param verbose: True to show the intermediate steps
    :param threshold: Threshold for the ratio test
    :param ransac_threshold: RANSAC threshold
    :param ransac_iterations: Number of RANSAC iterations
    :return:
    """
    assert logo_img.ndim == 2, 'The logo image must be in grayscale'
    assert dst_img.ndim == 2, 'The destination image must be in grayscale'
    assert 0 < threshold < 1, 'The threshold must be between 0 and 1'
    assert ransac_threshold > 0 and ransac_iterations > 0, 'The RANSAC threshold and iterations must be greater than 0'
    if verbose:
        print("Detecting the logo in the destination image...")
        print("-" * 50)

    # Instantiate a SIFT descriptor (we are using SIFT instead of ORB)
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(logo_img, None)
    kp2, des2 = sift.detectAndCompute(dst_img, None)
    if verbose:
        print('Number of keypoints in logo:', len(kp1))
        print('Number of keypoints in destination image:', len(kp2))

    # Match descriptors using FLANN matcher (FLANN matcher is way faster than BFMatcher)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test to find the best matches
    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append([m])

    if verbose:
        print('Number of matches:', len(matches))
        print('Number of good matches:', len(good_matches))
        # Show "good" matches
        img = cv2.drawMatchesKnn(logo_img, kp1, dst_img, kp2, good_matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img)
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.show()

    if len(good_matches) >= 4:
        # Get the points of the good matches
        points1 = []
        points2 = []
        for m in good_matches:
            points1.append([kp1[m[0].queryIdx].pt[0], kp1[m[0].queryIdx].pt[1], 1])
            points2.append([kp2[m[0].trainIdx].pt[0], kp2[m[0].trainIdx].pt[1], 1])

        points1 = np.asarray(points1)
        points1 = points1.T
        points2 = np.asarray(points2)
        points2 = points2.T

        H, indices_inlier_matches = Ransac_DLT_homography(points1, points2, ransac_threshold, ransac_iterations)
        inlier_matches = itemgetter(*indices_inlier_matches)(good_matches)

        if verbose:
            img_12 = cv2.drawMatchesKnn(logo_img, kp1, dst_img, kp2, inlier_matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            img_12 = plot_contour(H, logo_img, img_12, offset_x=logo_img.shape[1])

            plt.imshow(img_12)
            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(18.5, 10.5)
            plt.show()

        return H, inlier_matches, points1, points2
    else:
        warnings.warn('Not enough good matches to estimate the homography')
        return None


if __name__ == '__main__':
    # Read the images in grayscale
    logo_img = cv2.imread('Data/logos/logoUPF.png')  # Notice IMREAD_GRAYSCALE is platform dependent
    logo_img_gray = cv2.cvtColor(logo_img, cv2.COLOR_BGR2GRAY)

    blg_img = cv2.imread('Data/logos/UPFbuilding.jpg')
    blg_img_gray = cv2.cvtColor(blg_img, cv2.COLOR_BGR2GRAY)

    # Step 1: Detect the logo in the building image
    H, inliers_matches, points1, points2 = detect_logo(logo_img_gray, blg_img_gray, verbose=True)

    # Step 2: Transform the corners using the homography
    logo_img = cv2.imread('Data/logos/logo_master.png')  # Notice IMREAD_GRAYSCALE is platform dependent
    logo_img = cv2.resize(logo_img, (123, 122))

    corners = np.array([0, blg_img.shape[1] - 1, 0, blg_img.shape[0] - 1], dtype=np.float32).T
    imgA_w = apply_H_fixed_image_size(logo_img, H, corners)
    imgB_w = apply_H_fixed_image_size(blg_img, np.eye(3), corners)

    result_img = np.where(imgA_w > 0, imgA_w, imgB_w)
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    plot_img(result_img)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.show()




