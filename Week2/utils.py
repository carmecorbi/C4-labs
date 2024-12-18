import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import random
import plotly.graph_objects as go
import math
import sys

def plot_img(img, do_not_use=[0]):
    plt.figure(do_not_use[0])
    do_not_use[0] += 1
    plt.imshow(img)


def get_transformed_pixels_coords(I, H, shift=None):
    ys, xs = np.indices(I.shape[:2]).astype("float64")
    if shift is not None:
        ys += shift[1]
        xs += shift[0]
    ones = np.ones(I.shape[:2])
    coords = np.stack((xs, ys, ones), axis=2)
    coords_H = (H @ coords.reshape(-1, 3).T).T
    coords_H /= coords_H[:, 2, np.newaxis]
    cart_H = coords_H[:, :2]
    
    return cart_H.reshape((*I.shape[:2], 2))

def apply_H_fixed_image_size(I, H, corners):
    h, w = I.shape[:2] # when we convert to np.array it swaps
    
    # corners
    c1 = np.array([1, 1, 1])
    c2 = np.array([w, 1, 1])
    c3 = np.array([1, h, 1])
    c4 = np.array([w, h, 1])
    
    # transformed corners
    Hc1 = H @ c1
    Hc2 = H @ c2
    Hc3 = H @ c3
    Hc4 = H @ c4
    # Normalize them 
    Hc1 = Hc1 / Hc1[2]
    Hc2 = Hc2 / Hc2[2]
    Hc3 = Hc3 / Hc3[2]
    Hc4 = Hc4 / Hc4[2]
    
    xmin = corners[0]
    xmax = corners[1]
    ymin = corners[2]
    ymax = corners[3]

    size_x = ceil(xmax - xmin + 1)
    size_y = ceil(ymax - ymin + 1)
    
    # transform image
    H_inv = np.linalg.inv(H)
    
    out = np.zeros((size_y, size_x, 3))
    shift = (xmin, ymin)
    interpolation_coords = get_transformed_pixels_coords(out, H_inv, shift=shift)
    interpolation_coords[:, :, [0, 1]] = interpolation_coords[:, :, [1, 0]]
    interpolation_coords = np.swapaxes(np.swapaxes(interpolation_coords, 0, 2), 1, 2)
    
    out[:, :, 0] = map_coordinates(I[:, :, 0], interpolation_coords)
    out[:, :, 1] = map_coordinates(I[:, :, 1], interpolation_coords)
    out[:, :, 2] = map_coordinates(I[:, :, 2], interpolation_coords)
    
    return out.astype("uint8")

def Normalise_last_coord(x):
    xn = x  / x[2,:]
    
    return xn

def DLT_homography(points1, points2):
    
    # ToDo: complete this code .......

    # Normalize points in both images 
    
    # 1. Convert to homogeneous coordinates (normalization --> from (x,y,w) to (x/w, y/w, 1))
    points1_n = Normalise_last_coord(points1)
    points2_n = Normalise_last_coord(points2)

    # 2. Normalize them by transforming them so that their mean is 0 (origin)
    # and the avg. distance from the origin is sqrt(2) to improve stability
    #   - T1 and T2 are the transformation matrices defined to normalize poinst1
    #     and points2, respectively
    m1,s1 = np.mean(points1_n,1), np.std(points1_n)
    s1 = np.sqrt(2)/s1
    T1 = np.array([[s1, 0, -s1*m1[0]], [0, s1, -s1*m1[1]], [0, 0, 1]])

    m2,s2 = np.mean(points2_n,1), np.std(points2_n)
    s2 = np.sqrt(2)/s2
    T2 = np.array([[s2, 0, -s2*m2[0]], [0, s2, -s2*m2[1]], [0, 0, 1]])

    # Apply the normalization transform to the given sets of points
    points1n = T1 @ points1_n
    points2n = T2 @ points2_n


    # Note: For each point correspondence ((x,y,z) in the 1st image and (u,v,w) in the 2nd image, 
    #       where z and w are 1 after the normalization step), 2 linear equations are defined:
    #
    #           1. (h_11·x + h_12·y + h_13·1) - u(h_31·x + h_32·y + h_33·1) = 0  
    #           2. (h_21·x + h_22·y + h_23·1) - v(h_31·x + h_32·y + h_33·1) = 0  
    #
    # Then we can define the system of equations in a matricial form: Ah = 0
    A = []
    n = points1.shape[1]

    for i in range(n):
        # Get the normalized points
        x,y,z = points1n[0,i], points1n[1,i], points1n[2,i]
        u,v,w = points2n[0,i], points2n[1,i], points2n[2,i]
        # From the 1st equation (w=1 and z=1)
        A.append([w*x, w*y, w*z, 0, 0, 0, -u*x, -u*y, -u*z])
        # From the 2nd equation
        A.append([0, 0, 0, -w*x, -w*y, -w*z, v*x, v*y, v*z])
        
      
    # Note: Solve Ah=0 using the SVD, so that we obtain A = U·D·V^T and the
    #       solution h will be the last column of V (eigenvector associated to the 
    #       smallest eigenvalue)

    # Convert A to array and perform the SVD
    A = np.asarray(A)
    U, D, Vt = np.linalg.svd(A)

    # Extract homography (last row of Vt --> normalize it so that h_33 = 1)
    h = Vt[-1, :] / Vt[-1, -1]
    # Reshape to get homography matrix
    H = h.reshape(3, 3) 
    
    # Denormalize to get the transfomration matrix on the original coord. system
    H = np.linalg.inv(T2) @ H @ T1

    return H


def Inliers(H, points1, points2, th):
    
    # Check that H is invertible
    if abs(math.log(np.linalg.cond(H))) > 15:
        idx = np.empty(1)
        return idx
    
    # ToDo: complete this code .......
    # Given:
    #   - (x,y,z): point in the 1st image (points1)
    #   - (u,v,w): point in the 2nd image (points2)
    #   - H: homography matrix such that (x',y',z')^T = H·(x, y, z)^T and (u', v', w')^T = H^(-1)·(u, v, w)^T 

    # Compute transformed (normalized: 3rd coordinate equal to 1) points 
    #   - Transform points1 to the 2nd image space
    points1_transformed = Normalise_last_coord(H @ points1)
    #   - Transform points2 to the 1st image space
    points2_transfomed = Normalise_last_coord(np.linalg.inv(H) @ points2)
    # Normalize given points
    points1 = Normalise_last_coord(points1)
    points2 = Normalise_last_coord(points2)
    
    # Compute the reprojection error between all correspondences:
    #   - Euclidean distance between tranformed points and their correspondences 
    #     (combination of forward (points2-points1_transf) and backward 
    #     (points1-points2_transf) reprojection error)
    error = np.sqrt((np.sum((points2[:-1] - points1_transformed[:-1])**2, axis=0) + np.sum((points2_transfomed[:-1] - points1[:-1])**2, axis=0)))
    # Get as inliers the ones for which the reprojection error is under a threshold
    inliers_mask = error < th
    inliers_indices = np.where(inliers_mask)[0]
    
    return inliers_indices


def Ransac_DLT_homography(points1, points2, th, max_it):
    
    Ncoords, Npts = points1.shape
    
    it = 0
    best_inliers = np.empty(1)
    
    while it < max_it:
        # Take 4 point correspondences between the 2 images (randomly chosen)
        indices = random.sample(range(Npts), 4)
        # Estimate the homography matrix H using the DLT algorithm
        H = DLT_homography(points1[:,indices], points2[:,indices])
        # Get the indices of the points that are considered inliers
        inliers = Inliers(H, points1, points2, th)
        
        # test if it is the best model so far based on the number of inliers
        if inliers.shape[0] > best_inliers.shape[0]:
            best_inliers = inliers
        
        # update estimate of iterations (the number of trials) to ensure we pick, with probability p,
        # an initial data set with no outliers
        fracinliers = inliers.shape[0]/Npts
        pNoOutliers = 1 -  fracinliers**4
        eps = sys.float_info.epsilon
        pNoOutliers = max(eps, pNoOutliers)   # avoid division by -Inf
        pNoOutliers = min(1-eps, pNoOutliers) # avoid division by 0
        p = 0.99
        max_it = math.log(1-p)/math.log(pNoOutliers)
        
        it += 1
    
    # compute H from all the inliers
    H = DLT_homography(points1[:,best_inliers], points2[:,best_inliers])
    inliers = best_inliers
    
    return H, inliers



def optical_center(P):
    U, d, Vt = np.linalg.svd(P)
    o = Vt[-1, :3] / Vt[-1, -1]
    return o

def view_direction(P, x):
    # Vector pointing to the viewing direction of a pixel
    # We solve x = P v with v(3) = 0
    v = np.linalg.inv(P[:,:3]) @ np.array([x[0], x[1], 1])
    return v

def plot_camera(P, w, h, fig, legend):
    
    o = optical_center(P)
    scale = 200
    p1 = o + view_direction(P, [0, 0]) * scale
    p2 = o + view_direction(P, [w, 0]) * scale
    p3 = o + view_direction(P, [w, h]) * scale
    p4 = o + view_direction(P, [0, h]) * scale
    
    x = np.array([p1[0], p2[0], o[0], p3[0], p2[0], p3[0], p4[0], p1[0], o[0], p4[0], o[0], (p1[0]+p2[0])/2])
    y = np.array([p1[1], p2[1], o[1], p3[1], p2[1], p3[1], p4[1], p1[1], o[1], p4[1], o[1], (p1[1]+p2[1])/2])
    z = np.array([p1[2], p2[2], o[2], p3[2], p2[2], p3[2], p4[2], p1[2], o[2], p4[2], o[2], (p1[2]+p2[2])/2])
    
    fig.add_trace(go.Scatter3d(x=x, y=z, z=-y, mode='lines',name=legend))
    
    return

def plot_image_origin(w, h, fig, legend):
    p1 = np.array([0, 0, 0])
    p2 = np.array([w, 0, 0])
    p3 = np.array([w, h, 0])
    p4 = np.array([0, h, 0])
    
    x = np.array([p1[0], p2[0], p3[0], p4[0], p1[0]])
    y = np.array([p1[1], p2[1], p3[1], p4[1], p1[1]])
    z = np.array([p1[2], p2[2], p3[2], p4[2], p1[2]])
    
    fig.add_trace(go.Scatter3d(x=x, y=z, z=-y, mode='lines',name=legend))
    
    return
