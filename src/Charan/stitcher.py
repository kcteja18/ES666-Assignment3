import pdb
import glob
import cv2
import os
import numpy as np

class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self,path):
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        ####  Your Implementation here
        #### you can use functions, class_methods, whatever!! Examples are illustrated below. Remove them and implement yours.
        #### Just make sure to return final stitched image and all Homography matrices from here
        
        
       # Read all images
        images = [cv2.imread(img_path) for img_path in all_images]

        # List of homography matrices
        homographies = []

        # Start with the first image as the reference
        stitched_image = images[0]
        for i in range(1, len(images)):
            # Detect and compute keypoints/descriptors
            kp1, des1 = self.detect_and_compute(stitched_image)
            kp2, des2 = self.detect_and_compute(images[i])

            # Match features between consecutive images
            matches = self.match_features(des1, des2)

            # Find the homography matrix
            h_matrix = self.find_homography(kp1, kp2, matches)

            if h_matrix is not None:
                # Warp the next image using the homography matrix
                homographies.append(h_matrix)
                stitched_image = self.warp_and_blend(stitched_image, images[i], h_matrix)
            else:
                print(f"Homography could not be computed between images {i-1} and {i}")
                return None, []

        return stitched_image, homographies

    def detect_and_compute(self, img):
        """ Detect keypoints and compute descriptors using SIFT """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, des1, des2):
        """ Match descriptors using FLANN based matcher """
        index_params = dict(algorithm=1, trees=5)  # Using KD-Tree
        search_params = dict(checks=50)  # or any other value
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Ratio test as per Lowe's paper
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        return good_matches

    def find_homography(self, kp1, kp2, matches):
        """ Find homography using RANSAC """
        if len(matches) > 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return H
        else:
            return None

    def warp_and_blend(self, img1, img2, H):
        """ Warp img2 to img1 using the homography matrix H, and blend them """
        # Get dimensions of both images
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Get the canvas size for the panorama (considering both images)
        pts_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        pts_img2_transformed = cv2.perspectiveTransform(pts_img2, H)

        pts_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts = np.concatenate((pts_img1, pts_img2_transformed), axis=0)

        # Get the bounding box for the panorama
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)

        # Compute the translation homography to align the images
        translation = np.array([[1, 0, -xmin],
                                [0, 1, -ymin],
                                [0, 0, 1]])

        # Warp the second image
        warped_img2 = cv2.warpPerspective(img2, translation @ H, (xmax - xmin, ymax - ymin))

        # Place the first image on the canvas
        panorama = np.zeros((ymax - ymin, xmax - xmin, 3), dtype=np.uint8)
        panorama[-ymin:h1 - ymin, -xmin:w1 - xmin] = img1

        # Blend the warped image with the first one
        mask = (warped_img2 > 0).astype(np.uint8)
        panorama = cv2.addWeighted(panorama, 1.0, warped_img2, 0.5, 0)

        return panorama