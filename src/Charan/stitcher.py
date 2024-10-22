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

    def match_features(self, img1_features, img2_features):
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        best_matches = bf.match(img1_features,img2_features)

        matches = sorted(best_matches, key = lambda x:x.distance)

        return matches

    def find_homography(self, kps1, kps2, matches):
        """ Find homography using RANSAC """
        kps1 = np.float32([keypoint.pt for keypoint in kps1])
        kps2 = np.float32([keypoint.pt for keypoint in kps2])
        if len(matches) > 4:
            src_pts = np.float32([kps1[m.queryIdx] for m in matches])
            dst_pts = np.float32([kps2[m.trainIdx] for m in matches])

            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
            return H
        else:
            return None

    def warp_and_blend(self, img1, img2, H):
        """ Warp img2 to img1 using the homography matrix H, and blend them """
        width = img2.shape[1] + img1.shape[1]
        # print("width ", width) 

        height = max(img2.shape[0], img1.shape[0])

        panorama = cv2.warpPerspective(img1, H,  (width, height))

        panorama[0:img2.shape[0], 0:img2.shape[1]] = img2

        return panorama