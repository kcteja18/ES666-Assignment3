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
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        best_matches = bf.knnMatch(img1_features,img2_features,k=2)

        # matches = sorted(best_matches, key = lambda x:x.distance)
        matches = []

        # loop over the raw matches
        for m,n in best_matches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if m.distance < n.distance * 0.75:
                matches.append(m)

        return matches

    def find_homography(self, kps1, kps2, matches):
        """ Find homography using RANSAC """
        kps1 = np.float32([keypoint.pt for keypoint in kps1])
        kps2 = np.float32([keypoint.pt for keypoint in kps2])
        if len(matches) > 4:
            src_pts = np.float32([kps1[m.queryIdx] for m in matches])
            dst_pts = np.float32([kps2[m.trainIdx] for m in matches])

            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4)
            if H is not None:
                H = H.astype(np.float32)
                print("Homography Matrix:\n", H)  # Debug print
                print("Homography Shape:", H.shape)  # Debug print
            else:
                print("Homography could not be computed.")
            return H
        else:
            return None

    def warp_and_blend(self, img1, img2, H):
        """ Warp img2 to img1 using the homography matrix H, and blend them """
        # width = img2.shape[1] + img1.shape[1]
        # # print("width ", width) 

        # height = max(img2.shape[0], img1.shape[0])

        # panorama = cv2.warpPerspective(img1, H,  (width, height))

        # panorama[0:img2.shape[0], 0:img2.shape[1]] = img2

        # return panorama
        # Get the size of both images
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]

        # Determine the corners of img2
        corners_img2 = np.array([[0, 0], [width2, 0], [0, height2], [width2, height2]], dtype='float32')
        
        # Apply the homography to the corners of img2
        warped_corners = cv2.perspectiveTransform(corners_img2.reshape(-1, 1, 2), H)

        # Find the bounds of the panorama
        all_corners = np.concatenate((corners_img2, warped_corners.reshape(-1, 2)), axis=0)
        [x_min, y_min] = np.int32(all_corners.min(axis=0))
        [x_max, y_max] = np.int32(all_corners.max(axis=0))

        # Translation matrix to shift the image to the positive quadrant
        translate_dist = [-x_min, -y_min]
        translation_matrix = np.array([[1, 0, translate_dist[0]], 
                                    [0, 1, translate_dist[1]], 
                                    [0, 0, 1]])

        # Warp img1 and img2 using the homography and translation matrix
        panorama1 = cv2.warpPerspective(img1, translation_matrix @ np.eye(3), (x_max - x_min, y_max - y_min))
        panorama2 = cv2.warpPerspective(img2, translation_matrix @ H, (x_max - x_min, y_max - y_min))

        # Create a mask to indicate valid pixels (non-zero) in panorama2
        mask1 = np.zeros_like(panorama1, dtype=np.uint8)
        mask1[panorama1 > 0] = 1
        
        mask2 = np.zeros_like(panorama2, dtype=np.uint8)
        mask2[panorama2 > 0] = 1

        # Find the overlapping region (where both masks are non-zero)
        overlap_mask = cv2.bitwise_and(mask1, mask2)

        # Feathering to blend overlapping regions
        non_overlap = cv2.bitwise_and(panorama1, cv2.bitwise_not(overlap_mask)) + cv2.bitwise_and(panorama2, cv2.bitwise_not(overlap_mask))

        # Use feathering on overlapping regions
        feathered_blend = cv2.addWeighted(panorama1, 0.5, panorama2, 0.5, 0)

        # Combine the feathered overlap with the non-overlapping regions
        blended = cv2.add(non_overlap, cv2.bitwise_and(feathered_blend, overlap_mask))

        return blended