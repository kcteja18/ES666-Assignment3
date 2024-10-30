import glob
import cv2
import os
import numpy as np

class PanaromaStitcher():
        
    def __init__(self):
        self.sift_detector = cv2.SIFT_create()
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
    def make_panaroma_for_images_in(self, path):
        """Read images from the path, use precomputed homographies, and create a panorama"""
    
        all_images = sorted(glob.glob(path + os.sep + '*'))
        print(f'Found {len(all_images)} Images for stitching')
         
        images = [cv2.imread(img_path) for img_path in all_images]
        if not images or len(images) < 2:
            print("Not enough images to stitch.")
            return None, []

        total_homographies = self.calculate_homographies(images)
        
        total_width = sum([img.shape[1] for img in images])
        total_height = 2 * max([img.shape[0] for img in images])
        translation_matrix = np.array([[1, 0, total_width // 4], [0, 1, total_height // 4], [0, 0, 1]], dtype=np.float32)
        stitched_img = np.zeros((total_height, total_width, 3), dtype=np.uint8)
        
        # Apply homographies and stitch images onto the panorama
        for idx, homography in enumerate(total_homographies):
            translated_homography = np.dot(translation_matrix, homography)
            warped_img = cv2.warpPerspective(images[idx], translated_homography, (total_width, total_height))
            stitched_img = np.maximum(stitched_img, warped_img)

        return stitched_img, total_homographies

    def extract_keypoints_and_descriptors(self, img):
        """Detect keypoints and compute descriptors using SIFT"""
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift_detector.detectAndCompute(gray_img, None)
        return keypoints, descriptors

    def calculate_homographies(self, img_list):
        """Find pairwise homographies between consecutive images and accumulate them"""
        homographies_list = [np.eye(3)]  # First image has identity homography
        target_index = len(img_list) // 2  # The middle image is the target
        
        for i in range(len(img_list) - 1):
            keypoints1, descriptors1 = self.extract_keypoints_and_descriptors(img_list[i])
            keypoints2, descriptors2 = self.extract_keypoints_and_descriptors(img_list[i + 1])

            matches = self.bf_matcher.match(descriptors1, descriptors2)
            matches = sorted(matches, key=lambda x: x.distance)

            points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            homography_matrix, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 9.0)
            homographies_list.append(homography_matrix / (homography_matrix[-1, -1] if homography_matrix[-1, -1] != 0 else 1))
        
        return self.accumulate_homographies(homographies_list, target_index)

    def accumulate_homographies(self, homographies_list, target_index):
        """Compute total homographies relative to a target image"""
        accumulated_homographies = []
        identity_matrix = np.eye(3)

        for idx, homography in enumerate(homographies_list):
            if idx < target_index:
                accumulated_homography = np.eye(3)
                for j in range(idx, target_index):
                    accumulated_homography = np.dot(homographies_list[j + 1], accumulated_homography)
                accumulated_homographies.append(accumulated_homography)
            elif idx > target_index:
                accumulated_homography = np.eye(3)
                for j in range(idx, target_index, -1):
                    accumulated_homography = np.dot(np.linalg.inv(homographies_list[j]), accumulated_homography)
                accumulated_homographies.append(accumulated_homography)
            else:
                accumulated_homographies.append(identity_matrix)

        return accumulated_homographies
