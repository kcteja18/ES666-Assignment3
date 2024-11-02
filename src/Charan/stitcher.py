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
    
    def compute_homography_matrix(self, src_points, dst_points):
        """Compute the homography matrix using Direct Linear Transformation (DLT)."""
        num_points = src_points.shape[0]
        matrix_A = []

        for i in range(num_points):
            x_src, y_src = src_points[i]
            x_dst, y_dst = dst_points[i]
            matrix_A.append([-x_src, -y_src, -1, 0, 0, 0, x_src * x_dst, y_src * x_dst, x_dst])
            matrix_A.append([0, 0, 0, -x_src, -y_src, -1, x_src * y_dst, y_src * y_dst, y_dst])

        matrix_A = np.array(matrix_A)
        _, _, V_transpose = np.linalg.svd(matrix_A)
        homography_matrix = V_transpose[-1].reshape((3, 3))
        return homography_matrix / homography_matrix[2, 2] if homography_matrix[2, 2] != 0 else None


    def ransac_homography(self, src_points, dst_points, num_iterations=1000, threshold=5.0):
        """Estimate homography matrix using RANSAC."""
        optimal_homography = None
        max_inlier_count = 0
        optimal_inliers_mask = None
        src_points_homogeneous = np.hstack((src_points, np.ones((src_points.shape[0], 1))))
        np.random.seed(2)

        for _ in range(num_iterations):
            selected_indices = np.random.choice(len(dst_points), 4, replace=False)
            src_subset = src_points[selected_indices].reshape(-1, 2)
            dst_subset = dst_points[selected_indices].reshape(-1, 2)

            homography_matrix = self.compute_homography_matrix(src_subset, dst_subset)

            if homography_matrix is None:
                continue

            projected_dst_points_homogeneous = (src_points_homogeneous @ homography_matrix.T)
            projected_dst_points_homogeneous /= projected_dst_points_homogeneous[:, 2:3]  # Normalize
            reprojection_errors = np.linalg.norm(dst_points - projected_dst_points_homogeneous[:, :2], axis=1)

            inliers_mask = reprojection_errors < threshold
            inlier_count = np.sum(inliers_mask)

            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                optimal_homography = homography_matrix
                optimal_inliers_mask = inliers_mask

        return optimal_homography, optimal_inliers_mask


    def calculate_homographies(self, img_list):
        """Find pairwise homographies between consecutive images and accumulate them"""
        homographies_list = [np.eye(3)]  # First image has identity homography
        target_index = len(img_list) // 2  # The middle image is the target
        
        for i in range(len(img_list) - 1):
            keypoints1, descriptors1 = self.extract_keypoints_and_descriptors(img_list[i])
            keypoints2, descriptors2 = self.extract_keypoints_and_descriptors(img_list[i + 1])

            matches = self.bf_matcher.match(descriptors1, descriptors2)
            matches = sorted(matches, key=lambda x: x.distance)

            points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
            points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])


            homography_matrix, _ = self.ransac_homography(points1, points2)#cv2.findHomography(points1, points2, cv2.RANSAC, 9.0)
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
