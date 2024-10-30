import pdb
import glob
import cv2
import os
import numpy as np

class PanaromaStitcher():
        
    def __init__(self):
        self.sift_detector = cv2.SIFT_create()
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def extract_keypoints_and_descriptors(self, img):
        """Detect keypoints and compute descriptors using SIFT"""
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift_detector.detectAndCompute(gray_img, None)
        return keypoints, descriptors
    
    # def normalize_points(points):
    #     """Normalize points to have zero mean and average distance to origin sqrt(2)."""
    #     centroid = np.mean(points, axis=0)
    #     centered_points = points - centroid
    #     avg_dist = np.mean(np.linalg.norm(centered_points, axis=1))
    #     scale = np.sqrt(2) / avg_dist
    #     T = np.array([[scale, 0, -scale * centroid[0]],
    #                 [0, scale, -scale * centroid[1]],
    #                 [0, 0, 1]])
    #     normalized_points = np.dot(T, np.hstack((points, np.ones((points.shape[0], 1)))).T).T
    #     return normalized_points[:, :2], T

    def compute_homography(self,points_src, points_dst):
        """Compute the homography matrix from four point correspondences using DLT."""
        assert points_src.shape == points_dst.shape, "Source and destination points must have the same shape."
        num_points = points_src.shape[0]
        
        A = []
        for i in range(num_points):
            x_src, y_src = points_src[i]
            x_dst, y_dst = points_dst[i]
            A.append([-x_src, -y_src, -1, 0, 0, 0, x_dst * x_src, x_dst * y_src, x_dst])
            A.append([0, 0, 0, -x_src, -y_src, -1, y_dst * x_src, y_dst * y_src, y_dst])

        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        homography_matrix = Vt[-1].reshape(3, 3)

        return homography_matrix / (homography_matrix[2, 2] if homography_matrix[2, 2]!=0 else 1)

    def apply_homography(self,H, points):
        """Apply homography matrix H to points."""
        points_h = np.hstack((points, np.ones((points.shape[0], 1))))
        transformed_points_h = np.dot(H, points_h.T).T
        transformed_points = transformed_points_h[:, :2] / transformed_points_h[:, 2].reshape(-1, 1)
        return transformed_points

    def ransac_homography(self,points_src, points_dst, num_iterations=1000, threshold=5.0):
        """RANSAC to compute the best homography matrix."""
        max_inliers = 0
        best_H = None
        num_points = np.hstack((points_src, np.ones((points_src.shape[0], 1))))
        np.random.seed(32)

        for _ in range(num_iterations):
            # Randomly select 4 point correspondences
            random_indices = np.random.choice(num_points, 4, replace=False)
            src_sample = points_src[random_indices]
            dst_sample = points_dst[random_indices]
            try:
                H = self.compute_homography(src_sample, dst_sample)
            except np.linalg.LinAlgError:
                continue

            projected_dst = (num_points @ H.T)
            projected_dst /= projected_dst[:, 2:3]
            distances = np.linalg.norm(points_dst - projected_dst[:,:2], axis=1)
            inliers = distances < threshold
            num_inliers = np.sum(inliers)
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_H = H
                
        if best_H is not None:
            inlier_src = points_src[inliers]
            inlier_dst = points_dst[inliers]
            best_H = self.compute_homography(inlier_src, inlier_dst)

        return best_H, inliers

    def calculate_homographies(self, img_list):
        """Find pairwise homographies between consecutive images and accumulate them"""
        homographies_list = [np.eye(3)]  # First image has identity homography
        target_index = len(img_list)//2  # The middle image is the target

        for i in range(len(img_list) - 1):
            keypoints1, descriptors1 = self.extract_keypoints_and_descriptors(img_list[i])
            keypoints2, descriptors2 = self.extract_keypoints_and_descriptors(img_list[i + 1])

            if descriptors1 is None or descriptors2 is None:
                print(f"Insufficient keypoints detected in images {i} or {i + 1}. Skipping pair.")
                continue

            matches = self.bf_matcher.match(descriptors1, descriptors2)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) < 4:
                print(f"Not enough matches between images {i} and {i + 1}. Skipping pair.")
                continue

            points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
            points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

            homography_matrix, _ = self.ransac_homography(points1, points2)
            if homography_matrix is not None:
                homographies_list.append(homography_matrix)
            else:
                print(f"Could not compute homography between images {i} and {i + 1}. Skipping pair.")
                homographies_list.append(np.eye(3))  # Add identity as a fallback

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
    
    def warp_perspective(self, img, homography_matrix, out_shape):
        """Warp the image using the homography matrix and the specified output shape"""
        height_out, width_out = out_shape
        warped_img = np.zeros((height_out, width_out, img.shape[2]), dtype=img.dtype)
        homography_inv = np.linalg.inv(homography_matrix)
        x_dest, y_dest = np.meshgrid(np.arange(width_out), np.arange(height_out))
        ones_dest = np.ones_like(x_dest)
        dest_coords = np.stack((x_dest, y_dest, ones_dest), axis=-1).reshape(-1, 3)
        source_coords = (homography_inv @ dest_coords.T).T
        source_coords /= source_coords[:, 2].reshape(-1, 1)  

        src_x = source_coords[:, 0]
        src_y = source_coords[:, 1]
        valid_mask = (0 <= src_x) & (src_x < img.shape[1]) & (0 <= src_y) & (src_y < img.shape[0])

        src_x_clipped = np.clip(src_x, 0, img.shape[1] - 1)
        src_y_clipped = np.clip(src_y, 0, img.shape[0] - 1)

        x_floor = np.floor(src_x_clipped).astype(np.int32)
        x_ceil = np.clip(x_floor + 1, 0, img.shape[1] - 1)
        y_floor = np.floor(src_y_clipped).astype(np.int32)
        y_ceil = np.clip(y_floor + 1, 0, img.shape[0] - 1)

        dx = src_x_clipped - x_floor
        dy = src_y_clipped - y_floor

        # Perform bilinear interpolation for each channel
        for channel in range(img.shape[2]):
            top_left = img[y_floor, x_floor, channel] * (1 - dx) + img[y_floor, x_ceil, channel] * dx
            bottom_left = img[y_ceil, x_floor, channel] * (1 - dx) + img[y_ceil, x_ceil, channel] * dx
            warped_pixel_values = top_left * (1 - dy) + bottom_left * dy
            warped_img[..., channel].flat[valid_mask] = warped_pixel_values[valid_mask]

        return warped_img
    
    def make_panaroma_for_images_in(self, path):
        """Read images from the path, use precomputed homographies, and create a panorama"""
    
        all_images = sorted(glob.glob(path + os.sep + '*'))
        print(f'Found {len(all_images)} Images for stitching')
        
        images = [cv2.imread(img_path) for img_path in all_images]
        if not images or len(images) < 2:
            print("Not enough images to stitch.")
            return None, []

        total_homographies = self.calculate_homographies(images)
        
        # Ensure total_width and total_height are correct for the panorama canvas
        total_width = max([img.shape[1] for img in images])  # Maximum width of images
        total_height = sum([img.shape[0] for img in images])  # Sum of all heights

        # Instead of hardcoded translation matrix, calculate a proper one for the canvas size
        translation_matrix = np.array([[1, 0, total_width // 2], [0, 1, total_height // 2], [0, 0, 1]], dtype=np.float32)

        # Create an empty canvas for the stitched panorama
        stitched_img = np.zeros((total_height, total_width, 3), dtype=np.uint8)

        # Iterate over homographies and images
        for idx, homography in enumerate(total_homographies):
            translated_homography = np.dot(translation_matrix, homography)
            warped_img = self.warp_perspective(images[idx], translated_homography, (total_width, total_height))

            # Use np.maximum to blend the images, ensuring no shape mismatch
            if stitched_img.shape == warped_img.shape:
                stitched_img = np.maximum(stitched_img, warped_img)
            else:
                print(f"Dimension mismatch for stitched_img and warped_img at index {idx}")

        return stitched_img, total_homographies
