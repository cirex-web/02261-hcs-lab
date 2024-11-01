from custom_types import Image
from random import random
from skimage import feature
from skimage.feature import blob_dog, blob_log, blob_doh
import skimage
from skimage.util import view_as_windows
import numpy as np

# make sure to update get_widefield_feature_labels as well
def get_treated_features(img:Image):

  centers = [(random() * (img.image.shape[0] - 40) + 20, random() * (img.image.shape[1] - 40) + 20) for _ in range(1)]
  radius = 100
  features = [0,0,0]
  for idx, (cy, cx) in enumerate(centers):
    # Create a mask for the circle
    Y, X = np.ogrid[:img.image.shape[0], :img.image.shape[1]]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = dist_from_center <= radius

    # Extract the circular region
    sub_image = img.image * mask
    # Renormalize the grayscale image
    p_low, p_high = np.percentile(sub_image[sub_image > 0], (10, 90))
    sub_image = np.clip((sub_image - p_low) * 255.0 / (p_high - p_low), 0, 255).astype(np.uint8)

    sub_blobs = blob_doh(sub_image, max_sigma=30, threshold=0.01)
    sub_edges = feature.canny(sub_image, sigma=3)
    sub_edge_pixels = sum(map(sum, sub_edges))

    features[0] +=len(sub_blobs)
    features[1]+=sub_edge_pixels

    # sub_features = [round(sub_edge_pixels / len(sub_blobs), 1), sub_edge_pixels, len(sub_blobs)]
    # print(f"Features for circle {idx+1}: {sub_features}")
  features[2] = features[1]/features[0]
  # blurriness = skimage.measure.blur_effect(img.image, h_size=11)
  # features = [round(edge_pixels / len(blobs), 1),edge_pixels,len(blobs)]
  print(features)
  return features


def get_widefield_features(img:Image):
  # Flatten the image and sort the pixel values
  flattened_image = img.image.flatten()
  sorted_pixels = np.sort(flattened_image)

  # Calculate the number of pixels in the top 10%
  num_top_pixels = int(0.1 * len(sorted_pixels))

  # Get the top 10% pixel values
  top_10_percent_pixels = sorted_pixels[-num_top_pixels:]

  # Calculate the average intensity of the top 10% pixels
  avg_intensity_top_10_percent = np.mean(top_10_percent_pixels)

  blurriness = skimage.measure.blur_effect(img.image, h_size=11)
  features = [blurriness,avg_intensity_top_10_percent]
  print(features)
  return features

def get_widefield_feature_labels():
  return ["Blurriness","Avg. top 10% Pixel Intensity"]

def get_treated_feature_labels():
  return ["blob count","edge px count","edge count/blob count"]