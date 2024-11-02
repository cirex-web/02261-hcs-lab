from custom_types import Image
from random import random
from skimage import feature
from skimage.feature import blob_dog, blob_log, blob_doh
import skimage
from skimage.util import view_as_windows
import numpy as np

def get_line_count(img):
    edges = skimage.feature.canny(img, sigma=2,low_threshold=0)
    lines = skimage.transform.probabilistic_hough_line(edges, threshold=10, line_length=100, line_gap=12)
    # lines = skimage.transform.probabilistic_hough_line(edges, threshold=10, line_length=70, line_gap=12)

    for line in lines:
        p0, p1 = line
        # axes[1, i].plot((p0[0], p1[0]), (p0[1], p1[1]), color='red')
    return len(lines)

def get_blob_count(img):
  return len(blob_doh(img, max_sigma=30, threshold=0.01))

def get_edge_px_length(img):
  sub_edges = feature.canny(img, sigma=3)
  return sum(map(sum, sub_edges))

def get_avg_intensity_near_blobs(img):
    blobs = skimage.feature.blob_log(img, max_sigma=30,min_sigma=10, num_sigma=10)
    intensities = []
    ixs = np.indices(img.shape)
    if len(blobs) == 0:
      return 0
    for blob in blobs[:10]:
      y, x, r = blob
      blob_center = np.array([y, x])[:, np.newaxis, np.newaxis]
      mask = ((ixs - blob_center)**2).sum(axis=0) < r**2
      # rr, cc = skimage.draw.circle(y, x, r, shape=img.shape)
      # avg_intensity = np.mean(img[rr, cc])
      intensities.append(img[mask].mean())
      # print(f"Blob at ({y}, {x}) with radius {r} has average intensity {avg_intensity}")
    return sum(intensities)/len(intensities)


# make sure to update get_widefield_feature_labels as well
def get_treated_features(img:Image):
  cropped_image = img.image[:-150, :]
  # blobs = get_blob_count(Image(cropped_image))

  # edge_pixels = get_edge_px_length(Image(cropped_image))
  features = [get_line_count(cropped_image)]
  print(features)
  return features


def get_treated_feature_labels():
  return ["line count","blob avg. intensity"]
  # return ["blob count","edge px count","edge count/blob count"]

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

