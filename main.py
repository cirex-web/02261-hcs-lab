import os
import pandas as pd
import skimage
import roc
# Task - 1 Treated - Untreated feature discrimination
# Shortlist: Color Channel,
import matplotlib.pyplot as plt
import numpy as np

from custom_types import Image
from features import get_treated_features, get_widefield_features, get_widefield_feature_labels, get_treated_feature_labels



def read_image_with_cv(file_path):
    file_path = os.path.abspath(file_path)
    image = skimage.io.imread(file_path,as_gray=True)
    p2, p98 = np.percentile(image, (1, 99))
    image = skimage.exposure.rescale_intensity(image, in_range=(p2, p98))
    
    # image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    # image = cv2.threshold(image, find_threshold(image), 255, cv2.THRESH_BINARY)
    # fig, ax = try_all_threshold(image, figsize=(10, 8), verbo/se=False)
    # plt.show()
    if image is None:
        raise Exception
    return image


def read_tsv_file(file_path):
    def normalize_data(data):
        if data["Channel"] == "DIC":
            data["Channel"] = "TRANS"
        return data

    df = pd.read_csv(file_path, sep="\t")
    result = {
        row.iloc[0]: normalize_data(row.iloc[1:].to_dict()) for _, row in df.iterrows()
    }
    return result
    

def get_files(img_directory: str):
    try:
        files = os.listdir(img_directory)
        return [file for file in files]

    except FileNotFoundError:
        print(f"The directory {img_directory} does not exist.")


def get_properties_of_automated_file(file_name: str):
    channel = int(file_name[-5])
    return Image(
        wide_field= channel == 0 or channel == 1,
        channel="DAPI" if channel == 0 or channel == 2 else "GDP",
        treated=None,
        image=read_image_with_cv("HCS Images/Automated HCS Images/"+file_name),
        file_name=file_name
    )


def classify_treated_or_untreated(treated_images:list[Image], untreated_images:list[Image]):
    image_set = treated_images+untreated_images
    feature_set = [get_treated_features(img) for img in image_set]
    labels = [1 for treated_image in treated_images] + [0 for untreated_image in untreated_images]
    feature_labels = get_treated_feature_labels()
    false_positive_indices, false_negative_indices = roc.analyze_performance(feature_set, labels, feature_labels)
    true_positive_indices = set(range(0, len(treated_images))) - false_negative_indices
    true_negative_indices = set(range(len(treated_images), len(image_set))) - false_positive_indices
    fig, axes = plt.subplots(4, 2, figsize=(10, 10))
    print(f"True Positive Indices: {true_positive_indices}")
    print(f"True Negative Indices: {true_negative_indices}")
    print(f"False Positive Indices: {false_positive_indices}")
    print(f"False Negative Indices: {false_negative_indices}")
    for i, idx in enumerate(true_positive_indices):
        if i >= 2:
            break
        axes[0, i].imshow(image_set[idx].image, cmap='gray')
        axes[0, i].set_title("True Treated", fontsize=8)
        axes[0, i].set_xlabel(image_set[idx].file_name, fontsize=6)
        axes[0, i].axis('off')

    for i, idx in enumerate(true_negative_indices):
        if i >= 2:
            break
        axes[1, i].imshow(image_set[idx].image, cmap='gray')
        axes[1, i].set_title("True Untreated", fontsize=8)
        axes[1, i].set_xlabel(image_set[idx].file_name, fontsize=6)
        axes[1, i].axis('off')

    for i, idx in enumerate(false_positive_indices):
        if i >= 2:
            break
        axes[2, i].imshow(image_set[idx].image, cmap='gray')
        axes[2, i].set_title("False Treated", fontsize=8)
        axes[2, i].set_xlabel(image_set[idx].file_name, fontsize=6)
        axes[2, i].axis('off')

    for i, idx in enumerate(false_negative_indices):
        if i >= 2:
            break
        axes[3, i].imshow(image_set[idx].image, cmap='gray')
        axes[3, i].set_title("False Untreated", fontsize=8)
        axes[3, i].set_xlabel(image_set[idx].file_name, fontsize=6)
        axes[3, i].axis('off')

    plt.tight_layout()
    plt.show()

def classify_widefield_vs_confocal(confocal_images: list[Image], widefield_images: list[Image]):
    feature_labels = get_widefield_feature_labels()
    image_set = confocal_images + widefield_images
    feature_set = [get_widefield_features(confocal_image) for confocal_image in image_set]
    labels = [1 for confocal_image in confocal_images] + [0 for widefield_image in widefield_images] # Positive being confocal, Negative being widefield
    false_positive_indices, false_negative_indices = roc.analyze_performance(feature_set, labels, feature_labels)
    true_positive_indices = set(range(0,len(confocal_images))) - false_negative_indices
    true_negative_indices = set(range(len(confocal_images),len(image_set))) - false_positive_indices
    fig, axes = plt.subplots(4, 2, figsize=(10, 10))
    print(f"True Positive Indices: {true_positive_indices}")
    print(f"True Negative Indices: {true_negative_indices}")
    print(f"False Positive Indices: {false_positive_indices}")
    print(f"False Negative Indices: {false_negative_indices}")
    for i, idx in enumerate(true_positive_indices):
        if i >= 2:
            break
        axes[0, i].imshow(image_set[idx].image, cmap='gray')
        axes[0, i].set_title("True Confocal", fontsize=8)
        axes[0, i].set_xlabel(image_set[idx].file_name, fontsize=6)
        axes[0, i].axis('off')

    for i, idx in enumerate(true_negative_indices):
        if i >= 2:
            break
        axes[1, i].imshow(image_set[idx].image, cmap='gray')
        axes[1, i].set_title("True Widefield", fontsize=8)
        axes[1, i].set_xlabel(image_set[idx].file_name, fontsize=6)
        axes[1, i].axis('off')

    for i, idx in enumerate(false_positive_indices):
        if i >= 2:
            break
        axes[2, i].imshow(image_set[idx].image, cmap='gray')
        axes[2, i].set_title("False Confocal", fontsize=8)
        axes[2, i].set_xlabel(image_set[idx].file_name, fontsize=6)
        axes[2, i].axis('off')

    for i, idx in enumerate(false_negative_indices):
        if i >= 2:
            break
        axes[3, i].imshow(image_set[idx].image, cmap='gray')
        axes[3, i].set_title("False Widefield", fontsize=8)
        axes[3, i].set_xlabel(image_set[idx].file_name, fontsize=6)
        axes[3, i].axis('off')

    plt.tight_layout()
    plt.show()



def load_in_all_imgs():

    img_data_dict = read_tsv_file(
        "Fall 2024 - 02-261 Shared Data Table - Lab 7 Images.tsv"
    )
    manual_file_names = [
        file_name for file_name in get_files("HCS Images/Manual HCS Images")
    ]
    manual_imgs:list[Image] = [
        Image(
            wide_field=False,
            channel=v["Channel"],
            treated=v["Treated/Untreated"] == "Treated",
            image=read_image_with_cv("HCS Images/Manual HCS Images/"+file_name),
            file_name=file_name,
        )
        for file_name, v in img_data_dict.items()
        if file_name in manual_file_names
    ]
    automated_imgs = [] 
    # or [
    #     get_properties_of_automated_file(file_name)
    #     for file_name in get_files("HCS Images/Automated HCS Images/")
    # ]
    

    # print(automated_data_dict, len(automated_data_dict))


    return manual_imgs,automated_imgs



manual_imgs, auto_imgs = load_in_all_imgs()
treated_images = [img for img in manual_imgs if img.treated and img.channel == "GFP"]
untreated_images = [img for img in manual_imgs if not img.treated and img.channel == "GFP"]
# [img for img in manual_imgs if img.channel == "GFP" or img.channel=="DAPI"]+
confocal_images =  [img for img in auto_imgs if not img.wide_field]
widefield_images = [img for img in auto_imgs if img.wide_field]



# max_images = 10
# confocal_images = confocal_images[:max_images]
# widefield_images = widefield_images[:max_images]
# fig, axes = plt.subplots(2, max_images, figsize=(20, 4))

# for i, img in enumerate(confocal_images):
#     axes[0, i].imshow(img.image, cmap='gray')
#     axes[0, i].set_title("confocal")
#     axes[0, i].axis('off')

# for i, img in enumerate(widefield_images):
#     axes[1, i].imshow(img.image, cmap='gray')
#     axes[1, i].set_title("f")
#     axes[1, i].axis('off')

# plt.tight_layout()
# plt.show(block=False)

def get_lines(image):

    edges = skimage.feature.canny(image[:-200,:], sigma=5)
    lines = skimage.transform.probabilistic_hough_line(edges, threshold=10, line_length=100, line_gap=5)
    return lines
def show_treated_and_untreated_imgs(treated_images,untreated_images):
    max_images = 10
    treated_images = treated_images[:max_images]
    untreated_images = untreated_images[:max_images]
    fig, axes = plt.subplots(2, max_images, figsize=(20, 4))
    print(len(treated_images),len(untreated_images))
    for i, img in enumerate(treated_images):
        # Perform blob detection
        # blobs = skimage.feature.blob_log(img.image, max_sigma=30,min_sigma=10, num_sigma=10, threshold=0.2)
        
        # # Overlay circles on the original image
        # for blob in blobs:
        #     y, x, r = blob
        #     c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        #     axes[0, i].add_patch(c)

        # Using hough line transform
        # tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        # h, theta, d = skimage.transform.hough_line(img.image, theta=tested_angles)
        # for _, angle, dist in zip(*skimage.transform.hough_line_peaks(h, theta, d)):
        #     # y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        #     # y1 = (dist - img.image.shape[1] * np.cos(angle)) / np.sin(angle)
        #     # axes[0, i].plot((0, img.image.shape[1]), (y0, y1), '-r')
        #     (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        #     axes[0,i].axline((x0, y0), slope=np.tan(angle + np.pi / 2))

        # using canny edge detection
        for line in get_lines(img.image):
            p0, p1 = line
            axes[0, i].plot((p0[0], p1[0]), (p0[1], p1[1]), color='red')
        axes[0, i].imshow(img.image, cmap='gray')
        # axes[0, i].set_title(get_treated_features(img))
        axes[0, i].axis('off')

    for i, img in enumerate(untreated_images):
        # blobs = skimage.feature.blob_log(img.image, max_sigma=30,min_sigma=10, num_sigma=10, threshold=0.2)
        
        # # Overlay circles on the original image
        # for blob in blobs:
        #     y, x, r = blob
        #     c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        #     axes[1, i].add_patch(c)

        for line in get_lines(img.image):
            p0, p1 = line
            axes[1, i].plot((p0[0], p1[0]), (p0[1], p1[1]), color='red')
        axes[1, i].imshow(img.image, cmap='gray')

        # axes[1, i].set_title(get_treated_features(img))
        axes[1, i].axis('off')

        plt.tight_layout()
    plt.show(block=True)

# Main code is here
# classify_widefield_vs_confocal(confocal_images,widefield_images)
show_treated_and_untreated_imgs(treated_images,untreated_images)
classify_treated_or_untreated(treated_images,untreated_images)
