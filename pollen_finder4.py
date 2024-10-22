import cv2
import numpy as np
from pathlib import Path
import os
import time

# Goes through all the found circles to see if their edges overlap
def isolate_circles(circles):
    """
    The function checks all circles' and their radii to make sure there exists
    no overlap. If there is overlap, both circles are removed from the new list

    Parameters
    ----------
    circles : list
        A list of the location and radius of all circles (x coord, y coord, r).

    Returns
    -------
    isolated_circles : list
        A list of the location of all circles in the same format, but without
        circles that overlap.

    """
    circles = np.round(circles[0, :]).astype("int")
    isolated_circles = []
    for (x1, y1, r1) in circles:
        isolated = True
        for (x2, y2, r2) in circles:
            # If the radii of 2 pollen grains together are greater than the
            # distance between pollen grains using Pythagoras' theorem, they
            # cannot are not isolated from each other.
            if (x1, y1) != (x2, y2) and np.sqrt((x1 - x2)**2 + (y1 - y2)**2) < r1 + r2:
                # Assumes pollen grains are isolated until proven wrong by
                # the if-statement.
                isolated = False
                # Breaks as soon as the algorithm figures out a pollen grain
                # is not isolated to not waste resources.
                break
        if isolated:
            isolated_circles.append((x1, y1, r1))
            
    return isolated_circles


def process_files():
    """
    This function delivers one hsv panorama with its corresponding unique
    folder + species name combination per yield.

    Yields
    ------
    folder_spec : str
        Folder and species name of any panorama image.
    hsv : NumPy Array
        Hsv representation of a panorama image.

    """
    
    directory = Path("stitched")
    # Loops through all dictionaries in stitched directory.
    for folder in directory.iterdir():
        # Only accepts standard deviation panoramas for finding pollen.
        if folder.is_dir() and folder.name[-3:] == 'std':
            for img in folder.iterdir():
                cv2img = cv2.imread(str(img))
                if cv2img is None:
                    continue
                hsv = cv2.cvtColor(cv2img, cv2.COLOR_BGR2HSV)
                folder_spec = f"{folder.name}/{img.name}"
                yield folder_spec, hsv

# Modified find_pollen to process one image at a time
def find_pollen(image, name, bounds_dic):
    """
    This function uses masking and Hough Circle Transform to find the pollen
    grains in a panorama picture.
    
    Parameters
    ----------
    image : Numpy Array
        Hsv representation of a panorama image.
    name : String
        Species name.
    bounds_dic : Dictionary
        Contains a key for every species whose value is a list with the hsv
        bounds and minimum and maximum radii.

    Returns
    -------
    circles : List
        The locations and radii of all circles (pollen) that were found in an
        image.

    """
    
    # Define names and extract parameters to be used for finding the circles.
    param = name.split("/")[0].replace(" ", "_")
    lower_bound, upper_bound, mir, mar = bounds_dic[param]
    
    # Create the mask within the HSV bounds, blue, and find the circles.
    mask = cv2.inRange(image, lower_bound, upper_bound)
    blurred = cv2.GaussianBlur(mask, (31, 31), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=50, param2=30, minRadius=mir, maxRadius=mar)
    
    
    if circles is not None:
        pass
    else:
        print(f"No circles found in {name}")
    
    return circles


def merge_stitched_pollen(image_name, species_key):
    """
    The function finds the panorama of all three projections (min, max, and
    std), turns them to greyscale and merges them into one picture.

    Parameters
    ----------
    image_name : String
        The name of the image as its found in the directory.
    species_key : String
        Name of the species.

    Returns
    -------
    merge : NumPy Array
        The merged image after putting the greyscale image in the seperate
        color channels.
    """
    # Generalized format panorama loctions for all species.
    folder_paths = {
        "max": f"stitched/{species_key}_max/{image_name}",
        "min": f"stitched/{species_key}_min/{image_name}",
        "std": f"stitched/{species_key}_std/{image_name}",
    }

    # Read the three panoramas (max, min, std)
    maximum = cv2.imread(folder_paths["max"], cv2.IMREAD_GRAYSCALE)
    minimum = cv2.imread(folder_paths["min"], cv2.IMREAD_GRAYSCALE)
    std = cv2.imread(folder_paths["std"], cv2.IMREAD_GRAYSCALE)

    if maximum is None or minimum is None or std is None:
        print(f"Error reading image for {species_key} - {image_name}")
        return None

    # Create a 3-channel RGB image and the merge image by putting the images
    # into the seperate chaannels.
    merge = np.zeros((std.shape[0], std.shape[1], 3), dtype=np.uint8)
    merge[:, :, 0] = maximum  # R channel
    merge[:, :, 1] = minimum  # G channel
    merge[:, :, 2] = std      # B channel

    return merge


def save_images(circles, image, key, res):
    """
    This function filters out all pollen that should not be saved and saves
    the rest that can be used for the dataset.

    Parameters
    ----------
    circles : List
        All found circles by the Hough Circle Transform algorithm.
    image : Numpy Array
        Contains the image, represented by the array.
    key : String
        An identifier to distinguish between panorama images.
    res : Int
        The resolution for each individual square pollen image.

    Returns
    -------
    None.

    """
    species = "_".join(key.split("_")[:2]).replace(" ", "_")
    # Manipulates the list to only include pollen that do not include other
    # pollen.
    isolated_circles = isolate_circles(circles)
    # Res needs to be halved, because cutout size it's measured in difference
    # from the center in all directions.
    res = int(res / 2)
    
    # This loop defines the four points that creaate a square around a pollen
    # grain to save the pollen grain as a seperate image in resolution "res".
    for i, circle in enumerate(isolated_circles):
        x_1, x_2 = circle[1] - res, circle[1] + res
        y_1, y_2 = circle[0] - res, circle[0] + res
        # If part of the pollen image is not in the frame after creating the
        # square for slicing, the instance will not be saved.
        if min(x_1, y_1) < 0 or x_2 > image.shape[0] or y_2 > image.shape[1]:
            pass  # Out of bounds
        else:
            print(f"saving to nn_input/{species}/{key}_{i}.png")
            img_to_save = image[x_1:x_2, y_1:y_2]
            cv2.imwrite(f"nn_input/{species}/{key}_{i}.png", img_to_save)

def main():
    """
    Main function to call all other functions and record the time it takes.

    Returns
    -------
    None.

    """
    bounds_dic = {"B_pendula_std": [(68, 0, 72), (255, 255, 255), 120, 150],
                  "C_nootkatensis_std": [(123, 0, 0), (255, 255, 255), 140, 180],
                  "T_baccata_std": [(0, 0, 75), (255, 140, 150), 110, 165],
                  "C_japonica_std": [(80, 0, 0), (255, 140, 255), 150, 200],
                  "C_lawsoniana_std": [(95, 68, 0), (255, 140, 255), 150, 220],
                  "T_distichum_std": [(74, 51, 77), (255, 255, 255), 110, 180]}
    
    # Size of the square for each image
    resolution = 440
    
    print("Searching for pollen on stitched images one by one")
    counter = 0
    os.makedirs("pf4_times/", exist_ok=True)
    start_time = time.time()
    
    # process_files is used in this way (yield) to iterate through the images
    # in a manner with regards to memory efficiency by not loading all images
    # into memory.
    for name, hsv_image in process_files():
        print(f"Processing {name}")
        circles = find_pollen(hsv_image, name, bounds_dic)
        part1 = name.split("/")[1]
        species = "_".join(name.split("_")[:2])
        merge_img = merge_stitched_pollen(part1, species)
        os.makedirs(f"nn_input/{species}", exist_ok=True)
        save_images(circles, merge_img, f"{species}_{counter}", resolution)
        counter += 1
        end_time = time.time()
        title = name.split("/")[0]
        with open(f"pf4_times/{title}.txt", "a") as file:
            file.write(f"{str(end_time - start_time)}\n")
        start_time = time.time()
    
    print("All pollen saved")

main()
