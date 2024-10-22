import cv2
import numpy as np
from pathlib import Path

# Used as global variable to continue updating the image after adjusting
# parameters.
fp_image = 0

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
    # Ensure only isolated circles are considered
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


def display_image(image, name="test", circles=[]):
    """
    Displays an image using cv2.

    Parameters
    ----------
    image : Numpy Array
        Matrix representation of an image to show the end-user.
    name : String, optional
        Name of the window that displays the image. The default is "test".
    circles : List, optional
        Location and size of all found circles. The default is [].

    Returns
    -------
    None.

    """
    # Draws all circles on the image.
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (255, 0, 0), 12)
    cv2.namedWindow(f"{name} Detected Circles", cv2.WINDOW_NORMAL)
    cv2.imshow(f"{name} Detected Circles", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_files():
    """
    This function collects all images in the stitched folder per species and
    saves it in a dictionary.

    Returns
    -------
    pollen_dict : Dictionary
        Contains the hsv image of all std images in the stitched directory.

    """
    pollen_dict = dict()
    directory = Path("stitched")
    # Loops through all dictionaries in stitched directory.
    for folder in directory.iterdir():
        # Only accepts standard deviation panoramas for finding pollen.
        if folder.is_dir() and folder.name[-3:] == 'std':
            for img in folder.iterdir():
                cv2img = cv2.imread(str(img))
                hsv = cv2.cvtColor(cv2img, cv2.COLOR_BGR2HSV)
                pollen_dict[f"{folder.name}/{img.name}"] = hsv

    return pollen_dict


def update_image_hsv(*args):
    """
    Used to continually update the image after changing the parameters
    regarding to the masking bounds and viewport location.

    Parameters
    ----------
    *args : tuple
        OpenCV trackbars automatically pass the current trackbar value to the
        callback function. The *args parameter allows the function to accept
        this without needing to use them.

    Returns
    -------
    None.

    """
    global fp_image
    # Get current trackbar values
    h_min = cv2.getTrackbarPos('H Min', 'HSV Segmentation')
    h_max = cv2.getTrackbarPos('H Max', 'HSV Segmentation')
    s_min = cv2.getTrackbarPos('S Min', 'HSV Segmentation')
    s_max = cv2.getTrackbarPos('S Max', 'HSV Segmentation')
    v_min = cv2.getTrackbarPos('V Min', 'HSV Segmentation')
    v_max = cv2.getTrackbarPos('V Max', 'HSV Segmentation')
    x_pos = cv2.getTrackbarPos('X Position', 'HSV Segmentation')
    y_pos = cv2.getTrackbarPos('Y Position', 'HSV Segmentation')

    # Create a mask based on HSV bounds
    mask = cv2.inRange(fp_image, (h_min, s_min, v_min), (h_max, s_max, v_max))

    # Apply the mask to the original image
    result = cv2.bitwise_and(fp_image, fp_image, mask=mask)

    # Detect circles using Hough Circle Transform
    blurred = cv2.GaussianBlur(mask, (31, 31), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=50, param2=30, minRadius=110, maxRadius=165)
    blur2color = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    # If circles are found, overlay them on the image
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # print(r)
            cv2.circle(blur2color, (x, y), r, (0, 255, 255), 2)
    
    height, width, _ = result.shape
    # Assumes a monitor of resolution 1920x1080.
    viewport_width = min(1920, width)
    viewport_height = min(900, height)

    # Calculate the viewport region
    x_start = min(max(x_pos, 0), width - viewport_width)
    y_start = min(max(y_pos, 0), height - viewport_height)
    viewport = blur2color[y_start:y_start+viewport_height, x_start:x_start+viewport_width]
    
    # You see what the computer is seeing to detect circles + found circles.
    cv2.imshow('HSV Segmentation', viewport)


def find_pollen(pollen_dict, bounds_dic):
    """
    This function loops through all images located in "stitched/{species} std"
    to display the panorama one by one and allowing the end-user to adjust
    values for better recognition.

    Parameters
    ----------
    pollen_dict : dictionary
        Contains key-value pairs for every location - image.
    bounds_dic : dictionary
        Contains key-value paris for the default hsv bounds and min/max radius
        per species.

    Returns
    -------
    None.

    """
    global fp_image
    
    # Assign global fp_image variable to an image and loop through images.
    for name, fp_image in pollen_dict.items():
        print(name)
        lower_bound, upper_bound, minrad, maxrad = (bounds_dic[name.split("/")[0]])

        cv2.namedWindow("HSV Segmentation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("HSV Segmentation", 1920, 900)
        
        # Create all trackbars to adjust parameters for the window.
        cv2.createTrackbar('H Min', 'HSV Segmentation', 0, 255, update_image_hsv)
        cv2.createTrackbar('H Max', 'HSV Segmentation', 0, 255, update_image_hsv)
        cv2.createTrackbar('S Min', 'HSV Segmentation', 0, 255, update_image_hsv)
        cv2.createTrackbar('S Max', 'HSV Segmentation', 0, 255, update_image_hsv)
        cv2.createTrackbar('V Min', 'HSV Segmentation', 0, 255, update_image_hsv)
        cv2.createTrackbar('V Max', 'HSV Segmentation', 0, 255, update_image_hsv)
        cv2.setTrackbarPos('H Min', 'HSV Segmentation', 70)
        cv2.setTrackbarPos('H Max', 'HSV Segmentation', 255)
        cv2.setTrackbarPos('S Min', 'HSV Segmentation', 0)
        cv2.setTrackbarPos('S Max', 'HSV Segmentation', 255)
        cv2.setTrackbarPos('V Min', 'HSV Segmentation', 0)
        cv2.setTrackbarPos('V Max', 'HSV Segmentation', 255)
        cv2.createTrackbar('X Position', 'HSV Segmentation', 0, fp_image.shape[1], update_image_hsv)
        cv2.createTrackbar('Y Position', 'HSV Segmentation', 0, fp_image.shape[0], update_image_hsv)
       
        update_image_hsv()
        
        # Only after hitting a key, the loop goes to the next image to repeat
        # the process.
        cv2.waitKey(0)
        cv2.destroyAllWindows()
  

def main():
    """
    The main function to set parameters for each species and call all
    functions required to run the script.

    Returns
    -------
    None.

    """
    bounds_dic = {"B pendula std": [(0, 0, 0), (255, 161, 255), 110, 165],
                  "C nootkatensis std": [(0, 0, 0), (255, 167, 114), 140, 180],
                  "T baccata std": [(70, 0, 0), (255, 255, 255), 110, 165],
                  "C japonica std": [(0, 107, 113), (255, 255, 255), 150, 200],
                  "C lawsoniana std": [(80, 0, 0), (255, 255, 255), 150, 220],
                  "T distichum std": [(0, 70, 50), (255, 255, 255), 110, 180]} # Lower, upper, minrad, maxrad
    
    pollen_dict = process_files()
    find_pollen(pollen_dict, bounds_dic)
    
    

main()