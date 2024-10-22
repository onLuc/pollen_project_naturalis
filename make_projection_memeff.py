import numpy as np
import os
import cv2
from collections import defaultdict
import time


def read_images(path, pollen):
    """
    All slides of a given species are sorted and clustered per position to then
    be combined into seperate projections in the generate_projections function.
    
    Parameters
    ----------
    path : dictionary
        Contains the z-stacks.
    pollen : string
        Name of the pollen species being processed.

    Returns
    -------
    None.
    """
    print(f"Reading and saving {pollen} images")
    
    # Names of all tif files are collected and saved in the files list
    files = [f for f in os.listdir(path) if f.endswith('.tif')]
    
    # For every code there are 20 images that are all contained in a list in
    # the new file_groups folder for easy sorting and processing in the next loop
    file_groups = defaultdict(list)
    for file in files:
        # Since image names between species differ, splitting like this is a
        # generalized way of obtaining the codes to cluster the image names
        code = file.split("_z")[1]
        code = code.split(".")[0]
        code = code.split("m")[1]
        file_groups[code].append(file)

    # The previous for loop guarantees the completeness of the lists in the
    # file_groups dictionary
    for code, file_list in file_groups.items():
        print(f"Processing {pollen} {code}")
        current_stack = []
        
        # All found image names are read by the cv2 module and put in a new
        # folder: current_stack for every z-stack of images. With the current
        # dataset, the current_stack should always be of size 20, but this 
        # changes automatically with different microscope settings
        for file in file_list:
            img = cv2.imread(os.path.join(path, file))
            current_stack.append(img)
        
        generate_projections(code, current_stack, pollen)
    
    print(f"All images of {pollen} saved")


def generate_projections(key, z_stacks, pollen):
    """
    A standard deviation, minimum intensity, and maximum intensity projection
    is generated for the z_stack in this function and saved in seperate folders

    Parameters
    ----------
    key : string
        The code to differentiate this image from the others.
    z_stacks : list
        List of all images in the z-stack.
    pollen : string
        Name of the pollen species being processed.

    Returns
    -------
    None.

    """
    
    z_stack_array = np.array(z_stacks)

    # Standard deviation projection
    std_proj = np.std(z_stack_array, axis=0)
    std_proj_norm = np.uint8(255 * (std_proj - std_proj.min()) / (std_proj.max() - std_proj.min()))
    cv2.imwrite(f"projections/{pollen}/STD/{key}.png", std_proj_norm)

    # Minimum intensity projection
    min_proj = np.min(z_stack_array, axis=0)
    cv2.imwrite(f"projections/{pollen}/MIN/{key}.png", min_proj)

    # Maximum intensity projection 
    max_proj = np.max(z_stack_array, axis=0)
    cv2.imwrite(f"projections/{pollen}/MAX/{key}.png", max_proj)
    

def main():
    """
    Main function where the paths to microscope slide folders for each pollen
    species can be set. Times the run time of creating all projections per species.
    
    Returns
    -------
    None.
    """
    
    # Makes sure the cwd is set to the location of this Python script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Dictionary that contains pairs of species name and microscope slides folder
    species = {
        "C nootkatensis": 'ndor-copy/nootkatensis_full/',
        "B pendula": 'ndor-copy/pendula_full/',
        "T baccata": 'ndor-copy/baccata_full/',
        "C lawsoniana": "ndor-copy/lawsoniana_full/",
        "T distichum": "ndor-copy/Distichum_full/",
        "C japonica": "ndor-copy/2024_05_13__japonica-Image Export_FULL/"
    }
    
    # Iterate through species to create folders beforehand and time the read_images function
    for pollen, path in species.items():
        os.makedirs(f"projections/{pollen}/STD", exist_ok=True)
        os.makedirs(f"projections/{pollen}/MAX", exist_ok=True)
        os.makedirs(f"projections/{pollen}/MIN", exist_ok=True)
        start_time = time.time()
        read_images(path, pollen)
        end_time = time.time()
        tot_time = end_time - start_time
        with open(f"{pollen}_time_projection.txt", 'w') as file:
            file.write(str(tot_time))
        
    print("All done")

main()
