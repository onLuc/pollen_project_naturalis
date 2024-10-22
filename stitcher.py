import cv2
import numpy as np
import os
import time

# To change the location of the cwd to the location of the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def generate_img_dic(folder_dic, cutoffs):
    """
    This function uses the location of all image files to load all images per
    folder and let them be saved by the stitch_and_save function.
    
    Parameters
    ----------
    folder_dic : dictionary
        Contains key-pair values like this: pollen species + type of projection
        and its respective directory.
    cutoffs : dictionary
        A dictionary that specifies the cutoff number per panorama, thus
        specifying out of how many tiles a panorama consists per species.

    Returns
    -------
    None.

    """
    
    # Use the folders in folder_dic to create a new dictionary containing all
    # images.
    image_paths_dic = dict()
    for species, folder_path in folder_dic.items():
        image_paths = [folder_path + "/" + img for img in os.listdir(folder_path)]
        image_paths_dic[species] = image_paths

    # Loops through all images and its corresponding species to read all images
    # per species
    for species, image_paths in image_paths_dic.items():
        start_time = time.time()
        imagedic = dict()
        print(species)
        temp = dict()
        for image_path in image_paths:
            temp[image_path.split("/")[-1].split(".")[0]] = cv2.imread(image_path)
        imagedic[species] = temp
        # When all images from a folder are collected they are saved as 
        # panorama images by stitch_and_save
        stitch_and_save(imagedic, cutoffs)
        end_time = time.time()
        tot_time = end_time - start_time
        # Time per panorama is saved
        with open(f"{species}_stitch_time.txt", 'w') as file:
            file.write(str(tot_time))


def stitch_and_save(imagedic, cutoffs):
    """
    All tiles of min, max or std projection of a species are processed to
    create the proper panoramas 

    Parameters
    ----------
    imagedic : dictionary
        A dictionary containing an image per entry that are used together to be
        made panoramas.
    cutoffs : dictionary
        A dictionary that specifies the cutoff number per panorama, thus
        specifying out of how many tiles a panorama consists per species.

    Returns
    -------
    None.

    """
    
    # Loop through all min, max or std projections of a species
    for species, data in imagedic.items():
        # Set right panorama cutoff per species
        for sp, cosp in cutoffs.items():
            if sp == species[:-4]:
                cutoff = cosp
        
        species = "_".join(species.split(" "))
        counter = 1
        # The first 'result' is the first part of the panorama without the last
        # 200 pixels, so the next image can be add onto the end seamlessly
        result = list(data.values())[0][:, :200] # Start of panorama is first image
        allkeys = [x for x in sorted(data.keys(), key=int)]
        os.makedirs(f"stitched/{species}", exist_ok=True)
        
        # Loops through all key representation of images to create panoramas
        for cycle, key in enumerate(allkeys):
            #If statement checks if next image in queue should be stitched to the existing panorama.
            if (cutoff is None and int(key) - int(allkeys[cycle-1]) != 1 and cycle != 0) or (cutoff is not None and int(key) % cutoff == 0 and cycle != 0):
                cv2.imwrite(f'stitched/{species}/{counter}.png', result)
                result = data[key]
                counter += 1
            # If new image is not part of the previous, start of new panorama
            # is created and loop continues
            else: 
                img_trimmed = data[key][:, 200:]
                result = np.hstack((result, img_trimmed))
        result = data[key]


def main():
    """
    Main function parameters are set for folder locations that are created in
    the make_projection_memeff.py script. Cut off points are set for the
    species whose image codes are continuous.

    Returns
    -------
    None.

    """
    folder_dic = {"B pendula min": 'projections/B pendula/MIN',
                  "B pendula max": 'projections/B pendula/MAX',
                  "B pendula std": 'projections/B pendula/STD',
                  "C nootkatensis min": 'projections/C nootkatensis/MIN',
                  "C nootkatensis max": 'projections/C nootkatensis/MAX',
                  "C nootkatensis std": 'projections/C nootkatensis/STD',
                  "T baccata min": 'projections/T baccata/MIN',
                  "T baccata max": 'projections/T baccata/MAX',
                  "T baccata std": 'projections/T baccata/STD',
                  "C lawsoniana min": 'projections/C lawsoniana/MIN',   
                  "C lawsoniana max": 'projections/C lawsoniana/MAX',
                  "C lawsoniana std": 'projections/C lawsoniana/STD',
                  "T distichum min": 'projections/T distichum/MIN',
                  "T distichum max": 'projections/T distichum/MAX',
                  "T distichum std": 'projections/T distichum/STD',
                  "C japonica min": 'projections/C japonica/MIN',
                  "C japonica max": 'projections/C japonica/MAX',
                  "C japonica std": 'projections/C japonica/STD'}
    
    cutoffs = {"B pendula": 33,
               "C nootkatensis": 33,
               "T baccata": None,
               "C lawsoniana": None,
               "T distichum": None,
               "C japonica": None}
    
    generate_img_dic(folder_dic, cutoffs)


main()