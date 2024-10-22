Readme for pollen recognition and classification project.

Obtain Python scripts and necessary libraries:
1. Clone the respoitory
	git clone https://github.com/onLuc/pollen_project_naturalis
2. Navigate to the project directory
	cd pollen_project_naturalis
3. Install the needed libraries in your environment
	pip install libraries.txt

Data can be found here (Naturalis EduVPN connection required):
https://sram.surf.nl/invitations/accept/AxUb6-4SXdrxutQ96m1sIzVAFHPMVqGm-37ArmxzCv24

============
All python scripts in run order.

Python script: make_projection_memeff.py
Description: Make std, min, and max projections for all images located in the folders defined in the species dictionary.
Input: Any amount of folders that contain z-stacks of images, formatted: {date+species}_z{int}m{int}.tif.
	   This is necessary to identify which images (identified by m) belong to the same z-stack.
Output: projections/species/{MIN/MAX/STD}/{key}.png. Each species in the input gets its own output folder and all MIN/MAX/STD
		projections are saved per type of projection.
Required libraries: NumPy 1.26.4 and OpenCV 4.10.0.84.
------------
Python script: stitcher.py
Description: All projections that lie next to each other get connected to make a panorama.
Input: The output from make_projection_memeff.py, defined in the folder_dic dictionary.
Output: stitched/{species}_{projection_type}/{counter}.png. A panorama folder is created for each species - projection combination.
Required libraries: NumPy 1.26.4.
------------
Python script: pollen_finder4.py
Description: Find loose lying pollen in the panoramas and save the pollen as seperate pictures
Input: stitched/{species}_{projection_type}/{counter}.png.
Output: nn_input/{species}/{counter_key}.png.
Required libraries: NumPy 1.26.4 and OpenCV 4.10.0.84.
------------
Python script: resnet_complete.py
Description: Takes all images from nn_input and uses folder names within the directory to determine which classes exist in the dataset
Input: nn_input/{species}
Output: models/{model_type}.keras,
		nn_plots/{model_type}_time_{time}_valacc_{val_acc}_valloss_{val_loss}_plot.png,
		confusion_matrices/{model_type}_confusion_matrix.png
Required libraries: tensorflow 2.17.0, scikit-learn 1.5.2, matplotlib 3.9.2, and numpy 1.26.4.
------------
Python script: resnet_complete_kfold.py
Description: Validates the ResNet models by applying k-fold cross-validation with 5-fold as the default option.
Input: nn_input/{species}
Output: kfold/{model_type}.txt, # Every txt file contains the precision, recall, and f1-score per species.
		kfold_times/{model_type}.txt # Contains time take to train folds.
Required libraries: tensorflow 2.17.0, scikit-learn 1.5.2, matplotlib 3.9.2, and numpy 1.26.4
------------
Python script: pollen_finder4 live.py
Description: A script used to assist with finding the right HSV values for color thresholding in pollen_finder4.py per species.
Input: stitched/{species}
Output: None
Required libraries: NumPy 1.26.4 and OpenCV 4.10.0.84.
------------

To save time, all scripts can be run subequently by using this command: python make_projection_memeff.py && python stitcher.py && ...