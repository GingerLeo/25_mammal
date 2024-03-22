# 25_mammal
# dataset and trained neural network are available at: https://pan.bnu.edu.cn/l/m10gzj     
# Access code, please inquire with the author.

# The images in Folder '2024_train_85percent_ori' have been cropped to remove the scale bar at the bottom of the electron microscope images, and centered cropping into square images for ease of subsequent processing. 
# The folder retains sampling site information. 
# Reference table for specific sampling sites "samples_counts.csv"

# Using pic_30000.m, images are randomly flipped vertically, cropped along the direction of hair with a minimum unit of [300 300], and contrast is randomly adjusted within a small range (simulating different shooting conditions). Gaussian noise (random noise level between 0.001 and 0.008) or Poisson noise is randomly added, or no noise is added. Please note that this operation will overwrite the original folder. Enhanced generated filenames contain "enhanced". Due to forgetting to set the random seed, if you want to replicate the experimental results, it is recommended to use folder "2024_train_85percent_aug_rename" for subsequent operations.

# For the sake of simplicity, rename the enhanced images to the format of "subfolder name + image number", and obtain folder "2024_train_85percent_aug_rename".

# Modify the fully connected layer and output layer of ResNet-50, save the pre-trained parameters, and train the neural network. Utilize the built-in Experiment Manager app in MATLAB to explore the best hyperparameter combinations.Simply call the function "train_ResNet50_manager" in the Experiment Manager. The "train_ResNet50_manager.mlx" file should be placed in the Experiment Manager project directory.  When splitting the dataset, incorporate some additional image augmentation for the training set images: Randomly scale between 0.5 and 1.5 times the original size; randomly rotate between -45 and 45 degrees; perform random flips along the X and Y axes.

# The trained network is named "trained_ResNet50.mat".

# Use "test_rng97_20240214.m" to validate the network. Generate the top three predicted labels and corresponding scores for the test set images using the network. Generate and visualize the confusion matrix. Generate occlusion sensitivity maps.

# Detailed pixel statistics for sensitivity maps can calculate by "pixels_table.m"

# Use "Parallel Coordinates Plot.m" to draw parallel coordinate plots for the recognition pixel ranges of correctly classified images.

# Use "top3_score_map.m" generate the top three predicted results for the images along with their corresponding sensitivity maps.





