tic

%%
main_folder = '\2024_train_85percent_ori';% your folder
subfolders = dir(main_folder);
subfolders = subfolders([subfolders.isdir]); 
subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'}));

% Define the target number of images after augmentation
target_count = 1200;

% Loop through each subfolder
for folder_idx = 1:length(subfolders)
    current_folder = subfolders(folder_idx).name;
    current_folder_path = fullfile(main_folder, current_folder);
    tif_files = dir(fullfile(current_folder_path, '*.Tif'));
    
    % Calculate the number of images in the current subfolder
    current_count = length(tif_files);
    
    % Calculate the number of images to be added
    images_to_add = target_count - current_count;
    
    % Calculate the number of additional images for each image and the remainder
    N = floor(images_to_add / current_count); % Number of additional images for each original image
    N_more = mod(images_to_add, current_count); % Number of additional images that cannot be evenly distributed among each original image

    % Loop to generate and save the additional images
    for j = 1:current_count
        image = imread(fullfile(current_folder_path, tif_files(j).name));
        image_name= tif_files(j).name;
        if j <= N_more 
            % Generate N+1 images in a loop
            for k = 1:N+1
                enhanced_image = process_image(image);
                new_file_name = sprintf('%s_enhanced_%03d.tif',image_name, k);
                imwrite(enhanced_image, fullfile(current_folder_path, new_file_name));
            end
        else % Generate N images in a loop
            for k = 1:N
                enhanced_image = process_image(image);
                new_file_name = sprintf('%s_enhanced_%03d.tif',image_name, k);
                imwrite(enhanced_image, fullfile(current_folder_path, new_file_name));
            end
        end                   
    end
end

toc

%%

function enhanced_image = process_image(input_image)
    % Randomly flip vertically
    if rand() > 0.5
        input_image = flipud(input_image);
    end

    % Randomly generate the cropping width, minimum 300, maximum not exceeding the width of the original image
    crop_width = randi([300, size(input_image, 2)], 1);
    
     
    % Calculate the starting position for vertical center cropping
    x = floor((size(input_image, 1) - crop_width) / 2) + 1;
    y_sta = size(input_image, 2)-crop_width+1;
    y = randi(y_sta);
    % Crop the image
    cropped_image = imcrop(input_image, [x, y, crop_width, crop_width]);
    
    % Generate two random numbers in the range [0 1], with a difference > 0.7
    lowin = rand*0.25;
    highin= lowin+0.75; 
    lowout=rand*0.3;
    highout=lowout+0.7;

    adjusted_image = imadjust(cropped_image, [lowin, highin], [lowout,highout]);

    % Randomly generate the type of noise
    noise_type = randi([1, 3]); % 1 for Gaussian noise, 2 for Poisson noise, 3 for no noise
    
    % Randomly generate the noise level
    noise_level = 0.001 + 0.007 * rand(); % Random noise level
    
    % Add noise
    if noise_type == 1
        % Add Gaussian noise
        noisy_image = imnoise(adjusted_image, 'gaussian', 0, noise_level);
    elseif noise_type == 2
        % Add Poisson noise
        noisy_image = imnoise(adjusted_image, 'poisson');
        %noisy_image = noisy_image* noise_level ; % Adjust the level of Poisson noise
    else
        % Do not add noise
        noisy_image = adjusted_image;
    end

    enhanced_image = noisy_image;
end