% Set the random seed
rng(97);

input_folder = '\2024_test_15percent_aug_rename';
output_folder = '\your_output_folder';

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

imds = imageDatastore(input_folder, "IncludeSubfolders", true);

customReadFunction = @(filename) repmat(imread(filename), [1, 1, 3]);

imds.ReadFcn = customReadFunction;

% Load the pre-trained deep learning network
load('\MammalHairNet.mat','transferNet');

filePaths = imds.Files;
[~, labels] = cellfun(@(x) fileparts(fileparts(x)), filePaths, 'UniformOutput', false);
imds.Labels = categorical(labels);

auds_test = augmentedImageDatastore([224 224], imds);

%% Scoring
% Test the model
[hair_pred,scores] = classify(transferNet, auds_test);

% Initialize a cell array to store the results for each sample
results = cell(size(scores, 1), 7); 

% Get all labels in the dataset
labels_25 = categories(imds.Labels);

% Loop through each sample
for i = 1:size(scores, 1)
    % Get the probability scores for the current sample
    sample_scores = scores(i, :);
    % Get the image filename for the current sample (without extension)
    [~, filename, ~] = fileparts(imds.Files{i});
    
    
    % Find the top three classes with the highest probability scores
    [~, sorted_idx] = sort(sample_scores, 'descend');
    top_three_idx = sorted_idx(1:3);
    top_three_scores = sample_scores(top_three_idx);
    
    % Get the corresponding labels for the top three classes
    top_three_labels = cell(3, 1);
    for j = 1:3
        top_three_labels{j} = char(labels_25(top_three_idx(j)));
    end
    
    % Store in the results array
    results{i, 1} = filename; % Image filename (without extension) as the true label
    results{i, 2} = char(top_three_labels(1)); % Top predicted class with highest score
    results{i, 3} = top_three_scores(1); % Top score
    results{i, 4} = char(top_three_labels(2)); % Second top predicted class
    results{i, 5} = top_three_scores(2); % Second top score
    results{i, 6} = char(top_three_labels(3)); % Third top predicted class
    results{i, 7} = top_three_scores(3); % Third top score
end

T = cell2table(results, 'VariableNames', {'True_Label', 'Top1_Predicted_Label', 'Top1_Score', 'Top2_Predicted_Label', 'Top2_Score', 'Top3_Predicted_Label', 'Top3_Score'});

top_three = fullfile(output_folder, 'ResNet50_25Species_8_64_01_top_three_results.xlsx');
writetable(T, top_three);

%% Generate score table and confusion matrix

hair_test = imds.Labels;
accuracy = sum(hair_pred == hair_test) / numel(hair_test);

C = confusionmat(hair_test, hair_pred);
numClasses = size(C, 1);
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1Score = zeros(numClasses, 1);
for i = 1:numClasses
    tp = C(i, i);
    fp = sum(C(:, i)) - tp;
    fn = sum(C(i, :)) - tp;
    precision(i) = tp / (tp + fp);
    recall(i) = tp / (tp + fn);
    f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end

classNames = unique(imds.Labels);

resultsTable = table();
resultsTable.Accuracy = accuracy;
resultsTable.MeanPrecision = mean(precision);
resultsTable.MeanRecall = mean(recall);
resultsTable.MeanF1Score = mean(f1Score);

results_table_file = fullfile(output_folder, 'results_table_8_64_01.csv');
writetable(resultsTable, results_table_file);


% Save the confusion matrix
confusion_matrix_file = fullfile(output_folder, 'ResNet50_25Species_confusion_matrix_8_64_01.mat');
save(confusion_matrix_file, 'C');

%% Plot the confusion matrix

figure;
cm = confusionchart(hair_test, hair_pred);

% Set row summary to 'row-normalized' to display true positive rate and false positive rate
cm.RowSummary = 'row-normalized';

% Set column summary to 'column-normalized' to display positive prediction value and false discovery rate
cm.ColumnSummary = 'column-normalized';

% Adjust the chart size to display percentages
fig_Position = cm.Parent.Position;
fig_Position(3) = fig_Position(3) * 1.5;
cm.Parent.Position = fig_Position;

% Sort the confusion matrix based on true positive rate
cm.Normalization = 'row-normalized'; 
sortClasses(cm,'descending-diagonal');
cm.Normalization = 'absolute';

% Sort the confusion matrix based on positive prediction value
cm.Normalization = 'column-normalized';
sortClasses(cm,'descending-diagonal');
cm.Normalization = 'absolute';

confusion_matrix_fig = fullfile(output_folder, 'ResNet50_25Species_confusion_matrix_8_64_01.fig');
saveas(cm.Parent, confusion_matrix_fig);

%% Plot occlusion sensitivity maps

% Create a table to save the results
results_table = cell2table(cell(0, 16), 'VariableNames', {'Correct_Label', 'Class_Label', 'Image_Name', 'Total_Pixels', 'Recognized_Pixels', 'Confusion_Pixels', 'Pixels_above_75', 'Pixels_75_to_50', 'Pixels_50_to_25', 'Pixels_below_25', 'Recognized_Pixels_Percentage', 'Confusion_Pixels_Percentage', 'Pixels_above_75_Percentage', 'Pixels_75_to_50_Percentage', 'Pixels_50_to_25_Percentage', 'Pixels_below_25_Percentage'});

% Process each image one by one
for i = 1:numel(labels)

    img = readimage(imds, i);

    img_path = imds.Files{i};

    [~, img_name, ~] = fileparts(img_path);
    
    [~, class_label] = fileparts(fileparts(img_path));

    true_label = labels(i);

    % Get the predicted label
    predicted_label = hair_pred(i);

    % Resize the image to [224 224]
    resizedImage = imresize(img, [224 224]);
    
    % Compute occlusion sensitivity map
    map = occlusionSensitivity(transferNet,resizedImage,predicted_label);

    % Convert the three-channel image to grayscale
    grayImage = rgb2gray(resizedImage);

    blended_image = ind2rgb(gray2ind(grayImage), jet(256)); % Convert grayscale image to pseudocolor image
    blended_image(:,:,1) = map/max(map(:)); % Map the occlusion sensitivity map to the red channel
    imshow(blended_image);

    % 1. Calculate the maximum and minimum values of the occlusion sensitivity map, and calculate recognized pixels and confusion pixels
    max_value = max(map(:));
    min_value = min(map(:));
    recognized_pixels = sum(map(:) > 0);
    confusion_pixels = sum(map(:) <= 0);
    
    % 2. Calculate segmented thresholds
    threshold_75 = max_value - 0.25*(max_value - min_value);
    threshold_50 = max_value - 0.50*(max_value - min_value);
    threshold_25 = max_value - 0.75*(max_value - min_value);
    
    % 3. Count pixel numbers
    pixels_above_75 = sum(map(:) > threshold_75);
    pixels_75_to_50 = sum(map(:) <= threshold_75 & map(:) > threshold_50);
    pixels_50_to_25 = sum(map(:) <= threshold_50 & map(:) > threshold_25);
    pixels_below_25 = sum(map(:) <= threshold_25);
    
    % 4. Calculate percentages
    total_pixels = numel(map);
    
    recognized_pixels_percentage = recognized_pixels / total_pixels ;
    confusion_pixels_percentage = confusion_pixels / total_pixels ;
    pixels_above_75_percentage = pixels_above_75 / total_pixels ;
    pixels_75_to_50_percentage = pixels_75_to_50 / total_pixels ;
    pixels_50_to_25_percentage = pixels_50_to_25 / total_pixels ;
    pixels_below_25_percentage = pixels_below_25 / total_pixels ;
    
    output_class_folder = fullfile(output_folder, class_label);
    
    if ~exist(output_class_folder, 'dir')
        mkdir(output_class_folder);
    end
    
    output_image_name = sprintf('%s_%s_%.2f.png', img_name, predicted_label, recognized_pixels_percentage);
    
    % Save the overlaid image
    imwrite(blended_image, fullfile(output_class_folder, output_image_name));
    
    results_table = [results_table; {class_label, predicted_label, img_name, total_pixels, recognized_pixels, confusion_pixels, pixels_above_75, pixels_75_to_50, pixels_50_to_25, pixels_below_25, recognized_pixels_percentage, confusion_pixels_percentage, pixels_above_75_percentage, pixels_75_to_50_percentage, pixels_50_to_25_percentage, pixels_below_25_percentage}];
end

writetable(results_table, fullfile(output_folder, 'results_table_pixels.csv'));
