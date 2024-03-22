% Set random seed
rng(97);

inputFolder = '\2024_test_15percent_aug_rename';
outputFolder = 'your_outputFolder';

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

imds = imageDatastore(inputFolder, "IncludeSubfolders", true);

readImageFcn = @(filename) repmat(imread(filename), [1, 1, 3]);

imds.ReadFcn = readImageFcn;

load('\trained_ResNet50.mat', 'transferNet');

filePaths = imds.Files;
[~, labels] = cellfun(@(x) fileparts(fileparts(x)), filePaths, 'UniformOutput', false);
imds.Labels = categorical(labels);

audsTest = augmentedImageDatastore([224 224], imds);

[hairPred, scores] = classify(transferNet, audsTest);

%% Generate Occlusion Sensitivity Maps

resultsTable = cell2table(cell(0, 26), 'VariableNames', {'Class_Label', 'Predicted_Label', 'Image_Name', 'Total_Pixels', 'Max_Occlusion_Sensitivity', 'Min_Occlusion_Sensitivity', 'Pixels_Above_Threshold_75', 'Pixels_Between_Threshold_75_50', 'Pixels_Between_Threshold_50_25', 'Pixels_Below_Threshold_25', 'Recognized_Pixels', 'Confusion_Pixels', 'Negative_Pixels_Above_Threshold_25', 'Negative_Pixels_Between_Threshold_25_50', 'Negative_Pixels_Between_Threshold_50_75', 'Negative_Pixels_Below_Threshold_75', 'Recognized_Pixels_Percentage', 'Confusion_Pixels_Percentage', 'Pixels_Above_Threshold_75_Percentage', 'Pixels_Between_Threshold_75_50_Percentage', 'Pixels_Between_Threshold_50_25_Percentage', 'Pixels_Below_Threshold_25_Percentage', 'Negative_Pixels_Above_Threshold_25_Percentage', 'Negative_Pixels_Between_Threshold_25_50_Percentage', 'Negative_Pixels_Between_Threshold_50_75_Percentage', 'Negative_Pixels_Below_Threshold_75_Percentage'});

for i = 1:numel(labels)

    disp(['Processing image ', num2str(i), ' out of ', num2str(numel(labels))]);

    img = readimage(imds, i);

    imgPath = imds.Files{i};

    [~, imgName, ~] = fileparts(imgPath);

    [~, classLabel] = fileparts(fileparts(imgPath));

    trueLabel = labels(i);

    predictedLabel = hairPred(i);

    % Resize the image to [224 224]
    resizedImage = imresize(img, [224 224]);
    
    % Calculate occlusion sensitivity map
    map = occlusionSensitivity(transferNet, resizedImage, predictedLabel);

    % Calculate the maximum and minimum values of occlusion sensitivity map
    maxValue = max(map(:));
    minValue = min(map(:));

    % Calculate the percentages of recognized and confusion pixels
    totalPixels = numel(map);
    recognizedPixels = sum(map(:) > 0);
    confusionPixels = sum(map(:) <= 0);
    recognizedPixelsPercentage = recognizedPixels / totalPixels;
    confusionPixelsPercentage = confusionPixels / totalPixels;

    if minValue < 0 && maxValue > 0
        % Calculate thresholds for positive values
        threshold75 = maxValue - 0.25 * (maxValue - 0);
        threshold50 = maxValue - 0.50 * (maxValue - 0);
        threshold25 = maxValue - 0.75 * (maxValue - 0);
        
        % Count pixels in different intervals
        pixelsAbove75 = sum(map(:) > threshold75);
        pixels75To50 = sum(map(:) <= threshold75 & map(:) > threshold50);
        pixels50To25 = sum(map(:) <= threshold50 & map(:) > threshold25);
        pixelsBelow25 = sum(map(:) <= threshold25 & map(:) > 0);
    
        pixelsAbove75Percentage = pixelsAbove75 / totalPixels;
        pixels75To50Percentage = pixels75To50 / totalPixels;
        pixels50To25Percentage = pixels50To25 / totalPixels;
        pixelsBelow25Percentage = pixelsBelow25 / totalPixels;
    
        % Count negative pixel values
        negativeThreshold25 = 0 - 0.25 * (0 - minValue);
        negativeThreshold50 = 0 - 0.50 * (0 - minValue);
        negativeThreshold75 = 0 - 0.75 * (0 - minValue);
    
        negativePixelsAbove25 = sum(map(:) > negativeThreshold25 & map(:) < 0) ;
        negativePixels25To50 = sum(map(:) <= negativeThreshold25 & map(:) > negativeThreshold50);
        negativePixels50To75 = sum(map(:) <= negativeThreshold50 & map(:) > negativeThreshold75);
        negativePixelsBelow75 = sum(map(:) <= negativeThreshold75);
    
        
        negativePixelsAbove25Percentage = negativePixelsAbove25 / totalPixels;
        negativePixels25To50Percentage = negativePixels25To50 / totalPixels;
        negativePixels50To75Percentage = negativePixels50To75 / totalPixels;
        negativePixelsBelow75Percentage = negativePixelsBelow75 / totalPixels;
   
    elseif minValue >=0 
        % Calculate thresholds for positive values
        threshold_75 = maxValue - 0.25 * (maxValue - minValue);
        threshold_50 = maxValue - 0.50 * (maxValue - minValue);
        threshold_25 = maxValue - 0.75 * (maxValue - minValue);

        % Count pixel values
        pixelsAbove75 = sum(map(:) > threshold_75);
        pixels75To50 = sum(map(:) <= threshold_75 & map(:) > threshold_50);
        pixels50To25 = sum(map(:) <= threshold_50 & map(:) > threshold_25);
        pixelsBelow25 = sum(map(:) <= threshold_25);

        pixelsAbove75Percentage = pixelsAbove75 / totalPixels ;
        pixels75To50Percentage = pixels75To50 / totalPixels ;
        pixels50To25Percentage = pixels50To25 / totalPixels ;
        pixelsBelow25Percentage = pixelsBelow25 / totalPixels ;

        negativePixelsAbove25 = 0;
        negativePixels25To50 = 0;
        negativePixels50To75 = 0;
        negativePixelsBelow75 = 0;
    
        
        negativePixelsAbove25Percentage = 0;
        negativePixels25To50Percentage = 0;
        negativePixels50To75Percentage = 0;
        negativePixelsBelow75Percentage = 0;


    elseif maxValue <= 0
        % Count negative pixel values
        negativeThreshold25 = maxValue - 0.25 * (maxValue - minValue);
        negativeThreshold50 = maxValue - 0.50 * (maxValue - minValue);
        negativeThreshold75 = maxValue - 0.75 * (maxValue - minValue);
    
        negativePixelsAbove25 = sum(map(:) > negativeThreshold25);
        negativePixels25To50 = sum(map(:) <= negativeThreshold25 & map(:) > negativeThreshold50);
        negativePixels50To75 = sum(map(:) <= negativeThreshold50 & map(:) > negativeThreshold75);
        negativePixelsBelow75 = sum(map(:) <= negativeThreshold75);
    
        
        negativePixelsAbove25Percentage = negativePixelsAbove25 / totalPixels;
        negativePixels25To50Percentage = negativePixels25To50 / totalPixels;
        negativePixels50To75Percentage = negativePixels50To75 / totalPixels;
        negativePixelsBelow75Percentage = negativePixelsBelow75 / totalPixels;

        pixelsAbove75 = 0;
        pixels75To50 = 0;
        pixels50To25 = 0;
        pixelsBelow25 = 0;

        pixelsAbove75Percentage = 0;
        pixels75To50Percentage = 0;
        pixels50To25Percentage = 0;
        pixelsBelow25Percentage = 0;

    end

    resultsTable = [resultsTable; {classLabel, predictedLabel, imgName, totalPixels, maxValue, minValue, pixelsAbove75, pixels75To50, pixels50To25, pixelsBelow25, recognizedPixels, confusionPixels, negativePixelsAbove25, negativePixels25To50, negativePixels50To75, negativePixelsBelow75, recognizedPixelsPercentage, confusionPixelsPercentage, pixelsAbove75Percentage, pixels75To50Percentage, pixels50To25Percentage, pixelsBelow25Percentage, negativePixelsAbove25Percentage, negativePixels25To50Percentage, negativePixels50To75Percentage, negativePixelsBelow75Percentage}];
end

writetable(resultsTable, fullfile(outputFolder, 'results_table_sankey.csv'));