% Set random seed
rng(97);

% Define input and output folders
input_folder = 'your_input_folder';
output_folder = 'your_output_folder';

% Create output folder if it doesn't exist
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

imds = imageDatastore(input_folder, "IncludeSubfolders", true);

customReadFunction = @(filename) repmat(imread(filename), [1, 1, 3]);
imds.ReadFcn = customReadFunction;

% Load pretrained deep learning network
load('\trained_ResNet50.mat','transferNet');
classes = transferNet.Layers(end).Classes;

% Get file paths and labels
filePaths = imds.Files;
[~, labels] = cellfun(@(x) fileparts(fileparts(x)), filePaths, 'UniformOutput', false);
imds.Labels = categorical(labels);

% Create augmented image datastore
auds_test = augmentedImageDatastore([224 224], imds);

%% Predict labels and scores
[hair_pred,scores] = classify(transferNet, auds_test);
[~,topIdx] = maxk(scores, 3,2);
topScores = zeros(size(topIdx));
for i = 1:size(topIdx, 1)
    topScores(i, :) = scores(i, topIdx(i, :));
end
topClasses = classes(topIdx);

% Create a table for top three results and save to Excel
T = table(imds.Labels, topClasses(:,1), topScores(:,1), topClasses(:,2), topScores(:,2), topClasses(:,3), topScores(:,3), ...
    'VariableNames', {'True_Label', 'Top1_Predicted_Label', 'Top1_Score', 'Top2_Predicted_Label', 'Top2_Score', 'Top3_Predicted_Label', 'Top3_Score'});
top_three = fullfile(output_folder, 'top_three_results.xlsx');
writetable(T, top_three);

%% Generate sensitivity maps
for k = 1:length(imds.Files)

    colorImage = readimage(imds,k);

    img_path = imds.Files{k};

    [~, img_name, ~] = fileparts(img_path);
    
    [~, class_label] = fileparts(fileparts(img_path));

    colorImage = imresize(colorImage, [224 224]);

    class_3 = transpose(topClasses(k,:));

    % Calculate occlusion sensitivity map
    topClassesMap = occlusionSensitivity(transferNet, colorImage, class_3, ...
        "Stride", 10, ...
        "MaskSize", 15);
    figure('Position', [100, 100, 1200, 800]);
    
     for m = 1:3
        left = (m - 1) * 1/3 + 0.05;  
        subplot('Position', [left, 0.1, 1/3 - 0.1, 0.8]); 
        imshow(colorImage);
        hold on
        imagesc(topClassesMap(:,:,m), 'AlphaData', 0.5);
        colormap jet;
        colorbars(m) = colorbar;
        classLabel = string(classes(topIdx(k,m)));
        score = topScores(k,m);
        title(sprintf("%s (Score: %.2f)", classLabel, score), 'FontSize', 12);
    end

    output_class_folder = fullfile(output_folder, class_label);
    
    if ~exist(output_class_folder, 'dir')
        mkdir(output_class_folder);
    end
   
    output_image_name = sprintf('%s.fig', img_name);
    
    saveas(gcf, fullfile(output_class_folder, output_image_name));

    close(gcf);

end
