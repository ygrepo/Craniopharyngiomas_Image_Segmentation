% Specify the directory containing the image and mask files
dataDir = '../data/';

% Get a list of all image and mask files in the directory
imageFiles = dir(fullfile(dataDir, 'vs_gk_*_t1_3D_aligned_volume.nii'));
maskFiles = dir(fullfile(dataDir, 'vs_gk_*_t1_aligned_vol_mask.nii'));

% Initialize cell arrays to store paths for images and masks
imagePaths = {};
maskPaths = {};

% Pair image and mask files based on matching identifiers
for i = 1:numel(imageFiles)
    % Full path of the image file
    imagePath = fullfile(dataDir, imageFiles(i).name);
    
    % Extract the full identifier (e.g., 'vs_gk_36_t1') from the image file name
    identifier = regexp(imageFiles(i).name, 'vs_gk_\d+_t1', 'match', 'once');
    
    % Find the corresponding mask file with the exact identifier
    maskIdx = find(contains({maskFiles.name}, identifier));
    
    % Check if exactly one mask file is found for the identifier
    if isscalar(maskIdx)
        % Full path of the mask file
        maskPath = fullfile(dataDir, maskFiles(maskIdx).name);
        
        % Store the matched paths
        imagePaths{end + 1} = imagePath;
        maskPaths{end + 1} = maskPath;
    elseif numel(maskIdx) > 1
        % Print a warning if multiple mask files are found
        fprintf('Warning: Multiple masks found for %s\n', identifier);
    else
        % Print a warning if no matching mask file is found
        fprintf('Warning: No mask found for %s\n', imageFiles(i).name);
    end
end

% Display the matched pairs to verify correctness
for j = 1:numel(imagePaths)
    fprintf('Image: %s\n', imagePaths{j});
    fprintf('Mask: %s\n\n', maskPaths{j});
end

% Split data into training and test sets (e.g., 80% training, 20% test)
numSamples = numel(imagePaths);
randIndices = randperm(numSamples);
numTrain = round(0.8 * numSamples);

trainIndices = randIndices(1:numTrain);
testIndices = randIndices(numTrain + 1:end);

trainingImagePaths = imagePaths(trainIndices);
trainingMaskPaths = maskPaths(trainIndices);

testImagePaths = imagePaths(testIndices);
testMaskPaths = maskPaths(testIndices);

% Display the number of samples in each set
fprintf('Number of training samples: %d\n', numel(trainingImagePaths));
fprintf('Number of test samples: %d\n', numel(testImagePaths));
%%
% Initialize variables to accumulate intensity values, spatial information, and areas
intensityValues = [];
centroids = [];
boundingBoxDiagonals = [];
tumorAreas = [];  % To store area of tumor regions for min and max area calculation

% Loop through the training set
for i = 1:numel(trainingImagePaths)
    % Load image and mask
    fprintf("Image Data:%s\n", trainingImagePaths{i});
    fprintf("Mask:%s\n", trainingMaskPaths{i});
    imageData = niftiread(trainingImagePaths{i});
    maskData = niftiread(trainingMaskPaths{i});
    
    % Convert to double for processing
    imageData = double(imageData);
    maskData = logical(maskData);  % Ensure mask is binary
    
    % Find the slice with the largest tumor region
    tumorPixelCounts = squeeze(sum(sum(maskData, 1), 2));
    [~, maxSliceIndex] = max(tumorPixelCounts);
    sliceData = imageData(:, :, maxSliceIndex);
    maskSlice = maskData(:, :, maxSliceIndex);
    
    % Collect intensity values within the tumor region
    intensityValues = [intensityValues; sliceData(maskSlice)];
    
    % Calculate the centroid, bounding box, and area of the tumor region in this slice
    stats = regionprops(maskSlice, 'Centroid', 'BoundingBox', 'Area');
    if ~isempty(stats)
        % Centroid
        centroids = [centroids; stats.Centroid];
        
        % Bounding box diagonal
        boundingBox = stats.BoundingBox;
        boundingBoxDiagonal = sqrt(boundingBox(3)^2 + boundingBox(4)^2);
        boundingBoxDiagonals = [boundingBoxDiagonals; boundingBoxDiagonal];
        
        % Tumor area
        tumorAreas = [tumorAreas; stats.Area];
    end
end

% Calculate tumorIntensityRange as the min and max of all intensity values
tumorIntensityRange = [min(intensityValues), max(intensityValues)];

% Calculate roiCenter as the mean of the centroids
roiCenter = mean(centroids, 1);

% Calculate roiRadius as the mean of bounding box diagonals, representing a typical tumor size
roiRadius = mean(boundingBoxDiagonals) / 2;  % Approximate radius

% Calculate meanRadius and stdRadius from boundingBoxDiagonals
meanRadius = mean(boundingBoxDiagonals) / 2;  % Average radius
stdRadius = std(boundingBoxDiagonals) / 2;    % Standard deviation of radius


% Calculate minArea and maxArea based on the training set tumor areas
minArea = min(tumorAreas);
maxArea = max(tumorAreas);

% Display the extracted information
fprintf('Tumor Intensity Range: [%f, %f]\n', tumorIntensityRange(1), tumorIntensityRange(2));
fprintf('ROI Center: [%.2f, %.2f]\n', roiCenter(1), roiCenter(2));
fprintf('ROI Radius: %.2f\n', roiRadius);
fprintf('Mean Radius: %.2f\n', meanRadius);
fprintf('Standard Deviation of Radius: %.2f\n', stdRadius);
fprintf('Min Tumor Area: %.2f\n', minArea);
fprintf('Max Tumor Area: %.2f\n', maxArea);

%%
% Parameters derived from training set
nClusters = 5;  % Number of clusters for FCM

% Assume these values were calculated from the training set as shown previously
% tumorIntensityRange = [minIntensity, maxIntensity]; % Calculated from training
% roiCenter = [xCenter, yCenter]; % Calculated from training
% roiRadius = radius; % Calculated from training
% meanRadius, stdRadius = meand and standard radius; % Calculated from training
% minArea = minimum tumor area observed in training set
% maxArea = maximum tumor area observed in training set
detectedMasks = cell(size(testImagePaths));  % Store results for each test image

for i = 1:numel(testImagePaths)
    % Extract patient information from the file path
    [~, fileName, ~] = fileparts(testImagePaths{i});  % Extract file name without extension
    patientInfo = split(fileName, '_');  % Split file name by underscores
    patientID = join(patientInfo(1:3), ':');  % Join the first three parts (e.g., vs:gk:25)
    patientID = patientID{1};  % Convert to string
    
    % Load the test image and ground truth mask
    imageData = niftiread(testImagePaths{i});
    maskData = niftiread(testMaskPaths{i});  % Load the original ground truth mask
    
    % Calculate maxSliceIndex based on the slice with the largest area
    sliceTumorPixelCounts = squeeze(sum(sum(maskData, 1), 2));
    [~, maxSliceIndex] = max(sliceTumorPixelCounts);
    
    % Select the slice and the corresponding mask slice
    sliceData = imageData(:, :, maxSliceIndex);
    maskSlice = logical(maskData(:, :, maxSliceIndex));
    
    % Call the detection function with test image and learned parameters
    fprintf("Image:%s\n", testImagePaths{i});
    detectedMask = fcm_predict_tumor(testImagePaths{i},[],...
        nClusters, tumorIntensityRange, minArea, maxArea,...
        roiCenter, roiRadius, meanRadius, stdRadius, ...
        maxSliceIndex, true);

    % Store the detected mask for further use if needed
    detectedMasks{i} = detectedMask;
    
    % Display the original slice with the ground truth tumor overlay
    figure;
    t = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, sprintf('Patient: %s - Ground Truth (left) and Predicted (right)', patientID), ...
        'FontSize', 24, 'FontWeight', 'bold');
    
    % Display ground truth overlay
    nexttile;
    imshow(sliceData, []);
    hold on;
    % Create a red overlay for the ground truth mask
    originalOverlay = cat(3, ones(size(maskSlice)), zeros(size(maskSlice)), zeros(size(maskSlice)));
    hOriginal = imshow(originalOverlay);
    set(hOriginal, 'AlphaData', maskSlice * 0.5);  
    title('Original Test Slice with Ground Truth Tumor Overlay', 'FontSize', 24, 'FontWeight', 'bold');
    hold off;
    
    % Display predicted tumor overlay
    nexttile;
    imshow(sliceData, []);
    hold on;
    % Create a red overlay for the predicted tumor region
    predictedOverlay = cat(3, ones(size(detectedMask)), zeros(size(detectedMask)), zeros(size(detectedMask)));
    hPredicted = imshow(predictedOverlay);
    set(hPredicted, 'AlphaData', detectedMask * 0.5);  
    title('Original Test Slice with Predicted Tumor Overlay', 'FontSize', 24, 'FontWeight', 'bold');
    hold off;
end



%% Same predictions but without plot ----
nClusters = 5;  % Number of clusters for FCM

% Initialize slice indices for each image
sliceIndices = zeros(numel(testImagePaths), 1);

% Process each test image and store detected masks
detectedMasks = cell(size(testImagePaths));

for i = 1:numel(testImagePaths)
    % Load the test image and ground truth mask
    imageData = niftiread(testImagePaths{i});
    maskData = niftiread(testMaskPaths{i});  % Load the original ground truth mask
    
    % Calculate maxSliceIndex based on the slice with the largest tumor area
    sliceTumorPixelCounts = squeeze(sum(sum(maskData, 1), 2));
    [~, maxSliceIndex] = max(sliceTumorPixelCounts);
    sliceIndices(i) = maxSliceIndex;  % Store max slice index
    
    % Select the slice and the corresponding mask slice
    sliceData = imageData(:, :, maxSliceIndex);

    % Call the detection function with test image and learned parameters
    fprintf("Image:%s\n", testImagePaths{i})
    detectedMask = fcm_predict_tumor(testImagePaths{i}, [],...
        nClusters, tumorIntensityRange, minArea, maxArea,...
        roiCenter, roiRadius, meanRadius, stdRadius, ...
        maxSliceIndex, false);
    
    % Store the detected mask
    detectedMasks{i} = detectedMask;
end

% Evaluate performance metrics for the specific slices
result = evaluateTestSetSlice(testImagePaths, testMaskPaths, detectedMasks, sliceIndices);

% Display results
disp(result);

%% Histogram Matching with plots ----
% Specify the path to the known reference image and mask
referenceImagePath = fullfile(dataDir, 'vs_gk_5_t1_3D_aligned_volume.nii');
referenceMaskPath = fullfile(dataDir, 'vs_gk_5_t1_aligned_vol_mask.nii');

% Load the reference image and mask
referenceImageData = niftiread(referenceImagePath);
referenceMaskData = niftiread(referenceMaskPath);

% Convert the reference mask to logical for processing
referenceMaskData = logical(referenceMaskData);

% Calculate maxSliceIndex for the reference image based on the largest tumor area
tumorPixelCountsRef = squeeze(sum(sum(referenceMaskData, 1), 2));
[~, maxSliceIndexRef] = max(tumorPixelCountsRef);

% Select the reference slice for histogram matching
referenceSlice = referenceImageData(:, :, maxSliceIndexRef);

% Parameters derived from training set
nClusters = 5;  % Number of clusters for FCM

% Assume these values were calculated from the training set as shown previously
% tumorIntensityRange = [minIntensity, maxIntensity]; % Calculated from training
% roiCenter = [xCenter, yCenter]; % Calculated from training
% roiRadius = radius; % Calculated from training
% meanRadius, stdRadius = meand and standard radius; % Calculated from training
% minArea = minimum tumor area observed in training set
% maxArea = maximum tumor area observed in training set

detectedMasks = cell(size(testImagePaths));  % Store results for each test image

for i = 1:numel(testImagePaths)
    % Extract patient information from the file path
    [~, fileName, ~] = fileparts(testImagePaths{i});  % Extract file name without extension
    patientInfo = split(fileName, '_');  % Split file name by underscores
    patientID = join(patientInfo(1:3), ':');  % Join the first three parts (e.g., vs:gk:25)
    patientID = patientID{1};  % Convert to string

    % Load the test image and its ground truth mask
    imageData = niftiread(testImagePaths{i});
    maskData = niftiread(testMaskPaths{i});  % Load the ground truth mask
    
    % Convert mask to logical format
    maskData = logical(maskData);

    % Calculate maxSliceIndex based on the slice with the largest area in the test mask
    sliceTumorPixelCounts = squeeze(sum(sum(maskData, 1), 2));
    [~, maxSliceIndex] = max(sliceTumorPixelCounts);
    
    % Select the test slice and match its histogram to the reference slice
    testSlice = imageData(:, :, maxSliceIndex);
    matchedSlice = imhistmatch(testSlice, referenceSlice);  % Histogram match to reference

    % Replace the slice in the image data with the histogram-matched version
    % imageData(:, :, maxSliceIndex) = matchedSlice;
    for j = 1:size(imageData, 3)
     imageData(:, :, j) = imhistmatch(imageData(:, :, j), referenceSlice);
    end
    
    % Call the detection function with the histogram-matched test image and learned parameters
    fprintf("Image:%s\n", testImagePaths{i})
    detectedMask = fcm_predict_tumor(testImagePaths{i}, imageData, ...
        nClusters, tumorIntensityRange, minArea, maxArea,...
        roiCenter, roiRadius, meanRadius, stdRadius, ...
        maxSliceIndex, true);

    % Store the detected mask for further use if needed
    detectedMasks{i} = detectedMask;
    
    % Display the original slice with the ground truth tumor overlay
    figure;
    t = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, sprintf('Patient: %s - Ground Truth (left), Histogram Matched (middle) and Predicted (right)', patientID), ...
        'FontSize', 24, 'FontWeight', 'bold');
       
    % Display predicted tumor overlay
    nexttile;
    imshow(testSlice, []);
    hold on;
    % Create a red overlay for the ground truth mask
    originalOverlay = cat(3,...
        ones(size(maskData(:, :, maxSliceIndex))),...
        zeros(size(maskData(:, :, maxSliceIndex))),...
        zeros(size(maskData(:, :, maxSliceIndex))));
    hOriginal = imshow(originalOverlay);
    set(hOriginal, 'AlphaData', maskData(:, :, maxSliceIndex) * 0.5);  
    title('Original Test Slice with Ground Truth Tumor Overlay', 'FontSize', 24, 'FontWeight', 'bold');
    hold off;
    
    % Display the histogram-matched test slice
    nexttile;
    imshow(matchedSlice, []);
    title('Histogram Matched Test Slice', 'FontSize', 24, 'FontWeight', 'bold');
    
    % Display the detected tumor overlay on the original test slice
    nexttile;
    imshow(testSlice, []);
    hold on;
    % Create a red overlay for the predicted tumor region
    predictedOverlay = cat(3, ones(size(detectedMask)), zeros(size(detectedMask)), zeros(size(detectedMask)));
    hPredicted = imshow(predictedOverlay);
    set(hPredicted, 'AlphaData', detectedMask * 0.5);  
    title('Original Test Slice with Predicted Tumor Overlay', 'FontSize', 24, 'FontWeight', 'bold');
    hold off;
end

%% Histogram Matching without plots ----
% Specify the path to the known reference image and mask
referenceImagePath = fullfile(dataDir, 'vs_gk_5_t1_3D_aligned_volume.nii');
referenceMaskPath = fullfile(dataDir, 'vs_gk_5_t1_aligned_vol_mask.nii');

% Load the reference image and mask
referenceImageData = niftiread(referenceImagePath);
referenceMaskData = niftiread(referenceMaskPath);

% Convert the reference mask to logical for processing
referenceMaskData = logical(referenceMaskData);

% Calculate maxSliceIndex for the reference image based on the largest tumor area
tumorPixelCountsRef = squeeze(sum(sum(referenceMaskData, 1), 2));
[~, maxSliceIndexRef] = max(tumorPixelCountsRef);

% Select the reference slice for histogram matching
referenceSlice = referenceImageData(:, :, maxSliceIndexRef);

% Parameters derived from training set
nClusters = 5;  % Number of clusters for FCM

% Assume these values were calculated from the training set as shown previously
% tumorIntensityRange = [minIntensity, maxIntensity]; % Calculated from training
% roiCenter = [xCenter, yCenter]; % Calculated from training
% roiRadius = radius; % Calculated from training
% meanRadius, stdRadius = meand and standard radius; % Calculated from training
% minArea = minimum tumor area observed in training set
% maxArea = maximum tumor area observed in training set

sliceIndices = zeros(numel(testImagePaths), 1);

detectedMasks = cell(size(testImagePaths));  % Store results for each test image

for i = 1:numel(testImagePaths)
    % Load the test image and ground truth mask
    imageData = niftiread(testImagePaths{i});
    maskData = niftiread(testMaskPaths{i});  % Load the original ground truth mask
    
    % Calculate maxSliceIndex based on the slice with the largest tumor area
    sliceTumorPixelCounts = squeeze(sum(sum(maskData, 1), 2));
    [~, maxSliceIndex] = max(sliceTumorPixelCounts);
    sliceIndices(i) = maxSliceIndex;  % Store max slice index
    
    % Select the test slice and match its histogram to the reference slice
    testSlice = imageData(:, :, maxSliceIndex);
    matchedSlice = imhistmatch(testSlice, referenceSlice);  % Histogram match to reference
    
   % Replace the slice in the image data with the histogram-matched version
    %imageData(:, :, maxSliceIndex) = matchedSlice;
    for j = 1:size(imageData, 3)
     imageData(:, :, j) = imhistmatch(imageData(:, :, j), referenceSlice);
    end
    
    % Call the detection function with test image and learned parameters
    fprintf("Image:%s\n", testImagePaths{i})
    detectedMask = fcm_predict_tumor(testImagePaths{i},imageData,...
        nClusters, tumorIntensityRange, minArea, maxArea,...
        roiCenter, roiRadius, meanRadius, stdRadius, ...
        maxSliceIndex, true);
    
    % Store the detected mask
    detectedMasks{i} = detectedMask;

end

% Evaluate performance metrics for the specific slices
result = evaluateTestSetSlice(testImagePaths, testMaskPaths, detectedMasks, sliceIndices);

% Display results
disp(result);

%%
function detectedTumorMask = fcm_predict_tumor(imagePath, imageData, ...
    nClusters,...
    tumorIntensityRange, minArea, maxArea,...
    roiCenter, roiRadius, meanRadius, stdRadius,...
    maxSliceIndex,...
    useBAT)
    % Load the image
    if isempty(imageData)
        volumeData = niftiread(imagePath);
    else
        volumeData = imageData;
    end
    sliceData = volumeData(:, :, maxSliceIndex);  % Use the slice of interest
    preprocessedSliceData = imadjust(mat2gray(sliceData));
    data = double(preprocessedSliceData(:));

    rng("default")
    % Apply FCM Clustering
    if useBAT
    disp('Using BAT + FCM')
    % Apply FCM Clustering
    options = struct();
    options.NumClusters = nClusters;
    options.ClusterCenters = [];
    options.Exponent = 2;
    options.MaxNumIteration = 10;
    options.DistanceMetric = 'euclidean';
    options.MinImprovement = 1e-5;
    options.Verbose = 1;
    options.ClusterVolume = 1;
    options.alpha = 1;
    options.beta = 0.5;
    options.zeta = 1.5;
    
    % BAT Options.
    options.nBats = 50;
    options.BATIterMax = 10;
    options.lowerBound = min(data);
    options.upperBound = max(data);
    options.fmin = 0;
    options.fmax = 2;
    options.loudness = 0.5;
    options.loudnessCoefficient = .9;
    options.pulseRate = 0.5;
    options.gamma = 0.95;
    options.chaotic = false;
    options.MinNumIteration = 50;
    options.UsePerturbation = false;
    options.PerturbationFactor = 0.01;
    options.minLoudness = 0.01;
    options.globalRestartThreshold = 5;

    % Apply BAT + Fuzzy C-Means (FCM) clustering
    segImgInresults = MFBAFCM(data, options);
    clusterCenters = segImgInresults.centers;
    membership = segImgInresults.U;
    else
     disp('Using FCM alone!')
     options = fcmOptions(NumClusters=nClusters, MaxNumIteration=10);
     [clusterCenters, membership] = fcm(double(preprocessedSliceData(:)), options);
    end 

    % Reshape membership to match image dimensions
    [~, maxMembership] = max(membership, [], 1);
    segmentedSlice = reshape(maxMembership, size(preprocessedSliceData));
    
    % Select the cluster that falls within the tumor intensity range
    tumorClusterIdx = find(clusterCenters >= tumorIntensityRange(1) & clusterCenters <= tumorIntensityRange(2));
    if isempty(tumorClusterIdx)
        warning('No cluster found within the tumor intensity range.');
        detectedTumorMask = false(size(preprocessedSliceData));
        return;
    else
        % Create an initial mask for the detected tumor region
        detectedTumorMask = ismember(segmentedSlice, tumorClusterIdx);
    end
    
    % Apply spatial constraints: Restrict to ROI based on training statistics
    [X, Y] = meshgrid(1:size(preprocessedSliceData, 2), 1:size(preprocessedSliceData, 1));
    distanceFromCenter = sqrt((X - roiCenter(1)).^2 + (Y - roiCenter(2)).^2);
    spatialMask = distanceFromCenter <= roiRadius;
    detectedTumorMask = detectedTumorMask & spatialMask;

    % Filter based on size constraints derived from training data
    detectedTumorMask = bwareafilt(detectedTumorMask, [minArea, maxArea]);

    % Post-processing (e.g., adaptive dilation based on training size statistics)
    tumorStats = regionprops(detectedTumorMask, 'BoundingBox');
    % Use a scaled radius for dilation
    scaledRadius = meanRadius + 0.5 * stdRadius;  % Adjust factor based on desired sensitivity
    % if ~isempty(tumorStats)
    %     boundingBox = tumorStats.BoundingBox;
    %     boundingBoxDiagonal = sqrt(boundingBox(3)^2 + boundingBox(4)^2);
    %     dilationRadius = min(round(boundingBoxDiagonal * 0.1), scaledRadius);  % Adjust scaling factor
    %     detectedTumorMask = imdilate(detectedTumorMask, strel('disk', dilationRadius));
    % end

    dilationRadius = max(round(scaledRadius), 1);  % Ensure minimum dilation radius
    detectedTumorMask = imdilate(detectedTumorMask, strel('disk', dilationRadius));
end
%%
function result = evaluateTestSetSlice(testImagePaths, testMaskPaths, detectedMasks, sliceIndices)
    % Initialize result structure
    result = struct('accuracy', [], 'sensitivity', [], 'specificity', [], ...
        'NPV', [], 'PPV', [], 'AUC', [], 'jaccard', [], 'dice', []);

    % Initialize arrays to store metrics for all test images
    numImages = numel(testImagePaths);
    accuracies = zeros(numImages, 1);
    sensitivities = zeros(numImages, 1);
    specificities = zeros(numImages, 1);
    NPVs = zeros(numImages, 1);
    PPVs = zeros(numImages, 1);
    AUCs = zeros(numImages, 1);
    jaccardCoefficients = zeros(numImages, 1);
    diceCoefficients = zeros(numImages, 1);

    % Iterate through each test case
    for i = 1:numImages
        % Load ground truth mask for the relevant slice
        maskData = logical(niftiread(testMaskPaths{i}));
        maskSlice = maskData(:, :, sliceIndices(i));  % Extract the relevant slice
        
        % Get detected mask for the corresponding slice
        detectedMask = detectedMasks{i};

        % Ensure detectedMask and maskSlice have the same size
        if ~isequal(size(detectedMask), size(maskSlice))
            fprintf("Resizing detected mask for slice %d from %s to %s\n", sliceIndices(i), ...
                mat2str(size(detectedMask)), mat2str(size(maskSlice)));
            detectedMask = imresize(detectedMask, size(maskSlice), 'nearest');
        end

        % Handle cases where no tumor is detected
        if all(detectedMask(:) == 0) && all(maskSlice(:) == 0)
            % No tumor detected, and ground truth is also empty
            continue;  % Skip this slice
        elseif all(detectedMask(:) == 0)
            % No tumor detected but ground truth has a tumor
            sensitivities(i) = 0;
            PPVs(i) = 0;
            % Continue calculating other metrics based on TP, FP, TN, FN
        end

        % Compute confusion matrix elements
        TP = sum(detectedMask(:) & maskSlice(:));  % True Positives
        FP = sum(detectedMask(:) & ~maskSlice(:)); % False Positives
        TN = sum(~detectedMask(:) & ~maskSlice(:)); % True Negatives
        FN = sum(~detectedMask(:) & maskSlice(:));  % False Negatives

        % Compute metrics
        accuracies(i) = (TP + TN) / (TP + FP + TN + FN);
        sensitivities(i) = TP / (TP + FN);  % Sensitivity (Recall)
        specificities(i) = TN / (TN + FP);  % Specificity
        PPVs(i) = TP / (TP + FP);           % Positive Predictive Value (Precision)
        NPVs(i) = TN / (TN + FN);           % Negative Predictive Value
        jaccardCoefficients(i) = TP / (TP + FP + FN);  % Jaccard similarity
        diceCoefficients(i) = 2 * TP / (2 * TP + FP + FN);  % Dice similarity

        % AUC calculation (if applicable)
        if exist('roc_curve', 'file')
            [~, ~, ~, AUC] = perfcurve(maskSlice(:), double(detectedMask(:)), 1);
            AUCs(i) = AUC;
        else
            AUCs(i) = (sensitivities(i) + specificities(i)) / 2;  % Approximate AUC
        end
    end

    % Store metrics in the result structure
    result.accuracy = mean(accuracies);
    result.sensitivity = mean(sensitivities);
    result.specificity = mean(specificities);
    result.PPV = mean(PPVs);
    result.NPV = mean(NPVs);
    result.AUC = mean(AUCs);
    result.jaccard = mean(jaccardCoefficients);
    result.dice = mean(diceCoefficients);

    % Display overall metrics
    fprintf('Evaluation Results (Slice-Based):\n');
    fprintf('Accuracy: %.4f\n', result.accuracy);
    fprintf('Sensitivity: %.4f\n', result.sensitivity);
    fprintf('Specificity: %.4f\n', result.specificity);
    fprintf('PPV: %.4f\n', result.PPV);
    fprintf('NPV: %.4f\n', result.NPV);
    fprintf('AUC: %.4f\n', result.AUC);
    fprintf('Jaccard Similarity: %.4f\n', result.jaccard);
    fprintf('DICE Similarity: %.4f\n', result.dice);
end


%%
function varargout = customFCM(data, options)
    % Fuzzy c-means clustering with optional automatic cluster determination.

    dataSize = size(data, 1);
    maxAutoK = 10;
    objFcn = zeros(options.MaxNumIteration, 1);
    kValues = determineClusterRange(options, dataSize, maxAutoK);
    
    % Initialize info structure for storing results
    info = initializeInfoStruct(numel(kValues));
    centers = options.ClusterCenters;
    minKIndex = Inf;
    lastResults = [];

    % Perform clustering for each k in kValues
    for ct = 1:numel(kValues)
        k = kValues(ct);
        options = adjustOptions(options, k, centers, data);
        fuzzyPartMat = fuzzy.clustering.initfcm(options, dataSize);

        % Iterate to find cluster centers and partition matrix
        [center, objFcn, fuzzyPartMat, covMat] = iterateClustering(data, options, fuzzyPartMat, objFcn);

        % Calculate validity index and update best result if optimal
        validityIndex = fuzzy.clustering.clusterValidityIndex(data, center, fuzzyPartMat);
        info = updateInfo(info, center, fuzzyPartMat, objFcn, covMat, validityIndex, k, ct);
        [minKIndex, lastResults] = updateBestResults(lastResults, center,...
            fuzzyPartMat, objFcn, covMat, validityIndex, k, minKIndex, ct);

        % Update centers for next iteration
        if ~isempty(options.ClusterCenters)
            centers = lastResults.center;
        end
    end

    % Finalize info structure and remove unnecessary fields if Euclidean
    info.OptimalNumClusters = lastResults.k;
    if strcmp(options.DistanceMetric, 'euclidean')
        info = rmfield(info, "CovarianceMatrix");
    end

    % Assign outputs
    [varargout{1:nargout}] = assignOutputs(info, minKIndex);
end

% Helper Functions

function kValues = determineClusterRange(options, dataSize, maxAutoK)
    % Determine range of cluster numbers based on data size and options
    if isequal(options.NumClusters, fuzzy.clustering.FCMOptions.DefaultNumClusters)
        kStart = max(2, size(options.ClusterCenters, 1));
        maxNumClusters = min(kStart + maxAutoK - 1, dataSize - 1);
        kValues = kStart:maxNumClusters;
    else
        kValues = options.NumClusters;
    end
end

function info = initializeInfoStruct(numFolds)
    % Initialize info structure to store results for each k
    info = struct(...
        'NumClusters', zeros(1, numFolds), ...
        'ClusterCenters', cell(1, numFolds), ...
        'FuzzyPartitionMatrix', cell(1, numFolds), ...
        'ObjectiveFcnValue', cell(1, numFolds), ...
        'CovarianceMatrix', cell(1, numFolds), ...
        'ValidityIndex', zeros(1, numFolds), ...
        'OptimalNumClusters', 0);
end

function options = adjustOptions(options, k, centers, data)
    % Adjust options for current cluster count k
    options.ClusterCenters = [];
    options.NumClusters = k;
    options.ClusterVolume = adjustClusterVolume(options.ClusterVolume, k);
    options.ClusterCenters = initializeCenters(centers, k, data);
end

function clusterVolume = adjustClusterVolume(clusterVolume, k)
    % Adjust the cluster volume for each cluster if necessary
    if numel(clusterVolume) ~= k
        clusterVolume = repmat(clusterVolume(1), 1, k);
    end
end

function centers = initializeCenters(centers, k, data)
    % Initialize cluster centers if needed
    if ~isempty(centers) && size(centers, 1) < k
        fprintf("Init. centers using kmeans\n")
        centers = fuzzy.clustering.addKMPPCenters(data, centers, k);
    end
end

function [center, objFcn, fuzzyPartMat, covMat] = iterateClustering(data, options, fuzzyPartMat, objFcn)
    % Perform clustering iterations to optimize centers and partition matrix
    iterationProgressFormat = getString(message('fuzzy:general:lblFcm_iterationProgressFormat'));
    for iterId = 1:options.MaxNumIteration
        [center, fuzzyPartMat, objFcn(iterId), covMat, stepBrkCond, options] = ...
            stepfcm(data,fuzzyPartMat,options);
        brkCond = checkBreakCondition(options,objFcn(iterId:-1:max(1,iterId-1)),iterId,stepBrkCond);
        % Check verbose condition
        if options.Verbose
            fprintf(iterationProgressFormat, iterId, objFcn(iterId));
            if ~isempty(brkCond.description)
                fprintf('%s\n',brkCond.description);
            end
        end

        % Break if early termination condition is true.
        if brkCond.isTrue
            objFcn(iterId+1:end) = [];
            break
        end
    end
end

function info = updateInfo(info, center, fuzzyPartMat, objFcn, covMat, validityIndex, k, ct)
    % Store clustering results in info structure
    info.NumClusters(ct) = k;
    info.ClusterCenters{ct} = center;
    info.FuzzyPartitionMatrix{ct} = fuzzyPartMat;
    info.ObjectiveFcnValue{ct} = objFcn;
    info.CovarianceMatrix{ct} = covMat;
    info.ValidityIndex(ct) = validityIndex;
end

function [minKIndex, lastResults] = updateBestResults(lastResults, center, fuzzyPartMat, objFcn, covMat, validityIndex, k, minKIndex, ct)
    % Update best clustering results based on validity index
    if isempty(lastResults) || validityIndex < lastResults.validityIndex
        lastResults = struct('center', center, 'fuzzyPartMat', fuzzyPartMat, 'objFcn', objFcn, ...
                             'covMat', covMat, 'validityIndex', validityIndex, 'k', k);
        minKIndex = ct;
    end
end

function varargout = assignOutputs(info, minIndex)
    % Assign function outputs
    if nargout > 3, varargout{4} = info; end
    if nargout > 2, varargout{3} = info.ObjectiveFcnValue{minIndex}; end
    if nargout > 1, varargout{2} = info.FuzzyPartitionMatrix{minIndex}; end
    if nargout > 0, varargout{1} = info.ClusterCenters{minIndex}; end
end

function brkCond = checkBreakCondition(options,objFcn,iterId,stepBrkCond)

if stepBrkCond.isTrue
    brkCond = stepBrkCond;
    return
end

brkCond = struct('isTrue',false,'description','');
improvement = diff(objFcn);
if ~isempty(improvement) && (abs(improvement)<=options.MinImprovement || isnan(improvement))
    brkCond.isTrue = true;
    brkCond.description = getString(message('fuzzy:general:msgFcm_minImprovementReached'));
    return
end
if iterId==options.MaxNumIteration
    brkCond.isTrue = true;
    brkCond.description = getString(message('fuzzy:general:msgFcm_maxIterationReached'));
end
end

function [center, newFuzzyPartMat, objFcn, covMat, brkCond,...
    options] = stepfcm(data, fuzzyPartMat, options)
    % One step in fuzzy c-means clustering with a custom objective function.

    % Extract parameters from options
    numCluster = options.NumClusters;
    expo = options.Exponent;
    clusterVolume = options.ClusterVolume;
    brkCond = struct('isTrue', false, 'description', '');

    % Update the fuzzy partition matrix with the exponent
    memFcnMat = fuzzyPartMat .^ expo;

    % Compute or adjust cluster centers
    if isempty(options.ClusterCenters)
        center = (memFcnMat * data) ./ (sum(memFcnMat, 2) * ones(1, size(data, 2)));
        fprintf("Update centers\n");
    else
        fprintf("Use given centers\n");
        center = options.ClusterCenters;
        if options.UsePerturbation
             fprintf("Perturb\n");
            center = center + options.PerturbationFactor * randn(size(center));
        end
        options.ClusterCenters = [];
    end

    % Calculate distances and covariance matrix based on the selected metric
    switch options.DistanceMetric
        case 'mahalanobis'
            [dist, covMat, brkCond] = fuzzy.clustering.mahalanobisdist(center, data, memFcnMat, clusterVolume);
        case 'fmle'
            [dist, covMat, brkCond] = fuzzy.clustering.fmle(center, data, memFcnMat);
        otherwise
            dist = fuzzy.clustering.euclideandist(center, data);
            covMat = [];
    end

    % Calculate the traditional FCM objective function
    %fcmObjective = sum(sum((dist.^2) .* max(memFcnMat, eps)));

    % Calculate the custom fitness value
    %fitnessValue = calculateFitness(center, data, options);

    % Combine the traditional FCM objective and fitness function
    %fprintf("Lambda:%5.2f\n", lambda);
    %objFcn = fcmObjective + lambda * fitnessValue;

    % Calculate custom objective function
    objFcn = calculateFitness(center, data, options);

    % Update the fuzzy partition matrix
    tmp = max(dist, eps) .^ (-2 / (expo - 1));
    newFuzzyPartMat = tmp ./ (ones(numCluster, 1) * sum(tmp));
end



%%
function fitness = calculateFitness(clusterCenters, data, options)
    % Optimized fitness calculation based on intra-cluster distance, SC, PC, and CE

    m = options.Exponent;  % Fuzziness exponent
    alpha = options.alpha;
    beta = options.beta;
    zeta = options.zeta;

    % Compute squared distances between data points and cluster centers
    distances = max(pdist2(data, clusterCenters, 'squaredeuclidean'), 1e-10);

    % Calculate membership matrix U
    U = calculateMembership(distances, m);

    % Compute fitness components
    intraCluster = calculateIntraCluster(data, clusterCenters, U, m);
    SC = calculatePartitionIndex(U, distances, clusterCenters, m);
    PC = calculatePartitionCoefficient(U);
    CE = calculateClassificationEntropy(U);

    % Final fitness value
    fitness = alpha * intraCluster + beta * SC + zeta * (1 / PC + CE);
end

function U = calculateMembership(distances, m)
    % Calculate membership matrix U with fuzziness exponent m
    exponent = 2 / (m - 1);
    invDistances = 1 ./ distances;
    U = (invDistances .^ exponent) ./ sum(invDistances .^ exponent, 2);
end

function intraCluster = calculateIntraCluster(dataPoints, clusterCenters, U, m)
    % Vectorized calculation of weighted intra-cluster distance
    % U: N x c membership matrix, raised to the power m
    distances = pdist2(dataPoints, clusterCenters, 'squaredeuclidean');
    intraCluster = sum(sum((U .^ m) .* distances)) / size(dataPoints, 1);
end

function SC = calculatePartitionIndex(U, distances, clusterCenters, m)
    % Calculate the Partition Index (SC) based on intra- and inter-cluster distances
    % intra-cluster part
    intraClusterDist = sum((U .^ m) .* distances, 1);  % 1 x c vector

    % inter-cluster part
    clusterDistances = max(pdist2(clusterCenters, clusterCenters, 'squaredeuclidean'), 1e-10);
    N = size(U, 1);  % Number of data points
    denominator = N * sum(clusterDistances, 2)';  % 1 x c vector

    SC = sum(intraClusterDist ./ denominator);  % Sum for all clusters
end

function PC = calculatePartitionCoefficient(U)
    % Calculate Partition Coefficient (PC)
    N = size(U, 1);
    PC = sum(U .^ 2, 'all') / N;
end

function CE = calculateClassificationEntropy(U)
    % Calculate Classification Entropy (CE)
    epsilon = 1e-10;  % Small value to avoid log(0)
    N = size(U, 1);
    CE = -sum(U .* log(U + epsilon), 'all') / N;
end

function S = fuzzySeparationIndex(data, centroids, U, m)
    % data: matrix of size [num_samples, num_features]
    % centroids: matrix of size [num_clusters, num_features]
    % U: N x c matrix of membership values for each data point in each cluster
    % m: fuzziness exponent (typically m > 1)
    
    % Number of samples and clusters
    [num_samples, ~] = size(data);
    
    % Calculate the numerator
    % Compute the squared Euclidean distances in a vectorized way
    dist_matrix = pdist2(data, centroids, 'euclidean').^2; % N x c matrix of squared distances
    U_m = U'.^m; % Raise membership values to power m
    numerator = sum(sum(U_m .* dist_matrix)); % Compute the weighted sum of distances
    
    % Calculate the denominator
    % Compute pairwise squared distances between centroids
    centroid_distances = pdist(centroids, 'euclidean').^2; % Vector of pairwise squared distances
    min_inter_centroid_dist = min(centroid_distances); % Find the minimum non-zero distance
    
    % Calculate the separation index
    denominator = num_samples * min_inter_centroid_dist;
    S = numerator / denominator;
end

%
%%
function [bestClusterCenters, bestFitness] = batAlgorithm(data, options)
    % Initialize parameters
    pulseRates = options.pulseRate * ones(options.nBats, 1);  % Per-bat pulse rate
    loudnesses = options.loudness * ones(options.nBats, 1);   % Per-bat loudness
    
    numFeatures = size(data, 2);
    bats = repmat(options.lowerBound, options.nBats, options.NumClusters) + ...
        (repmat(options.upperBound, options.nBats, options.NumClusters) - ...
        repmat(options.lowerBound, options.nBats, options.NumClusters)) ...
        .* rand(options.nBats, options.NumClusters * numFeatures);

    velocities = zeros(options.nBats, options.NumClusters * numFeatures);
    fitness = zeros(options.nBats, 1);

    % Evaluate initial fitness for all bats
    for i = 1:options.nBats
        reshapedBat = reshape(bats(i, :), [options.NumClusters, numFeatures]);
        fitness(i) = calculateFitness(reshapedBat, data, options);
    end

    % Find the initial best solution
    [bestFitness, idx] = min(fitness);
    bestClusterCenters = bats(idx, :);  % Store best solution as flattened vector
    bestSolutions = bestClusterCenters;  % Initialize bestSolutions with the global best

    % Main loop for the BAT algorithm
    noImprovementCounter = 0;  % Counter for global stagnation
    for t = 1:options.BATIterMax
        fprintf("BAT Iter:%d\n", t);

        % Reset stagnation counters at each iteration
        repetitions = zeros(options.nBats, 1);% Counter for stagnation (rep_i)

        for i = 1:options.nBats
            % Update frequency, velocity, and position
            f = options.fmin + (options.fmax - options.fmin) * rand;
            velocities(i, :) = velocities(i, :) + (bats(i, :) - bestClusterCenters) * f;
            newClusterCenters = bats(i, :) + velocities(i, :);

            % Enforce boundary constraints for each cluster center
            newClusterCenters = enforceBoundaries(newClusterCenters, ...
                options.lowerBound, options.upperBound, ...
                options.NumClusters, numFeatures);

            % Local search (small random walk around the best solution)
            if rand > pulseRates(i)
                averageLoudness = mean(loudnesses);
                epsilon = -1 + 2 * rand;  % Random value in [-1, 1]
                newClusterCenters = bestClusterCenters + epsilon * averageLoudness;
            end

            % Evaluate the new solution's fitness
            reshapedNewClusterCenters = reshape(newClusterCenters, [options.NumClusters, numFeatures]);
            newFitness = calculateFitness(reshapedNewClusterCenters, data, options);

            % Acceptance criteria based on fitness, loudness, and pulse rate
            if (newFitness < fitness(i)) && (rand < loudnesses(i))
                fprintf("Update bat %d: fitness, loudness, pulse rate\n", i)
                bats(i, :) = newClusterCenters;
                fitness(i) = newFitness;
                loudnesses(i) = max(options.loudnessCoefficient * loudnesses(i), options.minLoudness);  % Decrease loudness but cap at minLoudness
                pulseRates(i) = pulseRates(i) * (1 - exp(-options.gamma * t));  % Increase pulse rate
            end

            % Update global best if a better solution is found
            if newFitness < bestFitness
                fprintf("Update with best centers and new fitness:%5.3f\n", newFitness)
                bestClusterCenters = newClusterCenters;
                bestFitness = newFitness;
                bestSolutions = [bestSolutions; bestClusterCenters];  % Add new best solution
                if size(bestSolutions, 1) > 5
                    bestSolutions(1, :) = [];  % Keep only the last 5 best solutions
                end
            end

            % Stagnation check (if bat does not move)
            if norm(newClusterCenters - bats(i, :)) == 0
                repetitions(i) = repetitions(i) + 1;
            else
                repetitions(i) = 0;
            end

            % Replace position if stagnation reaches threshold
            if repetitions(i) == 4
                fprintf("Replacing bat %d with average of the best 5 solutions\n", i);
                if ~isempty(bestSolutions) && size(bestSolutions, 1) >= 5
                    bats(i, :) = mean(bestSolutions(end-4:end, :), 1);  % Average of last 5 best solutions
                elseif ~isempty(bestSolutions)
                    bats(i, :) = mean(bestSolutions, 1);  % Average of all stored solutions (if < 5)
                else
                    bats(i, :) = bestClusterCenters;  % Fallback to global best
                end
                repetitions(i) = 0;  % Reset stagnation counter
            end
        end

        % Check for global stagnation across all bats
        if all(repetitions >= 4)
            fprintf("Global stagnation detected. Perturbing all bats.\n");
            % Perturb all bats by randomizing positions slightly around the best solution
            for i = 1:options.nBats
                bats(i, :) = bestClusterCenters + 0.1 * randn(size(bestClusterCenters));
                bats(i, :) = enforceBoundaries(bats(i, :), ...
                    options.lowerBound, options.upperBound, ...
                    options.NumClusters, numFeatures);
            end
            noImprovementCounter = 0;  % Reset the global stagnation counter
        else
            noImprovementCounter = noImprovementCounter + 1;
        end

        % Force restart if no improvement for a prolonged period
        if noImprovementCounter >= options.globalRestartThreshold
            fprintf("Global restart triggered due to prolonged stagnation.\n");
            bats = repmat(options.lowerBound, options.nBats, options.NumClusters) + ...
                (repmat(options.upperBound, options.nBats, options.NumClusters) - ...
                repmat(options.lowerBound, options.nBats, options.NumClusters)) ...
                .* rand(options.nBats, options.NumClusters * numFeatures);
            noImprovementCounter = 0;  % Reset the counter
        end
    end

    % Reshape bestClusterCenters into (numClusters x numFeatures) for output
    bestClusterCenters = reshape(bestClusterCenters, [options.NumClusters, numFeatures]);
end

% Function to enforce boundary constraints for each cluster center
function clusterCenters = enforceBoundaries(clusterCenters,...
    lowerBound, upperBound, numClusters, numFeatures)
    % Reshape to (numClusters x numFeatures) for easy boundary enforcement
    reshapedSolution = reshape(clusterCenters, [numClusters, numFeatures]);
    % Apply boundary constraints for each dimension
    boundedSolution = max(min(reshapedSolution, upperBound), lowerBound);
    % Flatten back to 1D array
    clusterCenters = boundedSolution(:)';
end

function results = MFBAFCM(data, options)
    % Run the Modified Bat Algorithm (MBA) to find initial cluster centers
    [batCenters, ~] = batAlgorithm(data, options);

    options.ClusterCenters = batCenters;
    disp('FCM Options')
    disp(options)
    [centers,U,objFcn,info] = customFCM(data, options);
    results = struct();
    results.U = U;
    results.batCenters = batCenters;
    results.centers = centers;
    results.objFcn = objFcn;
    results.info = info;
end