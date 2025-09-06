clearvars;

load mri;  % Preloaded data from MATLAB
D = squeeze(D);  % Removes singleton dimensions
filteredD = denoiseImage(D);
%%
figure;
t = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, 'Initial Slice and Denoised Slice', 'FontSize', 24, 'FontWeight', 'bold');
slice_number = 15;
nexttile;
sliceData = D(:,:,slice_number);
imshow(sliceData)
xlabel("Original Slice", 'FontSize', 24, 'FontWeight', 'bold')
nexttile;
sliceData = filteredD(:,:,slice_number);
imshow(sliceData)
xlabel("Denoised Slice", 'FontSize', 24, 'FontWeight', 'bold')
%%
[r,c,n,~] = size(D);
DTrain = reshape(D(:,:,:,1),[r*c n]);
kDim = [3 3];
DTrainFeatures = createMovingWindowFeatures(DTrain,kDim);
[r,c,n,~] = size(filteredD);
filteredDTrain = reshape(filteredD(:,:,:,1),[r*c n]);
kDim = [3 3];
filteredDTrainFeatures = createMovingWindowFeatures(filteredDTrain,kDim);

%%
nClusters = 15;
rng("default");
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

[fcmCenters, fcmU] = customFCM(filteredDTrainFeatures, options);

ecDist = findDistance(fcmCenters,filteredDTrainFeatures);
[~,ecFcmLabel] = min(ecDist',[],2); 
[r,c,n,~] = size(filteredD);
ecFcmLabel = reshape(ecFcmLabel,n,r*c)';
ecFcmLabel = reshape(ecFcmLabel,[r c n]);

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
options.lowerBound = min(filteredDTrainFeatures);
options.upperBound = max(filteredDTrainFeatures);
options.fmin= 0; % Min. frequency
options.fmax = 2; % Max.frequency
options.loudness = 0.5; % Initial loudness
options.loudnessCoefficient = .9;
options.pulseRate = 0.5; % Initial pulse rate
options.gamma = 0.95; % Decay rate for pulse rate
options.chaotic = false;
options.MinNumIteration = 50;
options.UsePerturbation = false;
options.PerturbationFactor = 0.01;

% Apply BAT + Fuzzy C-Means (FCM) clustering
segImgInresults = MFBAFCM(filteredDTrainFeatures, options);

ecDist = findDistance(segImgInresults.centers,filteredDTrainFeatures);
[~,ecBATFcmLabel] = min(ecDist',[],2); 
[r,c,n,~] = size(filteredD);
ecBATFcmLabel = reshape(ecBATFcmLabel,n,r*c)';
ecBATFcmLabel = reshape(ecBATFcmLabel,[r c n]);
%% Show the same slice segmented using FCM alone and BAT+FCM ----
figure;
t = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, 'Denoised Slice, Segmented Slice (FCM in the middle, BAT+FCM on the right)', 'FontSize', 24, 'FontWeight', 'bold');
slice_number = 15;
sliceData = filteredD(:,:,slice_number);
nexttile;
imshow(sliceData)
xlabel("Original Denoised Slice", 'FontSize', 24, 'FontWeight', 'bold');
nexttile;
FCMImg= ecFcmLabel(:,:,slice_number)/nClusters;
imshow(FCMImg)
xlabel("FCM Segmented Slice",'FontSize', 24, 'FontWeight', 'bold');
nexttile;
BATFCMImg = ecBATFcmLabel(:,:,slice_number)/nClusters;
imshow(BATFCMImg)
xlabel("BAT+FCM Segmented Slice",'FontSize', 24, 'FontWeight', 'bold');
%%
fcmUT = fcmU';
PC = calculatePartitionCoefficient(fcmUT);
CE = calculateClassificationEntropy(fcmUT);
distances = max(pdist2(filteredDTrainFeatures, fcmCenters, 'squaredeuclidean'), 1e-10);
SC = calculatePartitionIndex(fcmUT, distances, ...
     fcmCenters, options.Exponent);
S = fuzzySeparationIndex(filteredDTrainFeatures, fcmCenters,...
    fcmU, options.Exponent);
fprintf("FCM: PC:%5.3f-CE:%5.3f-SC:%5.3f-S:%5.3f\n", PC,CE,SC, S);
batFCMUT = segImgInresults.U';
PC = calculatePartitionCoefficient(batFCMUT);
CE = calculateClassificationEntropy(batFCMUT);
distances = max(pdist2(filteredDTrainFeatures, segImgInresults.centers, 'squaredeuclidean'), 1e-10);
SC = calculatePartitionIndex(batFCMUT, distances, ...
     segImgInresults.centers, options.Exponent);
S = fuzzySeparationIndex(filteredDTrainFeatures, segImgInresults.centers,...
    segImgInresults.U, options.Exponent);
fprintf("BAT+FCM PC:%5.3f-CE:%5.3f-SC:%5.3f-S:%5.3f\n", PC,CE,SC, S);

%%
% Define the centers and options for FCM
opt = struct();
opt.nClusters = nClusters;
opt.centerColors = lines(15);  % 15 distinct colors for each center
opt.centerNames = arrayfun(@(x) sprintf('Cluster %d', x), 1:15, 'UniformOutput', false);

% Create figure with tiled layout for two subplots
figure
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'tight');  % 1 row, 2 columns

% Plot FCM Centers
centers = fcmCenters;
sliceData = filteredD(:, :, 15);  % Example slice
nexttile
showCenters(sliceData, centers, opt);
title('FCM Cluster Centers', 'FontSize', 20, 'FontWeight', 'bold');

% Define the centers and options for BAT+FCM
opt.centerNames = arrayfun(@(x) sprintf('Cluster %d', x), 1:15, 'UniformOutput', false);
centers = segImgInresults.centers;

% Plot BAT + FCM Centers
nexttile
showCenters(sliceData, centers, opt);
title('BAT + FCM Cluster Centers', 'FontSize', 20, 'FontWeight', 'bold');

%%
function filteredI = denoiseImage(I)

filteredI = zeros(size(I));  % Initialize an array to store the filtered image
[~, ~, dim3] = size(I); 
 for i = 1:dim3
    fprintf('Slice:%d\n', i)
    slice = double(I(:, :, i));
    %slice = imdiffusefilt(slice);  % Apply  filter
    slice =  medfilt2(slice, [3 3]);
    % Normalize between 0 and 1
    slice = (slice - min(slice(:))) / (max(slice(:)) - min(slice(:)));
    filteredI(:, :, i) = slice;  % Store the filtered slice back
 end
end

% 
function y = createMovingWindowFeatures(in,dim)
% Create feature vectors using a moving window.

rStep = floor(dim(1)/2);
cStep = floor(dim(2)/2);

x1 = [zeros(size(in,1),rStep) in zeros(size(in,1),rStep)];
x = [zeros(cStep,size(x1,2));x1;zeros(cStep,size(x1,2))];

[row,col] = size(x);
yCol = prod(dim);
y = zeros((row-2*rStep)*(col-2*cStep), yCol);
ct = 0;
for rId = rStep+1:row-rStep
    for cId = cStep+1:col-cStep
        ct = ct + 1;
        y(ct,:) = reshape(x(rId-rStep:rId+rStep,cId-cStep:cId+cStep),1,[]);
    end
end
end

% Using padding
% function y = createMovingWindowFeatures(in, dim)
%     % Create feature vectors using a moving window with padding and filtering.
%     % 
%     % Parameters:
%     %   in  - Input matrix
%     %   dim - Dimension of the moving window (e.g., [3, 3] for a 3x3 window)
%     % 
%     % Output:
%     %   y - Output matrix where each row is a feature vector from the moving window
% 
%     % Step 1: Pad the array to handle borders
%     padSize = floor(dim / 2); % Padding size for each dimension
%     paddedIn = padarray(in, padSize, 'symmetric'); % Symmetric padding
% 
%     % Step 2: Extract patches using im2col with sliding windows
%     % 'sliding' mode extracts overlapping windows
%     y = im2col(paddedIn, dim, 'sliding')';
% 
%     % Each row of y now corresponds to a flattened version of each window in the input
% end


% Laplacian
% function y = createMovingWindowFeatures(in, dim)
%     % Define the Laplacian filter
%     laplacianFilter = fspecial('laplacian', 0.2); % 0.2 is a common alpha value for sharpening
% 
%     % Apply the Laplacian filter to the input image
%     laplacianImage = imfilter(in, laplacianFilter, 'symmetric');
% 
%     % Determine the number of rows and columns to step through based on the window size
%     rStep = floor(dim(1)/2);
%     cStep = floor(dim(2)/2);
% 
%     % Pad the filtered image
%     x1 = padarray(laplacianImage, [0, rStep], 'both');
%     x = padarray(x1, [cStep, 0], 'both');
% 
%     % Initialize output array
%     [row, col] = size(x);
%     yCol = prod(dim); % Number of elements in each window
%     y = zeros((row - 2 * rStep) * (col - 2 * cStep), yCol); % Preallocate the output matrix
% 
%     % Extract features using a moving window
%     ct = 0;
%     for rId = rStep + 1 : row - rStep
%         for cId = cStep + 1 : col - cStep
%             ct = ct + 1;
%             % Reshape the window into a row vector and store in output
%             y(ct, :) = reshape(x(rId - rStep : rId + rStep, cId - cStep : cId + cStep), 1, []);
%         end
%     end
% end
% 
% % Prewitt
% function y = createMovingWindowFeatures(in, dim)
%     % Define the Prewitt filters
%     prewittFilterX = fspecial('prewitt');
%     prewittFilterY = prewittFilterX';
% 
%     % Apply the Prewitt filters to get horizontal and vertical edges
%     prewittImageX = imfilter(double(in), prewittFilterX, 'symmetric');
%     prewittImageY = imfilter(double(in), prewittFilterY, 'symmetric');
% 
%     % Compute the gradient magnitude
%     gradientMagnitude = sqrt(prewittImageX.^2 + prewittImageY.^2);
% 
%     % Pad both the original and gradient images
%     paddedOriginal = padarray(in, floor(dim / 2), 'symmetric');
%     paddedGradient = padarray(gradientMagnitude, floor(dim / 2), 'symmetric');
% 
%     % Use im2col to extract sliding windows and concatenate
%     originalWindows = im2col(double(paddedOriginal), dim, 'sliding')';
%     gradientWindows = im2col(paddedGradient, dim, 'sliding')';
% 
%     % Concatenate the original and Prewitt-based features for each window
%     y = [originalWindows, gradientWindows];
% end


%Entropy
% function y = createMovingWindowFeatures(in, dim)
%     % Pad the input image
%     paddedIn = padarray(in, floor(dim/2), 'symmetric');
% 
%     % Mean feature
%     meanFilter = fspecial('average', dim);
%     meanFeature = imfilter(double(paddedIn), meanFilter, 'symmetric');
% 
%     % Standard deviation feature
%     localMeanSquare = imfilter(double(paddedIn).^2, meanFilter, 'symmetric');
%     localVariance = localMeanSquare - meanFeature.^2;
%     localVariance(localVariance < 0) = 0; % Ensure non-negative variance
%     stdDevFeature = sqrt(localVariance);
% 
%     % Entropy feature
%     entropyFeature = entropyfilt(paddedIn, true(dim));
% 
%     % Reshape each feature map into columns and concatenate
%     y = [im2col(meanFeature, dim, 'sliding')';
%          im2col(stdDevFeature, dim, 'sliding')';
%          im2col(entropyFeature, dim, 'sliding')']';
%     y = y';
% end



%% Calculate feature distance from cluster center.

function dist = findDistance(centers,data)

dist = zeros(size(centers, 1), size(data, 1));
for k = 1:size(centers, 1)
    dist(k, :) = sqrt(sum(((data-ones(size(data, 1), 1)*centers(k, :)).^2), 2));
end
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
    repetitions = zeros(options.nBats, 1);  % Counter for stagnation (rep_i)
    bestSolutions = [];  % To store the best solutions

    numFeatures = size(data, 2);
    % Initialize bat positions with bounds expanded to (nBats x (numClusters * numFeatures))
    bats = repmat(options.lowerBound, options.nBats, options.NumClusters) + ...
       (repmat(options.upperBound, options.nBats, options.NumClusters) - ...
       repmat(options.lowerBound, options.nBats, options.NumClusters)) ...
       .* rand(options.nBats, options.NumClusters * numFeatures);

    velocities = zeros(options.nBats, options.NumClusters * numFeatures);
    fitness = zeros(options.nBats, 1);

    % Evaluate initial fitness for all bats using the dataset
    for i = 1:options.nBats
        % Reshape each bat into a (numClusters x calculateFitness) format for calculateFitness
        reshapedBat = reshape(bats(i, :), [options.NumClusters, numFeatures]);
        fitness(i) = calculateFitness(reshapedBat, data, options);  % Calculate fitness based on clustering
    end

    % Find the initial best solution
    [bestFitness, idx] = min(fitness);
    bestClusterCenters = bats(idx, :);  % Store best solution as flattened vector

    % Main loop for the BAT algorithm
    for t = 1:options.BATIterMax
        fprintf("BAT Iter:%d\n", t);
        for i = 1:options.nBats
            % Update frequency, velocity, and position
            f = options.fmin+ (options.fmax - options.fmin) * rand;
            velocities(i, :) = velocities(i, :) + (bats(i, :) - bestClusterCenters) * f;
            newClusterCenters = bats(i, :) + velocities(i, :);

            % Enforce boundary constraints for each cluster center
            newClusterCenters = enforceBoundaries(newClusterCenters,...
                options.lowerBound, options.upperBound,...
                options.NumClusters, numFeatures);

            % Local search (small random walk around the best solution)
            if rand > pulseRates(i)
                % Calculate the average loudness of all bats
                averageLoudness = mean(loudnesses);
                
                % Generate epsilon as a random number in the range [-1, 1]
                epsilon = -1 + 2 * rand;
                
                % Update newClusterCenters based on the average loudness and epsilon
                newClusterCenters = bestClusterCenters + epsilon * averageLoudness;

            end

            % Evaluate the new solution's fitness with the dataset
            reshapedNewClusterCenters = reshape(newClusterCenters, [options.NumClusters, numFeatures]);
            newFitness = calculateFitness(reshapedNewClusterCenters, data, options);

            % Acceptance criteria based on fitness, loudness, and pulse rate
            if (newFitness < fitness(i)) && (rand < loudnesses(i))
                fprintf("Update bat %d: fitness, loudness, pulse rate\n", i)
                bats(i, :) = newClusterCenters;
                fitness(i) = newFitness;
                loudnesses(i) = options.loudnessCoefficient * loudnesses(i);  % Decrease loudness
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

            % Replace Xi if rep_i reaches 4
            if repetitions(i) == 4
                fprintf("Replacing bat %d with average of the best 5 solutions\n", i);
                if size(bestSolutions, 1) >= 5
                    bats(i, :) = mean(bestSolutions(end-4:end, :), 1);  % Average of last 5 best solutions
                else
                    bats(i, :) = mean(bestSolutions, 1);  % Average of all stored solutions (if < 5)
                end
                repetitions(i) = 0;  % Reset stagnation counter
            end            
        end
    end

    % Reshape bestSol into (numClusters x 9) for output
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

%%
function showCenters(Img, centers, opt)
    % Show the original image
    imshow(Img, []);
    hold on;

    % Set default values for optional parameters if not provided
    if ~isfield(opt, 'nClusters')
        opt.nClusters = size(centers, 1);  % Set based on the number of centers
    end

    % Ensure fixedColors has enough colors for all clusters
    if ~isfield(opt, 'centerColors') || size(opt.centerColors, 1) < opt.nClusters
        % Generate a colormap if insufficient colors are provided
        opt.centerColors = lines(opt.nClusters);  % Default to distinct colors
    end

    % Ensure center names are provided for each cluster
    if ~isfield(opt, 'centerNames')
        opt.centerNames = arrayfun(@(x) sprintf('Cluster %d', x), 1:opt.nClusters, 'UniformOutput', false);
    end

    % Extract 3x3 neighborhoods from the image as feature vectors
    [nRows, nCols] = size(Img);
    patchDim = sqrt(size(centers, 2));  % Assuming a 3x3 window for each center (9 elements)
    halfPatch = floor(patchDim / 2);

    % Pad the image to extract neighborhoods at the borders
    paddedImg = padarray(Img, [halfPatch, halfPatch], 'symmetric');
    featureVectors = im2col(paddedImg, [patchDim, patchDim], 'sliding')';

    % Match each center to the closest neighborhood in the image
    for i = 1:opt.nClusters
        % Get the i-th center feature vector
        centerVector = centers(i, :);

        % Compute Euclidean distances from center to all neighborhoods
        dists = sqrt(sum((featureVectors - centerVector).^2, 2));

        % Find the index of the closest neighborhood
        [~, closestIdx] = min(dists);

        % Convert linear index back to row, col in the original image
        [row, col] = ind2sub([nRows, nCols], closestIdx);

        % Plot the center with specified color and label
        plot(col, row, 'x', 'Color', opt.centerColors(i, :), 'MarkerSize', 18, 'LineWidth', 4, ...
            'DisplayName', opt.centerNames{i});
    end

    % Show legend with center names
    legend('show', 'Location', 'bestoutside');
    hold off;
end
