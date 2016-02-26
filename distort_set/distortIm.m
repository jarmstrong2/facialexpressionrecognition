function distortIm(filename)
    addpath('lensdistort/lensdistort');

    data = load(filename);
    trainingPts = data.trainingPts;
    testingPts = data.testingPts;
    trainingLabels = data.trainingLabels;
    testingLabels = data.testingLabels;
    
    numPts = size(trainingPts, 3);
    distortedImsNeg = [];
    distortedImsPos = [];
    for i = 1:numPts
        im = trainingPts(:,:,i);
        distortedImsNeg = cat(3,distortedImsNeg,lensdistort(im, -0.15));
        distortedImsPos = cat(3,distortedImsPos,lensdistort(im, 0.15));
    end
    
    trainingPts = cat(3,trainingPts,distortedImsNeg,distortedImsPos);
    trainingLabels = cat(1,trainingLabels,trainingLabels,trainingLabels);
    randIms = randperm(numPts*3);
    trainingPts = trainingPts(:,:,randIms);
    trainingLabels = trainingLabels(randIms,:);
    
    save('distortedset.mat', 'trainingPts', 'trainingLabels', ...
        'testingPts', 'testingLabels');
    
end