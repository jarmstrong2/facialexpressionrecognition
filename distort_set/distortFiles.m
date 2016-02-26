function [returnIms, returnTar] = distortFiles(images, targets)
    returnIms = images;
    returnTar = targets;
    for i=1:7
        targetInd = find(targets == i);
        imsToDistort = images(:,:,targetInd);
        imsDistorted = distortIms(imsToDistort);
        imsTarget = ones(size(targetInd,1),1) * i;
        returnIms = cat(3, returnIms, imsDistorted);
        returnTar = cat(1, returnTar, imsTarget);
    end
end

function newIms = distortIms(images)
   newIms = images;
   numpts = size(images, 3);
   randPerms_1 = randperm(numpts);
   randPerms_2 = randperm(numpts);
   newIms(1:18,:,randPerms_1) = images(1:18,:,randPerms_2);
end