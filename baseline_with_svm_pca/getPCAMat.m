function [coeffs, meanPCA] = getPCAMat(images)
    examples = size(images, 3);
    newImages = reshape(images,32*32,examples);
    newImages = transpose(newImages);
    [coeffs,b,c] = svd(cov(double(newImages)));
    meanPCA = mean(newImages);
end