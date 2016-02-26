function decompositions = decomposesvm(PCAcoeffs, compCount, images)
    % coeeficients to keep
    coeefs = PCAcoeffs(:, 1:compCount);
    decompositions = double(images) * coeefs;
end