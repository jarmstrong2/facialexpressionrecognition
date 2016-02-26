function getBatch (batchSize, dataPts, currentCount)
	returnData = nil
	returnTarget = nil

	dataSize = dataPts:size(1)
	for i=1,batchSize do
		if currentCount > dataSize then
			currentCount = 1
		end

		if returnData then
			returnData = torch.cat(returnData, dataPts[{{currentCount},{},{},{}}], 1)
		else
			returnData = dataPts[{{currentCount},{},{},{}}]
		end

		currentCount = currentCount + 1
	end

	return returnData, currentCount
end