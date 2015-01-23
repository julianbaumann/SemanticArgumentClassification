featureList = [...] # zu extrahierende Features
  for pbInstance in pbInstances :
    for pbArg in pbInstance.arguments :
	features = []
	for feature in featureList :
	  featureList.append(extFeature(feature, pbArg, pbInstance))
  # write features to file in ARFF