'''
	argext test
	
	This script tests the argext library by extracting the first 100 PropBankInstances to the file 'test-arff.arff'.
	The features extracted are 'predicate', 'path', 'phraseType', 'position', 'voice' and 'class'
	
	requirements :
		Python 3+
		NLTK 3.0+
		argext library
	
	authors :
		Julian Baumann, Kevin Decker and Maximilian Mueller-Eberstein
'''

from argext import *
from nltk.corpus import propbank_ptb

print('\n- testing feature extraction -\n')

pbi = propbank_ptb.instances()
# full feature list
# featurelist = ['predicate', 'path', 'phraseType', 'position', 'voice', 'headword', 'subcategorization', 'class']
# currently supported feature list
featurelist = ['predicate', 'path', 'phraseType', 'position', 'voice', 'class']
# initialize ARGInstanceBuilder with featurelist
arg_the_builder = ARGInstanceBuilder(dict.fromkeys(featurelist))
# arglist for the extracted ARGInstances
arglist = []

print('extracting ARGInstances...', end='')
# test extraction onb the first 10 Propbank Instances
for i in range(100) :
	# add extracted ARGInstances from current Propbank Instance to arglist
	arglist += arg_the_builder.get_arginstances(pbi[i])
print('done')
print(len(arglist), 'ARGInstances extracted \n')

print('writing to ARFF...', end='')
# attributes for ARFFDocument are of course the same as the previously extracted features in featurelist
doc_attributes = dict.fromkeys(featurelist)
# initialize all attributes as lists (since they are all nominal)
for doc_attribute in doc_attributes :
	doc_attributes[doc_attribute] = []
# initialize ARFFDocument with title, dict of attributes and data as all extracted ARGInstances
doc = ARFFDocument('SemanticArgumentClassification', doc_attributes, arglist)
# add all options to ARFFDocument attributes
for arg in arglist :
	for feature in featurelist :
		doc.add_to_attribute(feature, arg.get_feature(feature))
# write ARFFDocument to file
doc.write_to_file('test_arff.arff')
print('done')