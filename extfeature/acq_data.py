'''
	acq_data
	
	This script is designed to acquire experiment data in the ARFF from the PropBank corpus.
	
	The task it is designed to complete, is the extraction of arguments and their features for automatic semantic argument classification.
	It was written as part of the seminar "Designing Experiments for Machine Learning Tasks" by Eva Mujdricza-Maydt at the University of Heidelberg.
	
	
	requirements :
		Python 3+
		NLTK 3.0+
		argext library
		the full Penn Tree Bank subcorpus used in PropBank annotation
	
	authors :
		Julian Baumann, Kevin Decker and Maximilian Mueller-Eberstein
'''

from argext import *
from nltk.corpus import propbank_ptb
from math import floor
from sys import stdout

if __name__ == '__main__' :
	print('\n- acquiring experiment data -\n')
	
	# vars
	exp_name = 'SemanticArgumentClassification'
	files = [exp_name + '_data_train.arff', exp_name + '_data_dev.arff', exp_name + '_data_test.arff']
	ratios = [0.6, 0.2, 0.2]
	pbi_ratio = 1.
	# init
	pbi = propbank_ptb.instances()
	featurelist = ['predicate', 'path', 'phraseType', 'position', 'voice', 'class'] # initialize ARGInstanceBuilder with featurelist
	arg_the_builder = ARGInstanceBuilder(dict.fromkeys(featurelist))
	arglist = [] # arglist for the extracted ARGInstances

	# extract ARGInstances
	pbi_ratio_index = floor(len(pbi)*pbi_ratio)
	for i in range(pbi_ratio_index) :
		if (i%20) == 0 :
			stdout.write("\rextracting ARGInstances...%.2f%%" % (i*100/pbi_ratio_index))
			stdout.flush()
		try :
			arglist += arg_the_builder.get_arginstances(pbi[i]) # add extracted ARGInstances from current Propbank Instance to arglist
		except :
			print("Error at PropBankInstance with index : " + str(i))
	stdout.write("\rextracting ARGInstances...done   \n")
	stdout.flush()
	print('(' + str(len(arglist)) + ' ARGInstances extracted from ' + str(pbi_ratio_index) + ' PropBankInstances ) \n')
	
	# prepare ARFF
	print('preparing ARFF...', end='')
	doc_attributes = dict.fromkeys(featurelist)
	for doc_attribute in doc_attributes :
		doc_attributes[doc_attribute] = []
	doc = ARFFDocument(exp_name, doc_attributes, arglist)
	for arg in arglist :
		for feature in featurelist :
			doc.add_to_attribute(feature, arg.get_feature(feature))
	print('done\n')
	
	# write ARFFs to files
	print('writing ARFF to files...', end='')
	doc.write_to_file(exp_name + '_data_full.arff')
	doc.write_to_ratio_files(files, ratios)
	print('done')
	print('( saved as :')
	print('\t\'' + exp_name + '_data_full.arff' + '\'')
	for file in files :
		print('\t\'' + file + '\'')
	print(')')
	
	print('\n- end of program -')