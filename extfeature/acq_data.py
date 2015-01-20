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
	# init
	pbi = propbank.instances()
	featurelist = ['predicate', 'path', 'phraseType', 'position', 'voice', 'class'] # initialize ARGInstanceBuilder with featurelist
	arg_the_builder = ARGInstanceBuilder(dict.fromkeys(featurelist))
	arglist = [] # arglist for the extracted ARGInstances

	# extract ARGInstances
	cnt_inst = 0.
	for inst in pbi :
		if (cnt_inst%20) == 0 :
			stdout.write("\rextracting ARGInstances...%.2f%%" % (cnt_inst*100/len(pbi)))
			stdout.flush()
		try :
			arglist += arg_the_builder.get_arginstances(inst) # add extracted ARGInstances from current Propbank Instance to arglist
		except Error as err :
			print(err)
		cnt_inst += 1
	stdout.write("\rextracting ARGInstances...done\n")
	stdout.flush()
	print('(', len(arglist), 'ARGInstances extracted ) \n')
	
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
	doc.write_to_ratio_files(files, ratios)
	print('done')
	
	print('\n- end of program -')