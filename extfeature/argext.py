'''
	argext library
	
	This small library contains classes to facilitate the extraction of arguments from the PropBank Corpus.
	
	The task it is designed to complete, is the extraction of arguments and their features for automatic semantic argument classification.
	It was written as part of the seminar "Designing Experiments for Machine Learning Tasks" by Eva Mujdricza-Maydt at the University of Heidelberg.
	
	ARGInstanceBuilder :
		takes PropBankInstances and builds ARGInstances
	
	ARGInstances :
		contains normalized features in string format

	ARFFDocument :
		a class that can easily be converted to an ARFF-file
	
	
	requirements :
		Python 3+
		NLTK 3.0+
	
	authors :
		Julian Baumann, Kevin Decker and Maximilian Mueller-Eberstein
'''

from nltk.corpus import propbank_ptb
from nltk.corpus.reader import PropbankTreePointer, PropbankChainTreePointer, PropbankSplitTreePointer
from nltk.tree import ParentedTree
from math import floor
import re

class ARGInstanceBuilder :
	'''
		ARGInstanceBuilder
		
		This class is designed to take NLTK PropBankInstances as input and output ARGInstances as also defined in this library.
		
		A list of features that will be attributed to the ARGInstances can be defined. These will be normalized before attribution.
		
		__init__ :
			constructor
			
		get_arginstances :
			takes a PropBankInstance and returns an ARGInstance according to the feature-list
	'''
	
	def __init__(self, _features = {}) :
		'''
			the ARGInstanceBuilder constructor
			
			parameters :
				_features dict (optional)
					a dict containing the features to be distracted (e.g. { 'feature' : None } )
					default value is an empty dict
		'''
		self.features = _features

	def get_arginstances(self, _pbi) :
		'''
			returns a list of ARGInstances given a PropbankInstance and according to self.features
			
			Each feature is normalized according to the rules in its if-block.
			
			parameters :
				_pbi PropBankInstance
					a PropbankInstance that contains the arguments to be extracted
			return value :
				list of ARGInstances
		'''
		res = []
		for arg in _pbi.arguments : # iterate through all arguments in _pbi
			argfeatures = {}
			
			# predicate feature
			if 'predicate' in self.features :
				argfeatures['predicate'] = re.sub(r'(\w+)\..+', r'\1', _pbi.roleset) # lemmatize the predicate and then set
				# argfeatures['predicate'] = self.wnl.lemmatize(_pbi.predicate.select(_pbi.tree).leaves()[0], "v")
				# argfeatures['predicate'] = _pbi.predicate.select(_pbi.tree).leaves()[0]
			
			# path feature
			if 'path' in self.features :
				senTree = ParentedTree.convert(_pbi.tree)
				argTree = arg[0].select(senTree)
				predTree = _pbi.predicate.select(senTree)
				while argTree.label() == "*CHAIN*" or argTree.label() == "*SPLIT*":
					argTree = argTree[0]
				while predTree.label() == "*CHAIN*" or predTree.label() == "*SPLIT*":					
					predTree = predTree[0]
				
				argParents = []
				predParents = []
				while predTree != None:
					predParents.append(predTree)					
					predTree = predTree.parent()
					
				while argTree!= None:
					argParents.append(argTree)
					argTree = argTree.parent()
					
				jointNode = None
				for node in argParents:
					if node in predParents:
						jointNode = node
							
				stringPath = ""
				for i in range(0, argParents.index(jointNode), 1):	 
					node = argParents[i]
					stringPath += re.sub(r"(\w+)-.+", r"\1", node.label()) + "^"
				
				for i in range(predParents.index(jointNode) , 0, -1):
					node = predParents[i]
					stringPath += re.sub(r"(\w+)-.+", r"\1", node.label()) + "!"
				argfeatures['path'] = stringPath[:-1]
			
			# phraseType feature
			if 'phraseType' in self.features :
				argTree = arg[0].select(_pbi.tree)
				while argTree.label() == "*CHAIN*" or argTree.label() == "*SPLIT*": # traverse tree until a real constituent is found
					argTree = argTree[0]
				argfeatures['phraseType'] = re.sub(r"(\w+)[-=$\|].+", r"\1", argTree.label()) # normalize (e.g. NP-SUBJ -> NP) and set
			
			# position feature
			if 'position' in self.features :
				predTreePointer = _pbi.predicate
				while not type(predTreePointer) is PropbankTreePointer: # traverse tree while the pointer is not a real constituent
					predTreePointer = predTreePointer.pieces[0] 
				pred_wordnum = predTreePointer.wordnum # set predicate wordnumber
				arg_wordnum = None
				if type(arg[0]) is PropbankTreePointer :
					arg_wordnum = arg[0].wordnum
				# PropChainTreePointer and PropSplitTreePointer don't have wordnums and must be traversed
				elif (type(arg[0]) is PropbankChainTreePointer) or (type(arg[0]) is PropbankSplitTreePointer) :
					arg_pieces = arg[0].pieces
					# traverse the tree (always take the left-most subtree) until a PropbankTreePointer is found
					while type(arg_pieces[0]) is not PropbankTreePointer :
						arg_pieces = arg_pieces[0].pieces
					# then get the wordnum
					arg_wordnum = arg_pieces[0].wordnum
				# compare wordnumbers and normalize to 'before' or 'after'
				if arg_wordnum < pred_wordnum :
					argfeatures['position'] = 'before'
				else :
					argfeatures['position'] = 'after'
					
			# voice feature
			if 'voice' in self.features :
				# extract voice from PropBankInstance-inflection and normalize to 'active', 'passive' and 'NONE'
				if _pbi.inflection.voice == 'a' :
					argfeatures['voice'] = 'active'
				elif _pbi.inflection.voice == 'p' :
					argfeatures['voice'] = 'passive'
				else:
					argfeatures['voice'] = 'NONE'
			
			# class feature
			if 'class' in self.features :
				argfeatures['class'] = arg[1].split("-")[0]
				# argfeatures['class'] = re.sub(r'(ARG[0-5])\-\w+', r'\1', arg[1])
			
			res.append(ARGInstance(argfeatures)) # append the initialized ARGInstance to the result
		return res

class ARGInstance :
	'''
		ARGInstance
		
		This class contains normalized features in string format.
		
		The list of features it carries varies and is set upon initialization.
		
		__init__ :
			constructor
			
		__str__ :
			returns a str representation
			
		get_feature :
			returns the value of a feature or None if the feature is not present
	'''
	
	def __init__(self, _features = {}) :
		'''
			the ARGInstance constructor
			
			parameters :
				_features dict (optional)
					a dict containing the features the argument carries  (e.g. { 'feature' : 'value' } )
					default value is an empty dict
		'''
		self.features = _features
	
	def __str__(self) :
		'''
			returns a str representation in the form "feature : value\n"
		'''
		res = ''
		for feature in self.features :
			res += feature + ' : ' + str(self.features[feature]) + '\n'
		return res

	def get_feature(self, _feature) :
		'''
			returns the value of a feature or None if the feature is not present
			
			parameters :
				_feature str :
					a str of the name of the feature to get the value of
			
			return values :
				the value of the feature or None along with a printed warning message if the feature is not present
		'''
		res = None
		if _feature in self.features :
			res = self.features[_feature]
		else :
			print('Warning : no feature "' + _feature + '"')
		return res

class ARFFDocument :
	'''
		ARFFDocument
		
		This class facilitates the output of a file in the ARFF given a dataset in the form of ARGInstances.
		
		__init__ :
			constructor
		
		__str__ :
			returns a str representation in ARFF
		
		get_arff :
			returns a str representation in ARFF according to certain parameters
		
		add_to_attribute :
			add an option to a given attribute
		
		write_to_file :
			write data to an ARFF-file
		
		write_to_ratio_files :
			write data to ARFF-files in different ratios
	'''
	
	def __init__(self, _relation = '', _attributes = {}, _data = []) :
		'''
			the ARFFDocument constructor
			
			parameters :
				_relation str
					the ARFFDocument's title
				_attributes dict (optional)
					the ARFFDocument's attributes as a dict
					keys must be str, values are list for nominal features and str otherwise
					default value is an empty dict
				_data list (optional)
					the ARFFDocument's data as a list of ARGInstances
					default value is an empty list
		'''
		self.relation = _relation
		self.attributes = _attributes
		self.data = _data
	
	def __str__(self) :
		'''
			returns the full ARFFDocument in ARFF as str
		'''
		return self.get_arff() # get_arff() with default parameters returns full data
	
	def get_arff(self, _data_index_lower=0, _data_index_upper=None) :
		'''
			returns a str representation in ARFF according to certain parameters
			
			parameters :
				_data_index_lower int (optional)
					the lower bound index for the data to be included in the result
					default value is 0
				_data_index_upper int (optional)
					the upper bound index for the data to be included in the result
					
			return values :
				a str containing the ARFFDocument in ARFF according to the data-index-bounds
		'''
		res = ''
		# relation
		res += '@relation '
		# if relation contains spaces it must be in quotes (see ARFF specification)
		if ' ' in self.relation :
			res += '"' + self.relation + '"'
		else :
			res += self.relation
		res += '\n\n'
		# attributes
		for attribute in self.attributes :
			res += '@attribute '
			res += attribute + ' '
			# if attribute is nominal
			if type(self.attributes[attribute]) is list :
				res += '{'
				# if option contains spaces it must be in quotes (see ARFF specification)
				for option in self.attributes[attribute] :
					if ' ' in option :
						res += '"' + option + '"'
					else :
						res += option
					res += ','
				res = res[:-1]
				res += '}'
			else :
				res += str(self.attributes[attribute])
			res += '\n'
		res += '\n'
		# data
		res += '@data\n'
		if _data_index_upper is None : # if _data_index_upper is not set, it is set to the rest of the data
			_data_index_upper = len(self.data)
		for i in range(_data_index_lower, _data_index_upper) : # range is defined by _data_index_lower and _data_index_upper
			for attribute in self.attributes :
				# if attribute value contains spaces it must be in quotes (see ARFF specification)
				if ' ' in self.data[i].get_feature(attribute) :
					res += '"' + self.data[i].get_feature(attribute) + '"'
				else :
					res += self.data[i].get_feature(attribute)
				res += ','
			res = res[:-1] + '\n'
		# return
		return res
	
	def add_to_attribute(self, _attribute, _option) :
		'''
			add an option to a given attribute
			
			prints a warning message if attribute is not nominal and therefore cannot be added to or is non-existant
			
			parameters :
				_attribute str
					the name of the attribute to be added to as a str
				_option mixed
					the option to be added to the attribute
		'''
		if _attribute in self.attributes :
			if type(self.attributes[_attribute]) is list :
				# filter duplicates
				if _option not in self.attributes[_attribute] :
					self.attributes[_attribute].append(_option)
			else :
				print('Warning : could not add option since attribute "' + _attribute + '" is not nominal')
		else :
			print('Warning : could not add option since there is no attribute "' + _attribute + '"')
		
	def write_to_file(self, _file) :
		'''
			write all data to an ARFF-file
			
			prints an error message if an exception occurs
			
			parameters :
				_file str
					the filename to write the data into (duplicates will be overwritten)
		'''
		try :
			with open(_file, 'w') as fo :
				fo.write(str(self)) # write str representation of all data
		except :
			print('Error occured while writing ARFFDocument to "' + _file + '"')
	
	def write_to_ratio_files(self, _files, _ratios) :
		'''
			write data to ARFF-files in different ratios
			
			prints an error message if an exception occurs, the number of files and ratios is not equal or the ratios' sum is greater than 1
			
			parameters :
				_files list
					a list of filenames the data should be written into
				_ratios list
					a list of floats that specify the ratio of data to be written into the file at the same index
					these ratios must add up to less than or exactly 1
		'''
		if len(_files) == len(_ratios) : # check for files to ratios inconsistency
			if sum(_ratios) <= 1 : # check for invalid ratio sum
				data_cursor = 0
				for i in range(len(_files)) :
					ratio_index = floor(_ratios[i]*len(self.data))+data_cursor # set current upper data index to ratio length plus the current cursor index in data
					try :
						with open(_files[i], 'w') as fo :
							fo.write(self.get_arff(data_cursor, ratio_index)) # call get_arff with bounds corresponding to ratio and current cursor index in data
					except IOError as err:
						print('Error occured while writing ARFFDocument to "' + _files[i] + '"')
					data_cursor = ratio_index
			else :
				print('Error : ratios\' sum is greater than 1')
		else :
			print('Error : number of files and ratios are not equal')
