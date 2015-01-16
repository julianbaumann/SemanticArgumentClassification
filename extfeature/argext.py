'''
	argext library
	
	ARGInstanceBuilder :
		takes PropBankInstances and builds ARGInstances
	
	ARGInstances
		contains features in string format

	ARFFDocument
		a class that can easily be converted to an ARFF-file
'''

from nltk.corpus import propbank
from nltk.corpus.reader import PropbankTreePointer, PropbankChainTreePointer, PropbankSplitTreePointer
import re

class ARGInstanceBuilder :
	def __init__(self, _features = {}) :
		self.features = _features

	def get_arginstances(self, _pbi) :
		'''
			returns a list of ARGInstances given a PropbankInstance and according to self.features
			
			parameters:
				_pbi
					a PropbankInstance that contains the arguments to be extracted
			returns:
				list of ARGInstances
		'''
		res = []
		for arg in _pbi.arguments :
			argfeatures = {}
			if 'predicate' in self.features :
				argfeatures['predicate'] = _pbi.predicate.select(_pbi.tree).leaves()[0]
			if 'path' in self.features :
				# normalize paths by removing unnecessary newlines and whitespaces
				argfeatures['path'] = str(arg[0].select(_pbi.tree)).replace('\n','')
				argfeatures['path'] = re.sub(r'\s*?(\(+\*?[A-Z0-9-,]+\*?\s)+\s*', r'\1', argfeatures['path'])
				# ISSUE : returns all trees if root is CHAIN or SPLIT
				# ISSUE : should terminal nodes be kept or only the constituents?
				# -> ParentedTrees are the answer (start at ARG and PRED and call parent() until they match)
			if 'phraseType' in self.features :
				argfeatures['phraseType'] = arg[0].select(_pbi.tree).label()
				# ISSUE : returns CHAIN or SPLIT if it's the root
			if 'position' in self.features :
				pred_wordnum = _pbi.predicate.wordnum
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
				if arg_wordnum < pred_wordnum :
					argfeatures['position'] = 'before'
				else :
					argfeatures['position'] = 'after'
			if 'voice' in self.features :
				if _pbi.inflection.voice == 'a' :
					argfeatures['voice'] = 'active'
				elif _pbi.inflection.voice == 'p' :
					argfeatures['voice'] = 'passive'
			if 'class' in self.features :
				argfeatures['class'] = arg[1]
				argfeatures['class'] = re.sub(r'(ARG[0-5])\-\w+', r'\1', arg[1])
			# if 'headword' in features :
				# headword
				# not sure if annotated in PropBank
			# if 'subcategorization' in features :
				# the parent node of the predicate (e.g. [VP] -> VBD -> 'rose')
				# is this a necessary feature?
			res.append(ARGInstance(argfeatures))
		return res

class ARGInstance :
	def __init__(self, _features = {}) :
		self.features = _features
	
	def __str__(self) :
		res = ''
		for feature in self.features :
			res += feature + ' : ' + str(self.features[feature]) + '\n'
		return res

	def get_feature(self, _feature) :
		res = None
		if _feature in self.features :
			res = self.features[_feature]
		else :
			print('Warning : no feature "' + _feature + '"')
		return res

class ARFFDocument :
	def __init__(self, _relation = '', _attributes = {}, _data = []) :
		'''
			constructor of ARFFDocument
			
			parameters :
				_relation str
					the ARFFDocument's title
				_attributes dict
					the ARFFDocument's attributes as a dict
					keys must be str, values are list for nominal features and str otherwise
				_data list
					the ARFFDocument's data as a list of ARGInstances
		'''
		self.relation = _relation
		self.attributes = _attributes
		self.data = _data
	
	def __str__(self) :
		'''
			returns the ARFFDocument in ARFF as str
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
		for instance in self.data :
			for attribute in self.attributes :
				# if attribute value contains spaces it must be in quotes (see ARFF specification)
				if ' ' in instance.get_feature(attribute) :
					res += '"' + instance.get_feature(attribute) + '"'
				else :
					res += instance.get_feature(attribute)
				res += ','
			res = res[:-1] + '\n'
		# return
		return res
	
	def add_to_attribute(self, _attribute, _option) :
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
		try :
			with open(_file, 'w') as fo :
				fo.write(str(self))
		except :
			print('Error occured while writing ARFFDocument to "' + _file + '"')