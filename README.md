Semantic Argument Classification
================================
by the ARG-Group of the seminar "Designing Experiments for Machine Learning Tasks"
Julian Baumann (baumann@cl.uni-heidelberg.de)
Kevin Decker (decker@cl.uni-heidelberg.de)
Maximilian Mueller-Eberstein (eberstein@cl.uni-heidelberg.de)

Version History
	//TODO

Requirements
	Python 3+
	NLTK 3.0.0
		with the PropBank- and full PennTreeBank-corpus installed
	Weka 3.6+

Files
	data/
		contains corpora and ARFF-files with different featuresets
	extfeature/
		contains the libraries and scripts used for feature extraction
	papers/
		contains some of the papers referenced in the project
	presentation/
		contains the files used for the seminar presentation
	status/
		contains a status report from the 17.12.2014

Table of Contents
-----------------
//TODO

Introduction
------------
A supervised machine learning project for the classification of semantic arguments according to PropBank.
The data and tools provided present an overview of this field of research,
include a flexible library for feature extraction and data acquisition for quick integration into Weka,
as well as our results that were achieved with varying algorithms.

Data Setup
----------
The installation guides for the Python language and the NLTK can be found very well documented at their official sources //TODO and //TODO.
To install PropBank and PennTreeBank corpora, the nltk.download() interface should be used.
Call it by importing nltk into a python console and running nltk.download().
Under the 'corpora' tab, download 'propbank' and 'ptb'.
(Should ~10% of the PennTreeBank be sufficient for your experiments, you may also install 'treebank' instead of 'ptb')
Transfer your copy of the PennTreeBank into 'nltk_data/corpora/ptb/'.
Please be aware that all directory and filenames must be in capital letters, as NLTKs CorpusReaders will not find the directories otherwise.
Lastly, the installation guides for Weka can be found at the official source //TODO.
Now, you are good to go!

Feature Extraction
------------------
The libraries and scripts in the 'extfeature' directory can of course be used to prepare data
for semantic argument classification, but can also be helpful with other projects.

acq_data.py
	This script is used to extract features and data from the PropBank corpus and output them in the ARFF.
	For experiment purposes, data can be segmented into any number of files with varying ratios.
	The variables that control this behaviour are found in the init-block and are as follows:
		exp_name
			the experiment's name and @relation
		files
			a list of filenames that the data should be written into
		ratios
			a list of ratios (0.0 - 1.0) that determine the size of the data in the file with the corresponding index
			these should add up to 1
		pbi_ratio
			the ratio (0.0 - 1.0)of total data from PropBank that should be processed
	Example Usage
		>>> python acq_data.py
		
		- acquiring experiment data -

		extracting ARGInstances...done
		(263 ARGInstances extracted from 112 PropBankInstances )

		preparing ARFF...done

		writing ARFF to files...done
		( saved as :
				'SemanticArgumentClassification_data_full.arff'
				'SemanticArgumentClassification_data_train.arff'
				'SemanticArgumentClassification_data_dev.arff'
				'SemanticArgumentClassification_data_test.arff'
		)

		- end of program -
		
argext.py
	This small library contains classes to facilitate the extraction of arguments from the PropBank corpus.
	The ARFFDocument class is not limited to the PropBank corpus and can be used for all kinds of features supported in the ARFF.
	
	ARGInstanceBuilder :
		a helper class that takes PropBankInstances and builds ARGInstances
	
	ARGInstances :
		a class that contains normalized features in string format

	ARFFDocument :
		a class that can easily be converted to an ARFF-file
	
	More detailed documentation of all classes can be called via the standard pydoc implementation.

argext_test.py
	This script tests the argext library by extracting the first 100 PropBankInstances to the file 'test-arff.arff'.
	The features extracted are 'predicate', 'path', 'phraseType', 'position', 'voice' and 'class'.

refguide.arff
	A reference guide for the ARFF files we used in our experiments.
	The guide itself does not contain any data.

Notes on Weka
-------------
The usage guidelines for Weka are far better documented at the official sources found here: //TODO.
For the data acquired in this experiment, we have found that memory intensive algorithms such as J48
require a large amount of RAM. OutOfMemoryExceptions were raised even in a cluster with 10GB of available
memory and trained correctly only after setting the cap to 100GB.

Acknowledgements
----------------
We would like to thank Eva Mudricza-Maydt for her help setting up this project and her informative seminar sessions.
Thank you for your interest and good luck experimenting.