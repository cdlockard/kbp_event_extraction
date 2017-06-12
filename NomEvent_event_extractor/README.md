Dependencies:
Stanford CoreNLP 3.6 (https://stanfordnlp.github.io/CoreNLP/)
CRF Suite (https://github.com/chokkan/crfsuite)

Python packages:
pycorenlp
nltk
gensim
scipy
sklearn
numpy
python-crfsuite


---------------------------------------------------------


To preprocess TAC-KBP 2016 Event Argument Linking data run:
python readEAL2016.py
with a Stanford CoreNLP server running on port 9000.
Note that this file must be modified with to include the correct filepath to the raw data on line 82.


Then, to extract events and arguments, run:
python process_eal_competition.py
Correct filepaths will need to be provided on lines 19, 34, and 78.


___________________________________________________


To preprocess TAC-KBP 2016 Event Nuggets data run:
python readNuggets2016.py
with a Stanford CoreNLP server running on port 9000.
Note that this file must be modified with to include the correct filepath to the raw data on line 88.

Then, to extract events and arguments, run:
python process_eal_nuggets.py
Correct filepaths will need to be provided on lines 20, 23, and 29.