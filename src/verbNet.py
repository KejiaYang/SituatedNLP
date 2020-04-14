import pickle
import re

import nltk
from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import verbnet
# print (verbnet.classids('walk'))
# print (verbnet.classids('move'))
# print(verbnet.classids('be'))
# print (verbnet.classids('jog'))
# len(verbnet.lemmas())
# len(verbnet.wordnetids())
# len(verbnet.classids())

def read_data(filename):
	pickle_in = open(filename, "rb")
	data = pickle.load(pickle_in)
	# print(data[0])

	# annotation list
	# print(data[0][2])

	# prediction
	# print(data[0][3])

	# performance scores
	# print(data[0][4-7])

	return data


# [[[verb, [attribute_lists], #of_ambiguity], [], ...]]
def get_verNet_attributes(data):

	verbNet = []
	# i = 0;
	for ele in data:
	    tagged = nltk.pos_tag(ele[3])
	    verb_lists = []
	    for idx, tp in enumerate(tagged):
	        if (tp[1][:2] == 'VB'):
	            base_form = WordNetLemmatizer().lemmatize(ele[3][idx], 'v')
	            if (verbnet.classids(WordNetLemmatizer().lemmatize(ele[3][idx], 'v')) != []):
	                attr = verbnet.classids(base_form)
	                at_list = []
	                for at in attr:
	                    splitted_at = []
	                    splitted_string = at.split('-')
	                    splitted_at.append(splitted_string[0])
	                    splitted_at.append(splitted_string[1].split('.')[0])
	                    at_list.append([])
	                    at_list[-1] = splitted_at
	                verb_lists.append([base_form, at_list, len(attr)])
	    verbNet.append(verb_lists)

	#     i += 1
	#     if (i == 5):
	#         break
	        
	# print (verbNet)
	return verbNet


def write_data(res):
	# with open('verbNet.txt', 'w') as output_file:
	# 	for l in res:
	# 		output_file.write('%s\n' % l)

	# output_file.close()
	output = open('verbNet.pkl', 'wb')
	pickle.dump(res, output)
	output.close()
	

	pickle_in = open('verbNet.pkl', "rb")
	data = pickle.load(pickle_in)
	print(data)





def main():
	doc = read_data('model_prediction.pkl')
	vn = get_verNet_attributes(doc)
	write_data(vn)


if __name__ == '__main__':
	main()

