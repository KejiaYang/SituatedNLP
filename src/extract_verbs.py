import json
import nltk

def read_data(filename):
	doc = []
	with open(filename) as f:
		data = json.load(f)
		for i, img in enumerate(data):
			doc.append([img['img_id']])
			for e in img['sentences']:
				doc[i].append([extract_verbs(e)])
	f.close()

	return doc	
	

def extract_verbs(sentence):
	ret = []
	tokens = nltk.word_tokenize(sentence)
	tagged = nltk.pos_tag(tokens)
	for tp in tagged:
		if (tp[1][:2] == 'VB'):
			ret.append(tp[0])

	return ret


def write_data(doc):
	doc_list = []

	for k in doc:
		doc_list.append({'img_id': k[0], 'verbs': k[1]})

	with open('../annotated_data/annotated_verbs/test.json', 'w') as output_file:
		json.dump(doc_list, output_file)

	output_file.close()


def main():
	doc = read_data('../data/annotations/test.json')
	write_data(doc)

if __name__ == '__main__':
	main()