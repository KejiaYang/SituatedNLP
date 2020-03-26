import json
import spacy

# Installation: python -m spacy download en_core_web_sm

def read_data(filename):
	doc = []
	nlp = spacy.load("en")
	with open(filename) as f:
		data = json.load(f)
		for i, img in enumerate(data):
			dic = {}
			dic["img_id"] = img['img_id']
			dic["sentences"] = []
			for e in img['sentences']:
				sen = {}
				sen["subject"] = extract_subject(e, nlp)
				sen["verb"] = extract_verb(e, nlp)
				sen["object"] = extract_object(e, nlp)
				dic["sentences"].append(sen)
			doc.append(dic)
	f.close()

	return doc	
	

def extract_subject(sentence, nlp):
	sub = []
	parsed_text = nlp(sentence)

	#get token dependencies
	for text in parsed_text:
	    #subject would be
		if text.dep_ == "nsubj" or text.dep_ == "nsubjpass" \
			or text.dep_ == "csubj" or text.dep_ == "csubjpass":
			sub.append(text.orth_)

	return sub

def extract_verb(sentence, nlp):
	verb = []
	parsed_text = nlp(sentence)

	#get token dependencies
	for text in parsed_text:
	    #verb would be
		if text.dep_ == "ROOT":
			verb.append(text.orth_)

	return verb

def extract_object(sentence, nlp):
	obj = []
	parsed_text = nlp(sentence)

	#get token dependencies
	for text in parsed_text:
	    #object would be
		if text.dep_ == "pobj" or text.dep_ == "dobj" or text.dep_ == "obj":
			obj.append(text.orth_)

	return obj


def write_data(doc):
	# doc_list = []
	# for k in doc:
	# 	doc_list.append({'img_id': k[0], 'subject': k[1], 'verb': k[2], 'object': k[3]})

	with open('../annotated_data/annotated_frame/val.json', 'w') as output_file:
		json.dump(doc, output_file)

	output_file.close()


def main():
	doc = read_data('../data/annotations/val.json')
	write_data(doc)


if __name__ == '__main__':
	main()