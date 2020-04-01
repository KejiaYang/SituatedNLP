import json
import spacy

# Installation: python -m spacy download en_core_web_sm

def read_data(filename):
	doc = []
	count = {}
	count["num_sentences"] = {}
	count["num_subjects"] = {}
	count["num_verbs"] = {}
	count["num_objects"] = {}
	count["subjects"] = {}
	nlp = spacy.load("en")
	with open(filename) as f:
		data = json.load(f)
		for i, img in enumerate(data):
			num_sub = 0
			num_v = 0
			num_obj = 0
			dic = {}
			dic["img_id"] = img['img_id']
			dic["sentences"] = []
			for e in img['sentences']:
				sen = {}
				sen["subject"] = extract_subject(e, nlp)
				sen["verb"] = extract_verb(e, nlp)
				sen["object"] = extract_object(e, nlp)
				dic["sentences"].append(sen)
				if sen["subject"]:
					num_sub = num_sub + len(sen["subject"])
				if sen["verb"]:
					num_v = num_v + len(sen["verb"])
				if sen["object"]:
					num_obj = num_obj + len(sen["object"])
			doc.append(dic)
			if len(dic["sentences"]) not in count["num_sentences"]:
				count["num_sentences"][len(dic["sentences"])] = 1
			else:
				count["num_sentences"][len(dic["sentences"])] = count["num_sentences"][len(dic["sentences"])] + 1
			if num_sub not in count["num_subjects"]:
				count["num_subjects"][num_sub] = 1
			else:
				count["num_subjects"][num_sub] = count["num_subjects"][num_sub] + 1
			if num_v not in count["num_verbs"]:
				count["num_verbs"][num_v] = 1
			else:
				count["num_verbs"][num_v] = count["num_verbs"][num_v] + 1
			if num_obj not in count["num_objects"]:
				count["num_objects"][num_obj] = 1
			else:
				count["num_objects"][num_obj] = count["num_objects"][num_obj] + 1
	f.close()

	return doc, count
	

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

	with open('../annotated_data/annotated_frame/test.json', 'w') as output_file:
		json.dump(doc, output_file)

	output_file.close()

def evaluate(count):
	with open('../annotated_data/annotated_frame/test_evaluate.txt', 'w') as output_file:
		output_file.write("Number of sentences per image pair:\n")
		for line in sorted(count["num_sentences"].keys()):
			output_file.write(str(line))
			output_file.write(": ")
			output_file.write(str(count["num_sentences"][line]))
			output_file.write("\n")
		output_file.write("Number of subjects per image pair:\n")
		for line in sorted(count["num_subjects"].keys()):
			output_file.write(str(line))
			output_file.write(": ")
			output_file.write(str(count["num_subjects"][line]))
			output_file.write("\n")
		output_file.write("Number of verbs per image pair:\n")
		for line in sorted(count["num_verbs"].keys()):
			output_file.write(str(line))
			output_file.write(": ")
			output_file.write(str(count["num_verbs"][line]))
			output_file.write("\n")
		output_file.write("Number of objects per image pair:\n")
		for line in sorted(count["num_objects"].keys()):
			output_file.write(str(line))
			output_file.write(": ")
			output_file.write(str(count["num_objects"][line]))
			output_file.write("\n")

	output_file.close()


def main():
	doc, count = read_data('../data/annotations/test.json')
	write_data(doc)
	evaluate(count)


if __name__ == '__main__':
	main()