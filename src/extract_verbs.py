def read_data(filename):
	import json
	doc = []
	img_count = 0
	anno_count = 0
	with open(filename) as f:
		data = json.load(f)
		for i, img in enumerate(data):
			img_count += 1
			doc.append([img['img_id']])
			for e in img['sentences']:
				anno_count += 1
				doc[i].append([extract_verbs(e)])
			# if (i == 3):
			# 	break
	f.close()	

	# print (img_count)
	# print (anno_count)
	# print (doc)
	

def extract_verbs(sentence):
	import nltk
	ret = []
	tokens = nltk.word_tokenize(sentence)
	tagged = nltk.pos_tag(tokens)
	for tp in tagged:
		if (tp[1][:2] == 'VB'):
			ret.append(tp[0])

	return ret

def main():
	read_data('../data/annotations/val.json')

if __name__ == '__main__':
	main()