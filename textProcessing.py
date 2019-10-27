import nltk
import gensim
import spacy
# nltk.download()

def preprocess_text(corpus):
	combined = ""
	for entry in corpus:
		combined += entry + " "
	stopwords = list(nltk.corpus.stopwords.words('english'))
	stopwords.extend(["I", "we", "my", "they", "he", "she", "his", "her", "us", "you", "your", "them", "this", "these"])
	lemmatizer = nltk.stem.WordNetLemmatizer()
	# tokens = list(filter(lambda word: word.isalnum(), nltk.word_tokenize(corpus)))
	sents = nltk.tokenize.sent_tokenize(combined)
	print(sents)
	words = []
	for sent in sents:
		tokens = list(filter(lambda word: word.isalnum(), nltk.word_tokenize(sent)))
		processed_text = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stopwords]
		words.append(processed_text)
	# words = [list(filter(lambda word: word.isalnum(), nltk.word_tokenize(sent))) for sent in sents]
	# lemmatizer = nltk.stem.WordNetLemmatizer()
	# processed_text = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords]
	return words, sents

	# # Alternate version to include bigrams
	# def lemmatization(text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
	#     doc = nlp(" ".join(text)) 
	#     return [token.lemma_ for token in doc if token.pos_ in allowed_postags]

	# nlp = spacy.load('en', disable=['parser', 'ner'])
	# bigram = gensim.models.Phrases(data_words, min_count=3, threshold=75)
	# bigram_mod = gensim.models.phrases.Phraser(bigram)


def create_topic_model(words):
	id2word = gensim.corpora.Dictionary(words)
	corpus = [id2word.doc2bow(word) for word in words]
	lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=5, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True,
                                           minimum_probability = 0.25,
                                           minimum_phi_value = 0.5)
	return lda_model, id2word


def extract_keywords(lda_model, id2word):
	keywords = []
	topics = lda_model.get_topics()
	for topic in topics:
		paired = zip(topic, list(id2word))
		maxTerm = sorted(paired, key=lambda x: x[0], reverse=True)[:5]
		for term in maxTerm:
			keywords.append(id2word[term[1]])
	return keywords


def find_keyword_contexts(sentences, keywords):
	check = []
	for sentence in sentences:
		for keyword in keywords:
			if keyword in sentence:
				check.append(sentence)
				break
	return check

# Test from Ben's image processing text
# corpus = open("output.txt", "r").read()
# print(corpus)
# processed = preprocess_text(corpus)
# print(processed)
# lda = create_topic_model(processed)
# print(lda.print_topics())

data1 = [
	"These are great but not much better then gen1. Only addition is Siri feature. I will rather buy the previous model on discount and Save some green.",
	"Everyone is posting that there isn’t a difference between these and the 1st gen. This is misleading and inaccurate. Is the improvement drastic, no, but it is still an improvement. The improvement is that Apple has upgraded the on-board chip to the H1, which leads to faster and more stable pairing. This isn’t anecdotal. It’s been tested and proven to be faster. Also, if you opt for wireless charging, buy the case with the gen 2 AirPods and you’ll save a few dollars. If you already have the 1st gen, then it’s probably not worth the upgrade. If you are looking to buy your first pair of AirPods, then go for these.",
	"These AirPods are amazing they automatically play audio as soon as you put them in your ears and pause when you take them out. A simple double-tap during music listening will skip forward. To adjust the volume, change the song, make a call, or even get directions, just say 'Hey Siri' to activate your favorite personal assistant. Plus, when you're on a call or talking to Siri, an additional accelerometer works with dual beamforming microphones to filter out background noise and ensure that your voice is transmitted with clarity and consistency. Additionally, they deliver five hours of listening time on a single charge, and they're made to keep up with you thanks to a charging case that holds multiple additional charges for more than 24 hours of listening time. Just 15 minutes in the case gives you three hours of listening to time or up to two hours of talk time. I would highly recommend it to anyone looking to buy",
	"Poor quality microphone. Not suitable for a remote worker taking calls. If your job requires dictation or a high quality mic, go elsewhere",
	"We bought a brand new set of AirPods for $159. After using them for a week, I was listening to them and the right air pod went dead. I figured, maybe I just need to charge them. So when I got home I charged them, and the next day I went to use them again and the right air bud was still dead. So I did some research, tried to reset them, and then I couldn't reconnect them to my phone at all. Once I contacted Apple, they tried to give us the run around. They wanted us to let them fully die, which makes sense, so we did that. Then, when they still didn't work we wanted to exchange the faulty pair for a new set (BECAUSE WE PAID 159 DOLLARS AND ONLY USED THEM FOR A WEEK). Keep in mind, they were taken care of, not dropped, no water damage, they were expensive so they were treated delicately. Apple wanted a $180 deposit to get a new set, which is MORE THAN WHAT I ORIGINALLY PAID FOR. I decided to contact Amazon, and they suggested contacting Apple, but once I explained what was going, on Amazon offered to make things right and send out a replacement. Never had an issue with Amazon's customer service, but Apple was extremely disappointing. I won't purchase these again."
]

data2 = ["I bought this laptop on February 2019, it worked fine for one month then it started giving trouble. As soon as I open laptop screen flickers. once close and then open works fine. I wanted to return this laptop but it already passed 30 days (I am a elite member). I called apple support they did some diagnostics and ran some tests and said every thing is fine. Again after few day saw the same problem. Called Apple support again, they ask me to take this to local store. I took it to Troy Somerst store. technician completely serviced and said he fixed the problem. after a week again I have got the same issue. This time i went to local apple store ask for the replacement. That technician was so rude ,he said he can service it but can't replace. the worst service worst laptop. Never ever spend too much of money on this carp. Please don't recommend this product to your friends and family."]


processed1, sent1 = preprocess_text(data1)
lda1, id2word1 = create_topic_model(processed1)
# print(lda1.print_topics())
keywords1 = extract_keywords(lda1, id2word1)
# print(keywords1)
sentences1 = find_keyword_contexts(sent1, keywords1)
print(sentences1)

processed2, sent2 = preprocess_text(data2)
lda2, id2word2 = create_topic_model(processed2)
# print(lda2.print_topics())
keywords2 = extract_keywords(lda2, id2word2)
# print(keywords2)
sentences2 = find_keyword_contexts(sent1, keywords1)
print(sentences2)





