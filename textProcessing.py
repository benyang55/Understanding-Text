import nltk
import gensim
import spacy
# nltk.download()

def preprocess_text(corpus):
	stopwords = list(nltk.corpus.stopwords.words('english'))
	lemmatizer = nltk.stem.WordNetLemmatizer()
	tokens = list(filter(lambda word: word.isalnum(), nltk.word_tokenize(corpus)))
	sents = nltk.tokenize.sent_tokenize(corpus)
	words = []
	for sent in sents:
		tokens = list(filter(lambda word: word.isalnum(), nltk.word_tokenize(sent)))
		processed_text = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords]
		words.append(processed_text)
	# words = [list(filter(lambda word: word.isalnum(), nltk.word_tokenize(sent))) for sent in sents]
	# lemmatizer = nltk.stem.WordNetLemmatizer()
	# processed_text = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords]
	return words

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
                                           num_topics=10, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
	return lda_model

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

data2 = "These are great but not much better then gen1. Only addition is Siri feature. I will rather buy the previous model on discount and Save some green." +
	"Everyone is posting that there isn’t a difference between these and the 1st gen. This is misleading and inaccurate. Is the improvement drastic, no, but it is still an improvement. The improvement is that Apple has upgraded the on-board chip to the H1, which leads to faster and more stable pairing. This isn’t anecdotal. It’s been tested and proven to be faster. Also, if you opt for wireless charging, buy the case with the gen 2 AirPods and you’ll save a few dollars. If you already have the 1st gen, then it’s probably not worth the upgrade. If you are looking to buy your first pair of AirPods, then go for these." +
	"These AirPods are amazing they automatically play audio as soon as you put them in your ears and pause when you take them out. A simple double-tap during music listening will skip forward. To adjust the volume, change the song, make a call, or even get directions, just say 'Hey Siri' to activate your favorite personal assistant. Plus, when you're on a call or talking to Siri, an additional accelerometer works with dual beamforming microphones to filter out background noise and ensure that your voice is transmitted with clarity and consistency. Additionally, they deliver five hours of listening time on a single charge, and they're made to keep up with you thanks to a charging case that holds multiple additional charges for more than 24 hours of listening time. Just 15 minutes in the case gives you three hours of listening to time or up to two hours of talk time. I would highly recommend it to anyone looking to buy" +
	"Poor quality microphone. Not suitable for a remote worker taking calls. If your job requires dictation or a high quality mic, go elsewhere" +
	"We bought a brand new set of AirPods for $159. After using them for a week, I was listening to them and the right air pod went dead. I figured, maybe I just need to charge them. So when I got home I charged them, and the next day I went to use them again and the right air bud was still dead. So I did some research, tried to reset them, and then I couldn't reconnect them to my phone at all. Once I contacted Apple, they tried to give us the run around. They wanted us to let them fully die, which makes sense, so we did that. Then, when they still didn't work we wanted to exchange the faulty pair for a new set (BECAUSE WE PAID 159 DOLLARS AND ONLY USED THEM FOR A WEEK). Keep in mind, they were taken care of, not dropped, no water damage, they were expensive so they were treated delicately. Apple wanted a $180 deposit to get a new set, which is MORE THAN WHAT I ORIGINALLY PAID FOR. I decided to contact Amazon, and they suggested contacting Apple, but once I explained what was going, on Amazon offered to make things right and send out a replacement. Never had an issue with Amazon's customer service, but Apple was extremely disappointing. I won't purchase these again."


processed1 = [preprocess_text(review) for review in data1]
lda1 = [create_topic_model(review) for review in processed1]
for model in lda1:
	print("Review")
	print(model.print_topics())
	print()

processed2 = [preprocess_text(review) for review in data1]
lda1 = [create_topic_model(review) for review in processed1]
for model in lda1:
	print("Review")
	print(model.print_topics())
	print()



