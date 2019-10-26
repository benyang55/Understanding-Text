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

corpus = open("output.txt", "r").read()
print(corpus)
processed = preprocess_text(corpus)
print(processed)
lda = create_topic_model(processed)
print(lda.print_topics())




