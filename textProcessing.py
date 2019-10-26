import nltk
import gensim
import spacy
nltk.download()

def preprocess_text(corpus):
	stopwords = nltk.corpus.stopwords
	tokens = nltk.word_tokenize(corpus)
	text = nltk.Text(tokens)
	print(type(text))
	lemmatizer = nltk.stem.WordNetLemmatizer()
	processed_text = [lemmatizer.lemmatize(word) for word in text]

	return processed_text