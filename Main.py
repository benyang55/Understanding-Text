
# from Test import *
from TextProcessing import *
from Sentiment import *
from scraper import *
import sys

def Main():
	query = sys.argv[1]

	reviews = grabReviews(scrape(findQuery(query)))

	words, sents = preprocess_text(reviews)
	
	ldamodel, worddic = create_topic_model(words)
	
	keywords = extract_keywords(ldamodel, worddic)
	
	sentencestocheck = find_keyword_contexts(sents, keywords)
	
	finalsentences = sentimentcalculator(sentencestocheck, reviews)

	concatenated = ""
	for sentence in finalsentences:
		concatenated += sentence + " "
	
	print(concatenated)

	return concatenated

    # textprocessed = textprocessor()

Main()
