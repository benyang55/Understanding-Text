
# from Test import *
from TextProcessing import *
from Sentiment import *
from scraper import *
import sys

def Main():
	query = sys.argv[1]
	print(query)
   	
	reviews = grabReviews(scrape(findQuery(query)))
	
	words, sents = preprocess_text(reviews)
	
	ldamodel, worddic = create_topic_model(words)
	
	keywords = extract_keywords(ldamodel, worddic)
	
	sentencestocheck = find_keyword_contexts(sents, keywords)

	print(sentencestocheck)
	
	finalsentences = sentimentcalculator(sentencestocheck, reviews)
	
	print(finalsentence)

    # textprocessed = textprocessor()

Main()
