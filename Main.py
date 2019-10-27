
from .Test import textprocessor
from .textProcessing import *

def Main():
    #scraping the reviews goes here, goes into a list with each review as its own entry
    reviews = ...
    words, sents = preprocess_text(reviews)

    ldamodel, worddic = create_topic_model(words)

    keywords = extract_keywords(ldamodel, worddic)

    sentencestocheck = find_keyword_contexts(sents, keywords)

    finalsentences = sentimentcalculator(sentencestocheck, reviews)

    textprocessed = textprocessor()
