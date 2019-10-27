import pandas as pd
import re

dataset = [
	"These are great but not much better then gen1. Only addition is Siri feature. I will rather buy the previous model on discount and Save some green.",
	"Everyone is posting that there isn’t a difference between these and the 1st gen. This is misleading and inaccurate. Is the improvement drastic, no, but it is still an improvement. The improvement is that Apple has upgraded the on-board chip to the H1, which leads to faster and more stable pairing. This isn’t anecdotal. It’s been tested and proven to be faster. Also, if you opt for wireless charging, buy the case with the gen 2 AirPods and you’ll save a few dollars. If you already have the 1st gen, then it’s probably not worth the upgrade. If you are looking to buy your first pair of AirPods, then go for these.",
	"These AirPods are amazing they automatically play audio as soon as you put them in your ears and pause when you take them out. A simple double-tap during music listening will skip forward. To adjust the volume, change the song, make a call, or even get directions, just say 'Hey Siri' to activate your favorite personal assistant. Plus, when you're on a call or talking to Siri, an additional accelerometer works with dual beamforming microphones to filter out background noise and ensure that your voice is transmitted with clarity and consistency. Additionally, they deliver five hours of listening time on a single charge, and they're made to keep up with you thanks to a charging case that holds multiple additional charges for more than 24 hours of listening time. Just 15 minutes in the case gives you three hours of listening to time or up to two hours of talk time. I would highly recommend it to anyone looking to buy",
	"Poor quality microphone. Not suitable for a remote worker taking calls. If your job requires dictation or a high quality mic, go elsewhere",
	"We bought a brand new set of AirPods for $159. After using them for a week, I was listening to them and the right air pod went dead. I figured, maybe I just need to charge them. So when I got home I charged them, and the next day I went to use them again and the right air bud was still dead. So I did some research, tried to reset them, and then I couldn't reconnect them to my phone at all. Once I contacted Apple, they tried to give us the run around. They wanted us to let them fully die, which makes sense, so we did that. Then, when they still didn't work we wanted to exchange the faulty pair for a new set (BECAUSE WE PAID 159 DOLLARS AND ONLY USED THEM FOR A WEEK). Keep in mind, they were taken care of, not dropped, no water damage, they were expensive so they were treated delicately. Apple wanted a $180 deposit to get a new set, which is MORE THAN WHAT I ORIGINALLY PAID FOR. I decided to contact Amazon, and they suggested contacting Apple, but once I explained what was going, on Amazon offered to make things right and send out a replacement. Never had an issue with Amazon's customer service, but Apple was extremely disappointing. I won't purchase these again.",
    "VADER is smart, handsome, and funny.",
    "VADER is smart, handsome, and funny!"
]

sentences = ['These are great but not much better then gen1', 'Poor quality microphone',
'These AirPods are amazing they automatically play audio as soon as you put them in your ears and pause when you take them out',
'Everyone is posting that there isn’t a difference between these and the 1st gen','Also, if you opt for wireless charging, buy the case with the gen 2 AirPods and you’ll save a few dollars' ]

def sentimentcalculator(sentences, reviews):
    sent = pd.read_csv('vader_lexicon.txt', sep = '\t', header = None, names = ['polarity', 'weight', 'valence'])
    sent = sent.drop('weight', axis = 1).drop('valence', axis =1 )
    #needs to be converted to csv format later

    #ALL NEEDS TO BE MODIFIED WHEN GET THE ACTUAL TEXT FILE WITH COMMENTS
    testdata = pd.DataFrame(reviews, columns = ['text'])
    testdata['text'] = testdata['text'].str.lower()
    testdata['id'] = list(range(len(testdata)))
    testdatasplit = pd.DataFrame(sentences, columns = ['text'])
    '''#for i in range(len(testdata)):
        #sentences = testdata.loc[i, 'text'].split(".")
        #for x in sentences:
            if re.match(x, '\s'):
                continue

            testdatasplit = testdatasplit.append({'id': i, 'text': x}, ignore_index = True)'''

    temp = testdatasplit['text'].str.split(expand = True).stack().reset_index(level = 1)

    tidy_format = temp.rename(columns = {"level_1" : "num", 0 : "word"})
    tidy_format = tidy_format.reset_index(level = 0)
    merged = tidy_format.join(sent, on = 'word').fillna(value = 0)

    merged = merged.groupby('index').sum()
    merged = merged.drop(['num'], axis = 1)

    testdatasplit = testdatasplit.join(merged)
    #for the whole thing

    temp = testdata['text'].str.split(expand = True).stack().reset_index(level = 1)
    tidy_format = temp.rename(columns = {"level_1" : "num", 0 : "word"})
    tidy_format = tidy_format.reset_index(level = 0)
    merged = tidy_format.join(sent, on = 'word').fillna(value = 0)

    merged = merged.groupby('index').sum()
    merged = merged.drop(['num'], axis = 1)

    testdata = testdata.join(merged)


    #top 3 negative sentences


    #print(testdatasplit.columns)
    negativetable = testdatasplit.sort_values(by = 'polarity', ascending = True)
    negativevals = []
    negativevals.append(negativetable.iloc[0]['text'])
    negativevals.append(negativetable.iloc[1]['text'])
    negativevals.append(negativetable.iloc[2]['text'])


    #print(negativevals)
    #top 3 positive after checking for overall score
    positivetable = testdatasplit.sort_values(by = 'polarity', ascending = False)
    positivevals = []
    #print(positivetable)
    for i in range(len(positivetable)):
        #idnum = positivetable.iloc[i]['id']
        text = positivetable.iloc[i]['text']
        text = text.lower()
        #print(text)
        if len(positivevals) == 3 or i == len(positivetable):
            break
        if testdata[testdata['text'].str.contains(text)]['polarity'].iloc[0] > 0:
            positivevals.append(text)

        #print(testdata[testdata['text'].str.contains(text)]['polarity'].iloc[i])
        #print(testdata['text'])

    #print(positivevals)
    negativevals = negativevals + (positivevals)
    #print(negativevals)
    return negativevals


sentimentcalculator(sentences, dataset)
