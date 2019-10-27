from django.shortcuts import render
import requests
from subprocess import run, PIPE
import sys
import os
import re

def output(request, ):
	data = requests.get("https://regres.in/api/users").text
	return render(request, 'ProductReviews.html', {'data': data})


def searchButton(request):
	return render(request, 'ProductReviews.html')


def external(request):
	inp = request.POST.get("query")
	out = run(["./execute.sh", inp], capture_output=True, text=True).stdout
	sentences = re.split('[.?!]', out)
	print(type(sentences), len(sentences))
	html = "<h4>Customer Review of %s</h4>" % inp
	html += "<li>Summary of Product Negatives"
	for i in range(len(sentences) // 2):
		sentence = sentences[i]
		html += "<ol> " + sentence + " </ol>"
	html += "</li>"
	html += "<li>Summary of Product Positives"
	for i in range(len(sentences) // 2, len(sentences)):
		sentence = sentences[i]
		html += "<ol> " + sentence + " </ol>"
	html += "</li>"
	return render(request, 'ProductReviews.html', {'data_external': html})