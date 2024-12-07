from youtube_transcript_api import YouTubeTranscriptApi
import requests
from transformers import BartTokenizer, BartForConditionalGeneration
import os
from transformers import AutoTokenizer, RobertaModel
import serpapi
import openai
import http
import json

openai.api_key = os.environ.get("OPENAI_API_KEY")



from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

transcript = YouTubeTranscriptApi.get_transcript('tUX-frlNBJY', languages=['en'])

f = open("textfile.txt", "w", encoding="utf-8")

trans = transcript
justwords = ''

for setntence in trans:
    justwords = justwords + setntence['text'] + ' '
    f.write(setntence['text'] + ' ')
f.write("\n\n")

print(justwords)


def split_sentences(input_string):
    # Split the string by newlines and remove any leading/trailing whitespace from each sentence
    sentences = [sentence.strip() for sentence in input_string.split('\n') if sentence.strip()]
    return sentences

import http.client
import json

import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("get the claims from the following text and seperate each claim with a new line and only write the 5 most important claims just write the claims seperated by a new line:" + justwords)
split_response = split_sentences(response.text )
for item in split_response:
    f.write(item + '\n')
    print(item)
f.write("\n\n")


# f = open("textfile.txt", "r", encoding="utf-8")
# querys = f.read()
linkslist = []



for i in range(5):
    payload = json.dumps({
    "q": split_response[i]
    })
    headers = {
        'X-API-KEY': 'c40d9e39db24a307b761233401835fc29a618489',
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", "https://google.serper.dev/search", headers=headers, data=payload)
    search_results = response.json()['organic']
    temp = []
    counter = 0
    for k in search_results:
        if (counter == 3):
            break
        temp.append(k['link'])
        counter += 1

    linkslist.append(temp)

for item in linkslist:
    for link in item:
        f.write(link + "\n")
        print(link)
    f.write("\n")
    print()
# print(data.decode("utf-8"))


f.close()

# model_name = "facebook/bart-large-cnn"
# tokenizer = BartTokenizer.from_pretrained(model_name)
# model = BartForConditionalGeneration.from_pretrained(model_name)

# inputs = tokenizer.encode("summarize: " + justwords, return_tensors="pt", max_length=1024, truncation=True)
# summary_ids = model.generate(inputs, max_length=100, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

# # Decode and output the summary
# summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
# # print("Original Text:")
# print("\nSummary:")
# print(summary)


# from transformers import T5Tokenizer, T5ForConditionalGeneration

# # Load pre-trained T5 model and tokenizer
# model_name = "t5-small"
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name)

# # Input text to be summarized


# # Tokenize and summarize the input text using T5
# inputs = tokenizer.encode("summarize: " + justwords, return_tensors="pt", max_length=1024, truncation=True)
# summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

# # Decode and output the summary
# summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
# # print("Original Text:")
# # print(justwords)
# print("\nSummary:")
# print(summary)




# from summarizer import Summarizer

# # Input text to be summarized
# input_text = justwords

# # Create a BERT extractive summarizer
# summarizer = Summarizer()

# # Generate the summary
# summary = summarizer(input_text, min_length=50, max_length=150)  # You can adjust the min_length and max_length parameters

# # Output the summary
# print("Original Text:")
# print(input_text)
# print("\nSummary:")
# print(summary)



# tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
# model = RobertaModel.from_pretrained("FacebookAI/roberta-base")

# inputs = tokenizer('hello world', return_tensors="pt")
# outputs = model(**inputs)
# output = tokenizer.batch_decode(outputs)
# print(output)




# def extract_important_claims(transcript):
#     prompt = f"""
#     Please identify the 5 most important claims in the following transcript:
#     {transcript}

#     Format the response as:
#     Claim 1
#     Claim 2
#     Claim 3
#     Claim 4
#     Claim 5
#     """
    
#     response = openai.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.7
#     )
    
#     return response['choices'][0]['message']['content']

# # Example usage
# transcript = justwords
# important_claims = extract_important_claims(transcript)
# f.write(important_claims)
# f.write("\n\n")
# print(important_claims)
