from youtube_transcript_api import YouTubeTranscriptApi
import requests
from transformers import BartTokenizer, BartForConditionalGeneration

from transformers import AutoTokenizer, RobertaModel


from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

transcript = YouTubeTranscriptApi.get_transcript('tUX-frlNBJY', languages=['de', 'en'])

justwords = ''
for item in transcript:
    justwords = justwords + item['text'] + ' '


print(justwords)

model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

inputs = tokenizer.encode("summarize: " + justwords, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model.generate(inputs, max_length=100, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

# Decode and output the summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
# print("Original Text:")
print("\nSummary:")
print(summary)


from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pre-trained T5 model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Input text to be summarized


# Tokenize and summarize the input text using T5
inputs = tokenizer.encode("summarize: " + justwords, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

# Decode and output the summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
# print("Original Text:")
# print(justwords)
print("\nSummary:")
print(summary)




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