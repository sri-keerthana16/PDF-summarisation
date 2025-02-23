# import google.generativeai as genai



# def summary(text):
#    prompt="""You are a summary assistant.Give concise summary of text that includes.
#    1.Title of the paper
#    2.Authors of the paper
#    3.concise summary of the text
#    4.Limitations discussed in the text"""
#    genai.configure(api_key="AIzaSyBmYlYbWdiY9yjgqJP8z8Ce2Jd28xxs6MI")
#    model = genai.GenerativeModel('gemini-pro')
#    response = model.generate_content([prompt,text])
#    return response


# from rouge import Rouge
# import sys
# from PyPDF2 import PdfReader
# from langdetect import detect
# def extract_text_from_pdf(pdf_path):
#   """Extracts text content from a PDF."""
#   with open(pdf_path, 'rb') as pdf_file:
#     reader = PdfReader(pdf_file)
#     text = ""
#     for page in reader.pages:
#       text += page.extract_text()
#     language=detect(text)
#     print(language)
#   return text
# from rouge_score import rouge_scorer

# def rouge_l_precision_score(reference, hypothesis):
#     scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#     scores = scorer.score(reference, hypothesis)
#     return scores['rougeL'].precision
# spanish_text=extract_text_from_pdf('D:\Pdf_summarization\pdf_files\spanish.pdf')
# english_text=extract_text_from_pdf('D:\Pdf_summarization\pdf_files/neural.pdf')
# german_text=extract_text_from_pdf('D:\Pdf_summarization\pdf_files\german.pdf')
# portuguese_text=extract_text_from_pdf('pdf_files\portuguese.pdf')
# french_text=extract_text_from_pdf('D:\Pdf_summarization\pdf_files/french.pdf')
# spanish=summary(spanish_text)
# english=summary(english_text)
# german=summary(german_text)
# portuguese=summary(portuguese_text)
# french=summary(french_text)
# rouge_l_p_spanish = rouge_l_precision_score(spanish_text, spanish.text)
# rouge_l_p_english = rouge_l_precision_score(english_text, english.text)
# rouge_l_p_german = rouge_l_precision_score(german_text, german.text)
# rouge_l_p_portuguese = rouge_l_precision_score(portuguese_text, portuguese.text)
# rouge_l_p_french = rouge_l_precision_score(french_text, french.text)

# print("ROUGE-L Precision Score for spanish :", rouge_l_p_spanish)
# print("\nROUGE-L Precision Score for english:", rouge_l_p_english)
# print("\nROUGE-L Precision Score for german:", rouge_l_p_german)
# print("\nROUGE-L Precision Score portuguese:", rouge_l_p_portuguese)
# print("\nROUGE-L Precision Score french:", rouge_l_p_french)




import google.generativeai as genai
from PyPDF2 import PdfReader
from langdetect import detect
from rouge import Rouge
import matplotlib.pyplot as plt
def detect_lang(text):
    language=detect(text)
    language_mapping = {
    "es": "spanish",
    "en": "english",
    "de": "german",
    "pt": "portuguese",
    "fr": "french"}
    return language_mapping[language]

def summary(text):
    prompt = """You are a summary assistant. Give a concise summary of the text that includes:
    1. Title of the paper
    2. Authors of the paper
    3. Concise summary of the text
    4. Limitations discussed in the text"""
    genai.configure(api_key="AIzaSyABVplBbg9mPFN5d9jUZzJPf-RZwkdrh9g")  
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([prompt, text])
    return response
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from langdetect import detect
from nltk.tokenize import word_tokenize

def clean_text(text):
    language = detect_lang(text)
    result_text = re.sub(r"\[\d+\]", "", text)
    result_text = result_text.lower()
    cleaned_text = re.sub(r'[,.]+', '', result_text)
    text_without_references = re.sub(r"references[\s\S]*", "", cleaned_text, flags=re.MULTILINE)
    text_without_formulas = re.sub(r'\$[^$]*\$|\(.*?\)|\[.*?\]|\{.*?\}', "", text_without_references)
    text_without_urls = re.sub(r'\b(?:https?|ftp|ftps|mailto)://\S+|www\.\S+|\S+\.(?:com|net|org|edu|gov|mil|html|co\.uk)(?:/\S*)?', '', text_without_formulas)
    cleaned_text = re.sub(r'[{}[\]()#!@$^&*/\|><,.`]', '', text_without_urls)
    stop_words = set(stopwords.words(language))
    words = word_tokenize(cleaned_text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)
    n = len(nltk.word_tokenize(filtered_text))
    return filtered_text
def extract_text_from_pdf(pdf_path):
    """Extracts text content from a PDF."""
    with open(pdf_path, 'rb') as pdf_file:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        language = detect(text)
        print(language)
    text=clean_text(text)
    return text
from rouge_score import rouge_scorer
def calculate_rouge_l_scores(summary, reference):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = scorer.score(summary.text, reference)['rougeL']
    return rouge_l_scores
 

# Paths to PDF files
pdf_paths = ['D:/Pdf_summarization/pdf_files/spanish.pdf', 
             'D:/Pdf_summarization/pdf_files/engproc-59-00194.pdf',
             'D:/Pdf_summarization/pdf_files/german.pdf',
             'D:/Pdf_summarization/pdf_files/portuguese.pdf',
             'D:/Pdf_summarization/pdf_files/french.pdf']

# Extract text from PDF files
texts = [extract_text_from_pdf(pdf) for pdf in pdf_paths]

# Generate summaries for each text
summaries = [summary(text) for text in texts]


# Plotting the scores
languages = ['Spanish', 'English', 'German', 'Portuguese', 'French']


rouge_l_scores = [calculate_rouge_l_scores(summary, reference) for summary, reference in zip(summaries, texts)]

# Print ROUGE-L scores in a list format
print("ROUGE-L scores:")
for language, score in zip(languages, rouge_l_scores):
    print(f"{language}: Precision={score.precision}, Recall={score.recall}, F1-score={score.fmeasure}")