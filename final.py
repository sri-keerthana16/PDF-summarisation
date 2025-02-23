# from flask import Flask, render_template, request, make_response
# import os
# from PyPDF2 import PdfReader
# import re
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# from langchain_google_genai import ChatGoogleGenerativeAI
# import google.generativeai as genai

# app = Flask(__name__)

# uploads_dir = os.path.join(app.instance_path, 'pdf_files')
# os.makedirs(uploads_dir, exist_ok=True)

# def extract_text_from_pdf(file):
#     text = ""
#     with open(file, 'rb') as pdf_file:
#         pdf_reader = PdfReader(pdf_file)
#         for page_num in range(len(pdf_reader.pages)):
#             page = pdf_reader.pages[page_num]
#             text += page.extract_text()
#     return text

# def clean_text(text):
#     language = 'english'
#     result_text = re.sub(r"\[\d+\]", "", text)
#     result_text = result_text.lower()
#     cleaned_text = re.sub(r'[,.]+', '', result_text)
#     text_without_references = re.sub(r"references[\s\S]*", "", cleaned_text, flags=re.MULTILINE)
#     text_without_formulas = re.sub(r'\$[^$]*\$|\(.*?\)|\[.*?\]|\{.*?\}', "", text_without_references)
#     text_without_urls = re.sub(r'\b(?:https?|ftp|ftps|mailto)://\S+|www\.\S+|\S+\.(?:com|net|org|edu|gov|mil|html|co\.uk)(?:/\S*)?', '', text_without_formulas)
#     cleaned_text = re.sub(r'[{}[\]()#!@$^&*/\|><,.`]', '', text_without_urls)
#     stop_words = set(stopwords.words(language))
#     words = word_tokenize(cleaned_text)
#     filtered_words = [word for word in words if word.lower() not in stop_words]
#     filtered_text = ' '.join(filtered_words)
#     n = len(nltk.word_tokenize(filtered_text))
#     return filtered_text

# def summarize_text(text):
#     genai.configure(api_key="AIzaSyBmYlYbWdiY9yjgqJP8z8Ce2Jd28xxs6MI")
#     vision_model = genai.GenerativeModel('gemini-pro')
#     prompt="""You are a summary assistant. Give summary content that also include 
#     -title
#     -authors
#     -concise summary that contains about the paper,methodoligies followed,and results.

#    """
#     response = vision_model.generate_content([prompt, text])
#     response = response.text
#     response = response.replace("**", "\n\n")
#     return response

# def language_translation(text, language):
#     from deep_translator import GoogleTranslator
#     translator = GoogleTranslator(source='auto', target=language).translate(text)
#     return translator
# def combined1(text):
#     cleaned_text = clean_text(text)
#     summary = summarize_text(cleaned_text)
#     return summary
# def combined2(summary,language):
#     translated_summary = language_translation(summary, language)
#     translated_summary=translated_summary.replace("*",'\n').replace(".",'\n')
#     summary = summary +'\n\n\n'+ translated_summary
#     return summary
    






# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     file = request.files['file']
#     language = request.form.get('language')
#     if file.filename == '':
#         return render_template('index.html', error='No file selected')
    
#     # Save the uploaded PDF file
#     file_path = os.path.join(uploads_dir, file.filename)
#     file.save(file_path)

#     # Extract text from the PDF file
#     text = extract_text_from_pdf(file_path)
    
#     # Clean the extracted text
    
#     # Summarize the cleaned text
#     summary=combined1(text)
#     # Remove the uploaded PDF file
#     os.remove(file_path)


#     # Translate the summary if a language is selected
#     if language:
#         summary=combined2(summary,language)
        
#     return render_template('result.html', summary=summary)
# @app.route('/download_summary', methods=['GET'])
# def download_summary():
#     # summary = request.args.get('summary','')
#     if 'text' not in request.args:
#         return "No text provided"
    
#     text = request.args['text']
#     summary=combined2(combined1(text))
    
    
#     # Create a response object with the provided summary
#     response = make_response(summary)
    
#     # Set the appropriate headers for a text file download
#     response.headers['Content-Type'] = "text/plain"
#     response.headers['Content-Disposition'] = "attachment; filename=summary.txt"
    
#     return response

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, make_response
import os
from PyPDF2 import PdfReader
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

app = Flask(__name__)

uploads_dir = os.path.join(app.instance_path, 'pdf_files')
os.makedirs(uploads_dir, exist_ok=True)

def extract_text_from_pdf(file):
    text = ""
    with open(file, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text
def detect_lang(text):
    language=detect(text)
    language_mapping = {
    "es": "spanish",
    "en": "english",
    "de": "german",
    "pt": "portuguese",
    "fr": "french"}
    return language_mapping[language]
    


def clean_text(text):
    language = 'english'
    language=detect_lang(text)
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

def summarize_text(prompt,text):
    
    genai.configure(api_key="AIzaSyABVplBbg9mPFN5d9jUZzJPf-RZwkdrh9g")
    vision_model = genai.GenerativeModel('gemini-pro')
#     prompt="""You are a summary assistant. Give summary that includes 
#     -title
#     -authors
#     -A summary about methodologies,performance.

#    """
    prompt="You are a summary assistant."+ prompt
    response = vision_model.generate_content([prompt, text])
    response = response.text
    response = response.replace("**", "\n\n")
    return response

def language_translation(text, language):
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source='auto', target=language).translate(text)
    return translator

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    language = request.form.get('language')
    prompt= request.form.get('prompt')
    if file.filename == '':
        return render_template('index.html', error='No file selected')
    
    # Save the uploaded PDF file
    file_path = os.path.join(uploads_dir, file.filename)
    file.save(file_path)

    # Extract text from the PDF file
    text = extract_text_from_pdf(file_path)
    
    # Clean the extracted text
    cleaned_text = clean_text(text)
    
    # Summarize the cleaned text
    summary = summarize_text(prompt,cleaned_text)
    
    # Remove the uploaded PDF file
    os.remove(file_path)

    # Translate the summary if a language is selected
    if language:
        translated_summary = language_translation(summary, language)
        # summary = summary +'\n\n\n'+ translated_summary
        summary=translated_summary.replace("*",'\n').replace(".",'.\n')
    
    return render_template('result.html', summary=summary)

@app.route('/download_summary', methods=['GET'])
def download_summary():
    summary = request.args.get('summary','')
    
    # Create a response object with the provided summary
    response = make_response(summary)
    
    # Set the appropriate headers for a text file download
    response.headers['Content-Type'] = "text/plain"
    response.headers['Content-Disposition'] = "attachment; filename=summary.txt"
    
    return response

if __name__ == '__main__':
    app.run(debug=True)

