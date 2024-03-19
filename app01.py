from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from wordcloud import WordCloud
from flask import url_for
import uuid
import tempfile
import base64
from io import BytesIO
import os
from gensim import corpora, models
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import spacy  # 还要下载python -m spacy download en_core_web_sm
import nlp
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
import spacy
import collections
import os
import random
import time
from tqdm import tqdm
import torch
from torch import nn
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model
from collections import Counter
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import string
import pickle
import json

app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True
CORS(app)  # 允许所有来源的跨域请求，注意：在生产环境中，你可能需要限制这一点
# 加载spacy的英文模型
nlp = spacy.load("en_core_web_sm")

# 读取配置文件
with open("config/config.json") as f:
    config = json.load(f)


@app.route("/api/wordcloud", methods=["GET", "POST"])
def analyze_message():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    return generate_wordcloud(user_message)


def preprocess_text(text):
    # 使用spacy进行文本预处理
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.lemma_ !=
              "-PRON-" and token.is_alpha]

    return " ".join(tokens)


def generate_wordcloud(user_message):
    # 预处理文本
    processed_text = preprocess_text(user_message)

    # 生成词云
    wc = WordCloud(width=400, height=400,
                   background_color="white").generate(processed_text)
    img = BytesIO()
    wc.to_image().save(img, format="PNG")
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    result = {
        "response": f"data:image/png;base64,{img_base64}"
    }
    return jsonify(result)


nlp = spacy.load("en_core_web_sm")


def get_hotwords(text):
    result = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN']
    doc = nlp(text.lower())
    for token in doc:
        if (token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        if (token.pos_ in pos_tag):
            result.append(token.text)
    return result


@app.route("/api/lda", methods=["POST"])
def hotwords():
    data = request.json
    text = data.get('message', '')
    if not text:
        return jsonify(error="No message provided"), 400

    output = set(get_hotwords(text))
    most_common_list = Counter(output).most_common(10)
    hotwords_list = [item[0] for item in most_common_list]
    # return jsonify(hotwords=hotwords_list)
    topics = ",".join(hotwords_list)
    return jsonify({"topics": topics})


# 加载 T5 模型和分词器
model_name = "t5-large"  
tokenizer2 = T5Tokenizer.from_pretrained(model_name, model_max_length=config["max_length"])
model2 = T5ForConditionalGeneration.from_pretrained(model_name)

def generate_summary_with_t5(text):

    # 对文本进行分词
    inputs = tokenizer2.encode(
        "summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)

    # 生成摘要
    # 调整生成参数以优化性能
    summary_ids = model2.generate(
        inputs, max_length=200, min_length=50, length_penalty=1.0, num_beams=6, early_stopping=True)

    # 解码生成的文本
    summary = tokenizer2.decode(summary_ids[0], skip_special_tokens=True)

    return summary


@app.route("/api/generate_summary", methods=["GET", "POST"])
def generate_gpt_summary():
    try:
        user_message = request.json.get("message")
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        summary = generate_summary_with_t5(user_message)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Load the tokenizer and model
tokenizer3 = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
model3 = AutoModelForSequenceClassification.from_pretrained(
    "SamLowe/roberta-base-go_emotions")

@app.route("/api/sentiment", methods=["GET", "POST"])
def predict_emotion():
    # Get sentence from the request data
    data = request.json
    test_sentence = data.get("message", "")

    # Check if sentence is not empty
    if not test_sentence:
        return jsonify({"error": "No sentence provided"}), 400

    # Encode the sentence
    encoded_input = tokenizer3(test_sentence, return_tensors="pt")

    # Use the model to perform inference
    with torch.no_grad():
        output = model3(**encoded_input)

    # Get the logits
    logits = output.logits

    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=1)

    # Get the most likely emotion category for the sentence
    prediction = torch.argmax(probabilities, dim=1)

    # Create response data
    response_data = {
        "predicted_emotion": model3.config.id2label[prediction.item()],
        "probability": f"{probabilities[0][prediction].item():.4f}"
    }

    # Return JSON response
    return jsonify(response_data)


# Load the tokenizer and model
tokenizer4 = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
model4 = AutoModelForSequenceClassification.from_pretrained(
    "facebook/bart-large-mnli")

def predict_news_topics_with_probabilities(content, topics):
    # Create premise-hypothesis pairs
    premise_hypothesis_pairs = [
        (content, f"This text is about {topic}.") for topic in topics]

    # Encode the pairs
    encoded_input = tokenizer4(premise_hypothesis_pairs,
                               padding=True, truncation=True, return_tensors="pt")

    # Predict with the model
    with torch.no_grad():
        outputs = model4(**encoded_input)

    # Get logits and apply softmax
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)

    # Get 'entailment' probabilities
    entailment_probs = probabilities[:, model4.config.label2id['entailment']]

    # Sort the topics by probabilities
    sorted_probs, sorted_topics = torch.sort(entailment_probs, descending=True)
    sorted_topics = [topics[idx] for idx in sorted_topics]

    # Return sorted topics with their probabilities
    return list(zip(sorted_topics, sorted_probs.tolist()))


@app.route('/predict_topic', methods=['POST'])
def predict_topic():
    # Get content from the request JSON data
    data = request.json
    content = data.get("message", "")

    # Check if content is not empty
    if not content:
        return jsonify({"error": "No content provided"}), 400

    news_topics = [
        "business", "environment", "fashion", "medicine", "science",
        "music", "traffic", "weather", "technology", "sports",
        "politics", "health", "entertainment"]

    # Call the prediction function
    predicted_topics_with_probs = predict_news_topics_with_probabilities(
        content, news_topics)

    # Get the topic with the highest probability
    highest_prob_topic, highest_prob = predicted_topics_with_probs[0]

    # Create response data
    response_data = {
        "topic": highest_prob_topic,
        "probability": f"{highest_prob:.4f}"
    }

    # Return JSON response
    return jsonify(response_data)


# Function to preprocess the text
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)  # Remove special chars
    text = re.sub('https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# Assuming the model and vectorizer have been trained and saved as .pkl files
with open('rfc_model.pkl', 'rb') as model_file:
    RFC = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorization = pickle.load(vectorizer_file)


# Function to return the label
def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

# Route for predicting


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    news = data['message']
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_RFC = RFC.predict(new_xv_test)
    result = output_label(pred_RFC[0])

    return jsonify({"prediction": result})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
