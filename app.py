from flask import Flask, render_template, request, jsonify
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np
import networkx as nx
import nltk

nltk.download('stopwords')
nltk.download('punkt')

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = set()

    sent1 = [token.lower() for token in sent1 if token.lower() not in stopwords]
    sent2 = [token.lower() for token in sent2 if token.lower() not in stopwords]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)  
    vector2 = [0] * len(all_words)

    for w in sent1:
        vector1[all_words.index(w)] += 1

    for w in sent2:
        vector2[all_words.index(w)] += 1

    return 1 - nltk.cluster.util.cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stopwords):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stopwords)
    return similarity_matrix

def visualize_similarity_matrix(similarity_matrix):
    # Implementation of visualize_similarity_matrix function
    pass

def visualize_sentence_similarity_graph(graph, sentences):
    # Implementation of visualize_sentence_similarity_graph function
    pass

def summarize_text(text, summary_length):
    stop_words = set(stopwords.words('english'))
    sentences = sent_tokenize(text)
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    sentence_similarity_matrix = build_similarity_matrix(tokenized_sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    top_n = int(len(sentences) * summary_length / 100)
    summarize_text = [ranked_sentences[i][1] for i in range(top_n)]
    summary = ' '.join(summarize_text)
    return summary

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    summary_length = float(request.form['summaryLength']) / 100  # Convert to percentage
    summary = summarize_text(text, summary_length)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
