from flask import Flask, render_template, request, jsonify
from summarization_script import summarize_text

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    summary_length = int(request.form['summaryLength'])
    summary = summarize_text(text, summary_length)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
