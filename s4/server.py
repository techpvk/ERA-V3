from flask import Flask, render_template, jsonify, request
from pathlib import Path

app = Flask(__name__)

# Create directories if they don't exist
Path("templates").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

# Global variables to store training data
training_log = []
metrics = {
    'loss': {'x': [], 'y': []},
    'accuracy': {'x': [], 'y': []}
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_metrics', methods=['POST'])
def update_metrics():
    data = request.get_json()
    metrics['loss']['x'].append(data['iteration'])
    metrics['loss']['y'].append(data['loss'])
    metrics['accuracy']['x'].append(data['iteration'])
    metrics['accuracy']['y'].append(data['accuracy'])
    return jsonify({'status': 'success'})

@app.route('/update_log', methods=['POST'])
def update_log():
    data = request.get_json()
    training_log.append(data['message'])
    return jsonify({'status': 'success'})

@app.route('/get_metrics')
def get_metrics():
    return jsonify(metrics)

@app.route('/get_log')
def get_log():
    return jsonify({'log': training_log})

if __name__ == '__main__':
    app.run(debug=True) 