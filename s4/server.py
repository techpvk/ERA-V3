from flask import Flask, render_template, jsonify, request, send_from_directory
from pathlib import Path

app = Flask(__name__)

# Create directories if they don't exist
Path("templates").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

# Global variables to store training data
training_log = []
metrics = {
    'train_loss': {'x': [], 'y': []},
    'train_accuracy': {'x': [], 'y': []},
    'val_loss': {'x': [], 'y': []},
    'val_accuracy': {'x': [], 'y': []}
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_metrics', methods=['POST'])
def update_metrics():
    data = request.get_json()
    if data.get('is_validation', False):
        metrics['val_loss']['x'].append(data['iteration'])
        metrics['val_loss']['y'].append(data['loss'])
        metrics['val_accuracy']['x'].append(data['iteration'])
        metrics['val_accuracy']['y'].append(data['accuracy'])
    else:
        metrics['train_loss']['x'].append(data['iteration'])
        metrics['train_loss']['y'].append(data['loss'])
        metrics['train_accuracy']['x'].append(data['iteration'])
        metrics['train_accuracy']['y'].append(data['accuracy'])
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

@app.route('/static/ai-background.jpg')
def serve_background():
    return send_from_directory('static', 'ai-background.jpg')

if __name__ == '__main__':
    app.run(debug=True) 