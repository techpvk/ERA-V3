from flask import Flask, render_template, jsonify, request, send_from_directory
from pathlib import Path
import logging
from train import train_model
from datetime import datetime
import threading
from queue import Queue
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Create directories
Path("static").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# Global variables for tracking training status
is_training = False
current_model_id = None
current_epoch = 0
current_loss = 0.0
current_accuracy = 0.0
training_log = []
training_queue = Queue()

# Initialize empty model results
model_results = {
    1: {
        'train_loss': {'x': [], 'y': []},
        'train_accuracy': {'x': [], 'y': []},
        'val_loss': {'x': [], 'y': []},
        'val_accuracy': {'x': [], 'y': []},
        'config': None
    },
    2: {
        'train_loss': {'x': [], 'y': []},
        'train_accuracy': {'x': [], 'y': []},
        'val_loss': {'x': [], 'y': []},
        'val_accuracy': {'x': [], 'y': []},
        'config': None
    }
}

def generate_results_plot(model_id):
    """Generate and save result plots for the model"""
    try:
        # Load the best model
        checkpoint = torch.load(f'model_{model_id}_best.pth')
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot training and validation metrics
        results = model_results[model_id]
        
        # Plot 1: Loss
        ax1.plot(results['train_loss']['x'], results['train_loss']['y'], 
                label='Training Loss', color='#64ffda')
        ax1.plot(results['val_loss']['x'], results['val_loss']['y'], 
                label='Validation Loss', color='#ff6464')
        ax1.set_title('Loss Over Time')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        ax2.plot(results['train_accuracy']['x'], results['train_accuracy']['y'], 
                label='Training Accuracy', color='#64ffda')
        ax2.plot(results['val_accuracy']['x'], results['val_accuracy']['y'], 
                label='Validation Accuracy', color='#ff6464')
        ax2.set_title('Accuracy Over Time')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Set style for both plots
        for ax in [ax1, ax2]:
            ax.set_facecolor('none')
            ax.spines['bottom'].set_color('#ccd6f6')
            ax.spines['top'].set_color('#ccd6f6')
            ax.spines['right'].set_color('#ccd6f6')
            ax.spines['left'].set_color('#ccd6f6')
            ax.tick_params(colors='#ccd6f6')
            
        plt.tight_layout()
        plt.savefig(f'static/results_model_{model_id}.png', 
                   facecolor='none', 
                   edgecolor='none',
                   transparent=True)
        plt.close()
        
        return True
    except Exception as e:
        logger.error(f"Error generating results plot: {str(e)}")
        return False

def training_worker():
    """Background worker to process training requests"""
    global is_training, current_model_id, current_epoch
    
    while True:
        # Wait for training request
        training_params = training_queue.get()
        if training_params is None:
            continue
            
        try:
            is_training = True
            current_model_id = training_params['model_id']
            current_epoch = 0
            
            # Log training start
            timestamp = datetime.now().strftime('%H:%M:%S')
            training_log.append(f"[{timestamp}] Starting training for Model {current_model_id}")
            
            # Enable server mode for metrics updates
            from train import update_metrics
            update_metrics.server_mode = True
            
            # Start training
            best_accuracy = train_model(
                model_id=training_params['model_id'],
                channels=training_params['channels'],
                batch_size=training_params['batch_size'],
                epochs=training_params['epochs'],
                results=model_results[training_params['model_id']],
                training_log=training_log
            )
            
            # Generate results plot
            generate_results_plot(current_model_id)
            
            # Log completion
            timestamp = datetime.now().strftime('%H:%M:%S')
            training_log.append(
                f"[{timestamp}] Model {current_model_id} completed. Best accuracy: {best_accuracy:.2f}%"
            )
            
            logger.info(f"Training completed for model {current_model_id}. Best accuracy: {best_accuracy:.2f}%")
            
        except Exception as e:
            timestamp = datetime.now().strftime('%H:%M:%S')
            error_message = f"Error in training: {str(e)}"
            training_log.append(f"[{timestamp}] {error_message}")
            logger.error(error_message)
        finally:
            is_training = False
            current_model_id = None
            current_epoch = 0
            training_queue.task_done()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_training', methods=['POST'])
def start_training():
    global is_training, current_model_id
    
    if is_training:
        return jsonify({
            'status': 'error',
            'message': 'Training already in progress'
        })
    
    try:
        data = request.get_json()
        model_id = data['model_id']
        channels = data['channels']
        batch_size = int(data.get('batch_size', 512))
        epochs = int(data.get('epochs', 1))
        
        # Clear previous results for this model
        model_results[model_id] = {
            'train_loss': {'x': [], 'y': []},
            'train_accuracy': {'x': [], 'y': []},
            'val_loss': {'x': [], 'y': []},
            'val_accuracy': {'x': [], 'y': []},
            'config': {
                'channels': channels,
                'batch_size': batch_size,
                'epochs': epochs
            }
        }
        
        # Queue the training request
        training_queue.put({
            'model_id': model_id,
            'channels': channels,
            'batch_size': batch_size,
            'epochs': epochs
        })
        
        return jsonify({
            'status': 'success',
            'message': f'Training queued for model {model_id}'
        })
        
    except Exception as e:
        logger.error(f"Error queuing training: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/update_metrics', methods=['POST'])
def update_metrics():
    global current_loss, current_accuracy, current_epoch, training_log
    
    try:
        data = request.get_json()
        print(f"Received metrics update: {data}")  # Debug print
        
        model_id = int(data['model_id'])
        iteration = int(data['iteration'])
        loss = float(data['loss'])
        accuracy = float(data['accuracy'])
        current_epoch = int(data.get('current_epoch', 0))
        
        # Always update current metrics
        current_loss = loss
        current_accuracy = accuracy
        
        # Update model results
        model_results[model_id]['train_loss']['x'].append(iteration)
        model_results[model_id]['train_loss']['y'].append(loss)
        model_results[model_id]['train_accuracy']['x'].append(iteration)
        model_results[model_id]['train_accuracy']['y'].append(accuracy)
        
        # Add log message
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp}] Model {model_id} - Epoch {current_epoch} - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%"
        training_log.append(log_message)
        print(log_message)  # Debug print
        
        return jsonify({
            'status': 'success',
            'model_id': model_id,
            'current_loss': loss,
            'current_accuracy': accuracy,
            'current_epoch': current_epoch
        })
    except Exception as e:
        logger.error(f"Error updating metrics: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_comparison_data')
def get_comparison_data():
    """Add this route to get data for both models"""
    return jsonify({
        'model1': model_results[1],
        'model2': model_results[2]
    })

@app.route('/get_metrics')
def get_metrics():
    """Modified to return metrics even if no model is training"""
    if current_model_id is None:
        return jsonify({
            'status': 'success',
            'message': 'No model currently training',
            'train_loss': {'x': [], 'y': []},
            'train_accuracy': {'x': [], 'y': []},
            'val_loss': {'x': [], 'y': []},
            'val_accuracy': {'x': [], 'y': []}
        })
    return jsonify({
        'status': 'success',
        **model_results[current_model_id]
    })

@app.route('/get_training_status')
def get_training_status():
    return jsonify({
        'is_training': is_training,
        'current_model': current_model_id,
        'current_epoch': current_epoch,
        'current_loss': float(current_loss),  # Ensure it's serializable
        'current_accuracy': float(current_accuracy)  # Ensure it's serializable
    })

@app.route('/get_log')
def get_log():
    return jsonify({'log': training_log})

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # Start the training worker thread
    training_thread = threading.Thread(target=training_worker, daemon=True)
    training_thread.start()
    
    # Get the environment setting
    env = os.environ.get('FLASK_ENV', 'development')
    
    if env == 'production':
        # Basic production configuration
        app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
    else:
        # Development configuration
        app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False) 