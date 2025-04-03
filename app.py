#!/usr/bin/env python3
# TTS System for CPU-only Ubuntu Server
# Requirements: Python 3.12

import os
import sys
import argparse
import logging
import json
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from urllib.request import urlretrieve, build_opener, install_opener
from zipfile import ZipFile
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests
from gtts import gTTS
import tempfile
import io
from flask import Response, stream_with_context, Flask, request, jsonify, send_file, Blueprint, render_template
import soundfile as sf
import noisereduce as nr
from scipy import signal
import re
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("tts_system")

class TTSSystem:
    def __init__(self, output_dir="output", sample_rate=22050):
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.history_file = self.output_dir / "history.json"
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize history
        self._init_history()
    
    def _init_history(self):
        """Initialize or load history file"""
        if not self.history_file.exists():
            self.history = {"entries": []}
            self._save_history()
        else:
            try:
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
            except Exception as e:
                logger.error(f"Error loading history: {e}")
                self.history = {"entries": []}
    
    def _save_history(self):
        """Save history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def _add_to_history(self, text, output_file, settings):
        """Add an entry to history"""
        try:
            entry = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "text": text,
                "output_file": str(output_file),
                "settings": settings
            }
            self.history["entries"].append(entry)
            self._save_history()
        except Exception as e:
            logger.error(f"Error adding to history: {e}")
    
    def _apply_audio_effects(self, y, sr, effects):
        """Apply audio effects to the signal"""
        try:
            # Normalize volume using librosa
            if effects.get('normalize', True):
                y = librosa.util.normalize(y)
            
            # Reduce noise
            if effects.get('reduce_noise', True):
                y = nr.reduce_noise(y, sr=sr)
            
            # Add reverb
            if effects.get('reverb', False):
                reverb = np.random.normal(0, 0.1, len(y))
                y = y + reverb
            
            # Apply compression
            if effects.get('compress', False):
                y = np.clip(y, -0.5, 0.5)
            
            return y
        except Exception as e:
            logger.error(f"Error applying audio effects: {e}")
            return y
    
    def _preprocess_text(self, text):
        """Preprocess text for TTS"""
        try:
            # Remove special characters
            text = re.sub(r'[^\w\s.,!?-]', '', text)
            
            # Handle SSML tags if present
            if '<speak>' in text:
                # Basic SSML support
                text = re.sub(r'<[^>]+>', '', text)
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text
    
    def text_to_speech(self, text, output_file=None, speed=1.0, stream=False, 
                      format='wav', effects=None, lang='en', voice_name=None):
        """
        Convert text to speech with enhanced features
        
        Args:
            text (str): Text to convert to speech
            output_file (str): Path to save audio file
            speed (float): Speech speed multiplier
            stream (bool): Whether to return a streaming response
            format (str): Output format (wav, mp3, ogg)
            effects (dict): Audio effects to apply
            lang (str): Language code (e.g., 'en', 'es', 'fr')
            voice_name (str): Deprecated parameter, use lang instead
            
        Returns:
            str or Response: Path to the generated audio file or streaming response
        """
        try:
            # Handle backward compatibility with voice_name
            if voice_name is not None:
                logger.warning("voice_name parameter is deprecated, use lang instead")
                # Convert voice_name to language code if needed
                lang = voice_name if len(voice_name) == 2 else 'en'
            
            # Preprocess text
            text = self._preprocess_text(text)
            
            # Create a temporary file for gTTS output
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Generate speech using gTTS with specified language
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(temp_path)
            
            # Convert to WAV format and resample if needed
            y, sr = librosa.load(temp_path, sr=None)
            if sr != self.sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            
            # Apply audio effects
            if effects:
                y = self._apply_audio_effects(y, self.sample_rate, effects)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            if stream:
                # Create an in-memory buffer for the audio file
                buffer = io.BytesIO()
                sf.write(buffer, y, self.sample_rate, format=format.upper())
                buffer.seek(0)
                
                # Create a streaming response
                return Response(
                    stream_with_context(buffer),
                    mimetype=f'audio/{format}',
                    headers={
                        'Content-Disposition': 'inline',
                        'Content-Type': f'audio/{format}'
                    }
                )
            else:
                # Save to file if output_file is provided
                if output_file is None:
                    short_text = text[:20].replace(" ", "_").lower()
                    output_file = self.output_dir / f"tts_{short_text}.{format}"
                else:
                    output_file = Path(output_file)
                
                sf.write(str(output_file), y, self.sample_rate, format=format.upper())
                
                # Add to history
                self._add_to_history(text, output_file, {
                    "speed": speed,
                    "format": format,
                    "effects": effects,
                    "language": lang
                })
                
                return str(output_file)
            
        except Exception as e:
            logger.error(f"Error in text-to-speech conversion: {e}")
            raise
    
    def batch_process(self, input_files, output_dir=None, format='wav', effects=None):
        """
        Process multiple text files
        
        Args:
            input_files (list): List of input file paths
            output_dir (str): Directory to save output files
            format (str): Output format
            effects (dict): Audio effects to apply
            
        Returns:
            list: Paths to generated audio files
        """
        try:
            if output_dir is None:
                output_dir = self.output_dir / "batch"
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            output_files = []
            for input_file in tqdm(input_files, desc="Processing files"):
                try:
                    with open(input_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    if not text:
                        logger.warning(f"Empty file: {input_file}")
                        continue
                    
                    output_file = output_dir / f"{Path(input_file).stem}.{format}"
                    result = self.text_to_speech(
                        text,
                        output_file=output_file,
                        format=format,
                        effects=effects
                    )
                    output_files.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing {input_file}: {e}")
                    continue
            
            return output_files
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise
    
    def get_history(self, limit=10):
        """Get recent history entries"""
        try:
            return self.history["entries"][-limit:]
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []
    
    def clear_history(self):
        """Clear history"""
        try:
            self.history = {"entries": []}
            self._save_history()
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            raise
    
    def visualize_speech(self, audio_file, output_file=None):
        """
        Create a visualization of the speech audio
        
        Args:
            audio_file (str): Path to audio file
            output_file (str): Path to save visualization
            
        Returns:
            str: Path to visualization file
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_file, sr=None)
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Plot waveform
            plt.subplot(2, 1, 1)
            plt.title('Waveform')
            librosa.display.waveshow(y, sr=sr)
            
            # Plot spectrogram
            plt.subplot(2, 1, 2)
            plt.title('Spectrogram')
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            
            # Set title
            plt.suptitle(f"Speech Visualization: {Path(audio_file).name}")
            plt.tight_layout()
            
            # Create output filename if not provided
            if output_file is None:
                output_file = str(Path(audio_file).with_suffix('.png'))
                
            # Save figure
            plt.savefig(output_file)
            plt.close()
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error in speech visualization: {e}")
            return None
    
    def list_voices(self):
        """List available languages/voices"""
        try:
            # Return a list of common languages supported by gTTS
            return [
                {"id": "en", "name": "English"},
                {"id": "es", "name": "Spanish"},
                {"id": "fr", "name": "French"},
                {"id": "de", "name": "German"},
                {"id": "it", "name": "Italian"},
                {"id": "pt", "name": "Portuguese"},
                {"id": "ru", "name": "Russian"},
                {"id": "ja", "name": "Japanese"},
                {"id": "ko", "name": "Korean"},
                {"id": "zh", "name": "Chinese"}
            ]
        except Exception as e:
            logger.error(f"Error listing voices: {e}")
            return [{"id": "en", "name": "English"}]  # Fallback to English

# Initialize Flask app
app = Flask(__name__)
tts_system = TTSSystem()

# Web interface route
@app.route('/')
def index():
    return render_template('index.html')

# Web interface TTS endpoint
@app.route('/tts', methods=['POST'])
def text_to_speech():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        text = data.get('text')
        response_format = 'mp3'  # Always use MP3 format

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Generate speech
        try:
            # Create a temporary file for gTTS output
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name

            # Generate speech using gTTS (always in English)
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(temp_path)

            # Convert to WAV format and resample if needed
            y, sr = librosa.load(temp_path, sr=None)
            if sr != tts_system.sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=tts_system.sample_rate)

            # Create an in-memory buffer for the audio file
            buffer = io.BytesIO()
            sf.write(buffer, y, tts_system.sample_rate, format=response_format.upper())
            buffer.seek(0)

            # Clean up temporary file
            os.unlink(temp_path)

            # Return the audio file
            return send_file(
                buffer,
                mimetype=f'audio/{response_format}',
                as_attachment=True,
                download_name=f'speech.{response_format}'
            )

        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

# OpenAI-compatible endpoints
@app.route('/v1/models', methods=['GET'])
def get_models():
    try:
        # Return available models
        models = [
            {
                "id": "tts-1",
                "name": "TTS-1",
                "description": "Standard TTS model",
                "voice_count": 6,
                "permission": [
                    {
                        "id": "model_permission-1",
                        "object": "model_permission",
                        "created": 1683912666,
                        "allow_create_engine": True,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": True,
                        "allow_view": True,
                        "allow_fine_tuning": True,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False
                    }
                ],
                "root": "tts-1",
                "parent": None
            }
        ]
        return jsonify({"data": models})
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/v1/audio/speech', methods=['POST'])
def openai_tts():
    try:
        # Log the incoming request
        logger.info("Received TTS request")
        logger.info(f"Path: {request.path}")
        logger.info(f"Headers: {dict(request.headers)}")
        logger.info(f"Data: {request.get_data()}")
        
        # Parse request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Extract parameters
        text = data.get('input')
        voice = data.get('voice', 'alloy')
        model = data.get('model', 'tts-1')
        response_format = data.get('response_format', 'mp3')

        if not text:
            return jsonify({"error": "No input text provided"}), 400

        # Map OpenAI voices to language codes
        voice_map = {
            'alloy': 'en',
            'echo': 'en',
            'fable': 'en',
            'onyx': 'en',
            'nova': 'en',
            'shimmer': 'en'
        }

        # Check if the voice is supported
        if voice not in voice_map:
            return jsonify({"error": f"Voice '{voice}' is not supported"}), 400

        # Convert voice to language code
        lang = voice_map.get(voice, 'en')

        # Generate speech
        try:
            # Create a temporary file for gTTS output
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name

            # Generate speech using gTTS
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(temp_path)

            # Convert to WAV format and resample if needed
            y, sr = librosa.load(temp_path, sr=None)
            if sr != tts_system.sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=tts_system.sample_rate)

            # Create an in-memory buffer for the audio file
            buffer = io.BytesIO()
            sf.write(buffer, y, tts_system.sample_rate, format=response_format.upper())
            buffer.seek(0)

            # Clean up temporary file
            os.unlink(temp_path)

            # Return the audio file
            return send_file(
                buffer,
                mimetype=f'audio/{response_format}',
                as_attachment=True,
                download_name=f'speech.{response_format}'
            )

        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Log all registered routes
    for rule in app.url_map.iter_rules():
        logger.info(f"Registered route: {rule}")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
