# TTS System

A lightweight text-to-speech system with a web UI and API endpoints for CPU-only Ubuntu servers.

## Features

- Simple web UI for text-to-speech conversion
- RESTful API for programmatic access
- OpenAI-compatible API endpoints
- Support for multiple languages
- Audio effects (normalization, noise reduction, reverb, compression)
- Batch processing of text files
- Audio visualization
- History tracking of generated speech

## Requirements

- Python 3.12
- Ubuntu Server (CPU-only, no GPU required)
- The following Python packages:
  - Flask
  - gTTS
  - librosa
  - soundfile
  - noisereduce
  - matplotlib
  - tqdm
  - numpy
  - scipy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tts-system.git
   cd tts-system
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the Server

Run the main script to start the Flask server:

```bash
python app.py
```

By default, the server will run on port 5000. You can access the web UI at `http://localhost:5000`.

### Web UI

The web UI provides a simple interface for text-to-speech conversion:
1. Enter your text in the textarea
2. Click "Generate Speech"
3. Listen to the generated audio through the audio player

### API Endpoints

#### Local API

- `POST /tts`
  - Converts text to speech
  - Request body: `{"text": "Your text here", "lang": "en"}`
  - Returns: Audio file (MP3)

#### OpenAI-Compatible API

The system provides an OpenAI-compatible endpoint for easy integration with existing applications:

- `POST /v1/audio/speech`
  - Converts text to speech using OpenAI-compatible parameters
  - Request body:
    ```json
    {
      "input": "Your text here",
      "voice": "alloy",
      "model": "tts-1",
      "response_format": "mp3"
    }
    ```
  - Returns: Audio file in the specified format

### Programmatic Usage

You can use the `TTSSystem` class directly in your Python code:

```python
from tts_system import TTSSystem

# Initialize the system
tts = TTSSystem()

# Basic text-to-speech
output_file = tts.text_to_speech("Hello, world!", format="mp3")
print(f"Audio saved to: {output_file}")

# With effects
effects = {
    "normalize": True,
    "reduce_noise": True,
    "reverb": True,
    "compress": True
}
output_file = tts.text_to_speech(
    "Hello with effects!",
    format="wav",
    effects=effects
)

# Batch processing
input_files = ["file1.txt", "file2.txt"]
output_files = tts.batch_process(input_files, format="mp3")
```

## Available Languages

The system supports the following languages but only English is implemented in the basic code example:
- English (en)
- Spanish (es) 
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Russian (ru)
- Japanese (ja)
- Korean (ko)
- Chinese (zh)

## Configuration

The system uses reasonable defaults, but you can modify them:

- `output_dir`: Directory for saved audio files (default: "output")
- `sample_rate`: Audio sample rate (default: 22050 Hz)
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
