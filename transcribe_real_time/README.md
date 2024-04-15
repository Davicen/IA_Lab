# Real-Time Speech Transcription with Whisper

Real-Time Speech Transcription with Whisper is a Python project that transcribes speech in real-time using the Whisper speech recognition library.

## Features

- Supports multiple models for transcription, including tiny, base, small, medium, and large.
- Allows customization of energy level for microphone detection.
- Provides options to adjust recording and phrase timeout durations.
- Automatically adjusts for ambient noise during recording.
- Displays transcription in real-time, updating as speech is detected.

## Prerequisites

- Python 3.7 or higher
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/real-time-speech-transcription.git
cd real-time-speech-transcription
```

## Usage

Run the main.py script with appropriate command-line arguments:

```bash

python main.py --model small --energy_lvl 1000 --record_timeout 2 --phrase_timeout 3
```

Available command-line arguments:

    --model: Specify the model to use for transcription (choices: tiny, base, small, medium, large).
    --energy_lvl: Set the energy level for microphone detection.
    --record_timeout: Set the duration of real-time recording in seconds.
    --phrase_timeout: Set the duration of empty space between recordings to consider it a new line in the transcription.

Note: Additional instructions for Linux users are provided in the script.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
