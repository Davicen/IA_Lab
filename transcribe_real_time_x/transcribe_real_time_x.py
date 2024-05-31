import argparse
import os
import numpy as np
import speech_recognition
import whisperx
import torch

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--energy_lvl", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=4,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    parser.add_argument("--time_debug", default=False,
                        help="Add delta time for the transcription duration", type=bool)
    
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_q = Queue()
    # Use SpeechRecognizer to record our audio. And turn off dynamic threshold
    recorder = speech_recognition.Recognizer()
    recorder.energy_threshold = args.energy_lvl
    recorder.dynamic_energy_threshold = False

    # Important for linux users.
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(speech_recognition.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(speech_recognition.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = speech_recognition.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = speech_recognition.Microphone(sample_rate=16000)

    # Load model
    model = args.model
    audio_model = whisperx.load_model(model, "cuda", compute_type="float16")

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:speech_recognition.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Get the data and push it on the Queue
        data = audio.get_raw_data()
        data_q.put(data)

    # Create a background thread that will pass us raw audio bytes.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\nReady.\n")

    result = audio_model.transcribe("C:\\Users\\vicee\\Desktop\\Grabacion.m4a" , language="es")

    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_q.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    print ('...')
                    # Flush stdout.
                    print('', end='', flush=True)
                    phrase_complete = True

                phrase_time = now
                
                # Combine audio data from queue
                audio_data = b''.join(data_q.queue)
                data_q.queue.clear()
                
                # Convert in-ram buffer to something the model can use directly without needing a temp file.
                # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
                # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Include marker from duration to obtain transcription times
                time_zero = now
                # Read the transcription.
                result = audio_model.transcribe(audio_np, language="es")#, fp16=torch.cuda.is_available())

                text = result['segments'][0]["text"]
                # Obtain duration of the transcription
                delta_duration = datetime.utcnow() -time_zero
                
                # Add duration at the end of the phrase if time debug enabled.
                if args.time_debug:
                    text = text + ' ' + str(delta_duration)

                # If we detected a pause between recordings, add a new item to our transcription.
                # Otherwise edit the existing one.
                '''
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text
                '''

                transcription.append(text)

                # Clear the console to reprint the updated transcription.
                os.system('cls' if os.name=='nt' else 'clear')
                for line in transcription:
                    print(line)

                # Flush stdout.
                print('', end='', flush=True)
            else:
                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)


if __name__ == "__main__":
    main()
