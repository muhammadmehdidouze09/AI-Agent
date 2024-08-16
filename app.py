from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
from pydub import AudioSegment
import io
import concurrent.futures
from functools import lru_cache
import pickle
from gtts import gTTS
import base64

# Initialize Flask app
app = Flask(__name__)

# Load chatbot pipeline
with open('chatbot_pipeline.pkl', 'rb') as file:
    chatbot_pipe = pickle.load(file)

# Initialize gTTS for TTS
def convert_text_to_speech_gtts(response_text):
    # Create an in-memory BytesIO object for MP3 audio
    mp3_audio_io = io.BytesIO()
    
    # Use gTTS to generate the audio file
    tts = gTTS(text=response_text, lang='en')
    tts.write_to_fp(mp3_audio_io)
    
    # Rewind the in-memory MP3 file
    mp3_audio_io.seek(0)

    return mp3_audio_io

@app.route('/')
def index():
    return render_template('index.html')

def recognize_speech(wav_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(wav_file) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio"
    except sr.RequestError as e:
        return f"Sorry, there was an error with the speech recognition service: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@lru_cache(maxsize=100)
def get_chatbot_response(text):
    return chatbot_pipe.predict([text])[0]

def process_audio_task(audio_data):
    # Convert audio data to wav format
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
        wav_file = io.BytesIO()
        audio_segment.export(wav_file, format="wav")
        wav_file.seek(0)
    except Exception as e:
        return {"error": f"Audio file could not be processed: {str(e)}"}

    # Recognize speech
    recognized_text = recognize_speech(wav_file)
    print(f"Recognized Text: {recognized_text}")

    # If no valid text is recognized, respond with a default message
    if recognized_text.strip() in ["Sorry, I could not understand the audio", "Sorry, there was an error with the speech recognition service", "An unexpected error occurred"]:
        response_text = "Couldn't understand"
    else:
        # Get chatbot response
        response_text = get_chatbot_response(recognized_text)
    print(f"Chatbot Response: {response_text}")

    # Convert chatbot response to speech
    response_audio = convert_text_to_speech_gtts(response_text)

    return {"response_text": response_text, "response_audio": response_audio}

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    audio_data = audio_file.read()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(process_audio_task, audio_data)
        result = future.result()

    if "error" in result:
        return jsonify(result), 400

    # Convert audio to Base64
    audio_bytes = result["response_audio"].read()
    encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')

    return jsonify({
        "response_text": result["response_text"],  # The chatbot's response text
        "response_audio": encoded_audio  # Return Base64 encoded audio
    })

if __name__ == "__main__":
    app.run(debug=True)
