import speech_recognition as sr
from gtts import gTTS
import os
from openai import OpenAI
from putergenai import PuterClient
import tempfile
import sounddevice as sd
import soundfile as sf
import numpy as np

# try:
#     from secrets import PuterClientUSERNAME, PuterClientPASSWORD
# except ImportError:
#     print("Please create a secrets.py file with your OPENROUTER_API_KEY")
#     exit(1)

# Initialize OpenAI client
# client = OpenAI(
# #     base_url="https://openrouter.ai/api/v1",
#     api_key="sk-OsMMq65tXdfOIlTUYtocSL7NCsmA7CerN77OkEv29dODg1EA",
# )
# client = PuterClient()
# client.login("your_username", "your_password")
# Initialize the recognizer
recognizer = sr.Recognizer()

# def analyze_joke(joke):
#     """Analyze how funny a joke is using GPT"""
#     try:
#         # response = client.chat.completions.create(
#         #     model="gpt-3.5-turbo",
#         #     messages=[
#         #         {"role": "system", "content": "You are a comedy expert. Rate Hebrew jokes on a scale of 1-10. Return only a number (the rate)."},
#         #         {"role": "user", "content": f"Rate this joke in Hebrew on a scale of 1 to 10. Return only a number of the rate, no other text: {joke}"}
#         #     ]
#         # )
#         # return response.choices[0].message.content
#         # Call ai.chat or relevant API based on Puter SDK docs
#         response = client.ai.chat(
#             messages=[
#                 {"role": "system", "content": "את/ה מומחה/ית לקומדיה. דרג/י בדיחות בעברית מ‑1 עד 10 והסבר בקצרה בעברית."},
#                 {"role": "user", "content": f"דרג/י את הבדיחה הזאת: {joke}"}
#             ]
#         )

#         # Access the response from the correct API call
#         return response['message']['content']  
#     except Exception as e:
#         print(f"Error analyzing joke: {e}")
#         return "לא הצלחתי להעריך את הבדיחה, אבל אני בטוח שהיא מצחיקה!"

def speak_text(text):
    """Convert text to speech and play it"""
    temp_file = None
    try:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        
        # Generate speech
        tts = gTTS(text=text, lang='iw', slow=False)
        tts.save(temp_file.name)
        
        # Read the audio file
        audio_data, sample_rate = sf.read(temp_file.name)
        
        # Close the file before playing
        temp_file.close()
        
        # Play the audio using sounddevice
        sd.play(audio_data, sample_rate)
        sd.wait()  # Wait until the audio is finished playing
    finally:
        # Clean up the temporary file
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                print(f"Warning: Could not delete temporary file: {e}")

def recognize_speech():
    """Capture and recognize speech in Hebrew"""
    # You can change this device index based on list_microphones() output
    device_index = 3  # Change this number based on your microphone
    with sr.Microphone(device_index=1) as source:
        print("Listening...")
        # Adjust the recognizer sensitivity to ambient noise
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        # Configure the recognizer
        recognizer.pause_threshold = 2.0  # Wait for 3 seconds of silence before stopping
        recognizer.phrase_threshold = 0.3  # Minimum length of silence to consider the phrase complete
        recognizer.non_speaking_duration = 1.0  # Minimum length of silence after speaking
        
        # Listen for audio input
        audio = recognizer.listen(source, timeout=None, phrase_time_limit=None)
        
        try:
            # Using Google's speech recognition with Hebrew language
            text = recognizer.recognize_google(audio, language='he-IL')
            print("You said:", text)
            return text
        except sr.UnknownValueError:
            print("Could not understand the audio")
            return None
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
            return None

# Main interaction loop
def main():
    user_name = None
    
    # Initial greeting and ask for name
    print("שלום, אני מושי הרובוט הרגיש. תספור בלב עד שלוש ואז תאמר לי - מה השם בבקשה?")
    speak_text("שלום, אני מושי הרובוט הרגיש. מה השם בבקשה?")
    
    while not user_name:
        text = recognize_speech()
        if text:
            user_name = text
            speak_text(f"נעים להכיר {user_name}! אני שמח שהגעת")
            with open("user_name.txt", "w", encoding="utf-8") as f:
                f.write(user_name)
        else:
            speak_text("מה השם בבקשה?")
    
    # Ask for a joke
    print((f"אני רוצה לשמוע בדיחה"))
    speak_text(f"אני רוצה לשמוע בדיחה")
    
    joke = None
    while not joke:
        text = recognize_speech()
        if text:
            joke = text
            print("הממממ אני צריך לחשוב על זה רגע...")
            speak_text("קורע!!")
            
            # Analyze the joke
            analysis = analyze_joke(joke)
            
            # Save the joke and its analysis
            with open("jokes.txt", "a", encoding="utf-8") as f:
                f.write(f"{user_name}: {joke}\nAnalysis: {analysis}\n\n")
            
            # Speak the analysis
            speak_text(analysis)
        else:
            print((f"אני רוצה לשמוע בדיחה"))
    
    while True:
        # Get speech input
        # text = recognize_speech()
        # if text:
        #     # Echo back what was heard
        #     speak_text("שמעתי: " + text)
            
        # Optional: add a way to exit the loop
        if text and "ביי" in text:
            speak_text(f"להתראות {user_name}!")
            break

if __name__ == "__main__":
    main()


