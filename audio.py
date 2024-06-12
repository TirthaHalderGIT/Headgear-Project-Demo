import speech_recognition as sr
import pyttsx3

def get_threshold_from_audio():
    recognizer = sr.Recognizer()
    engine = pyttsx3.init()

    # Provide instructions via speech
    engine.say("Please say the threshold value.")
    engine.runAndWait()

    with sr.Microphone() as source:
        print("Listening...")
        audio_data = recognizer.listen(source, timeout=5)

    # Convert audio data to text
    try:
        threshold_text = recognizer.recognize_google(audio_data, language="en-US")
        threshold = float(threshold_text)
        print("Threshold value received:", threshold)
        return threshold
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print("Error retrieving audio: {0}".format(e))
        return None
    except ValueError:
        print("Invalid threshold value received.")
        return None

# Example usage:
threshold = get_threshold_from_audio()
if threshold is not None:
    print("Selected threshold:", threshold)
else:
    print("No threshold value received.")