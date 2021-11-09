import speech_recognition as sr
import os
import time
from .recording import record_to_file
from play_voice import play_voice


class VoiceRecognition():
    def __init__(self, language="en"):
        self.language = language

    def speech_recog(self):
        mic = sr.Recognizer()

        with sr.AudioFile("audio_recognition/output.wav") as source:
            audio = mic.record(source)

            try:
                word = mic.recognize_google(audio, language=self.language)
                os.remove("audio_recognition/output.wav")
                return word.lower()

            except Exception as e:
                print(e)
                return None
            
    def list_elements(self, list_e):
        list_elem = ""
        for item in list_e:
            list_elem += str(item) +", "
        play_voice(list_elem, self.language)
        
    def list_categories(self, dictionary):
        list_categories = ""
        for item in list(dictionary.keys()):
            list_categories += str(item) +", "
        play_voice(list_categories, self.language)

    def repeat(self, typeof):
        play_voice("Which {} do you want?".format(typeof), self.language)

    def greetings(self):
        play_voice("Query mode activated. Which category do you want?", self.language)
    