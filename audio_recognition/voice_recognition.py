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
            
    def list_elements(self, list_e, translated_dict=None):
        list_elem = ""
        for item in list_e:
            if (self.language == "pt-br"):
                list_elem += str(translated_dict[item]) + ", "
                continue
            list_elem += str(item) +", "
        if (self.language == "pt-br"):
            play_voice(list_elem, self.language[:2])
            return
        play_voice(list_elem, self.language)
        
    def list_categories(self, dictionary, translated_dict=None):
        list_categories = ""
        for item in list(dictionary.keys()):
            if (self.language == "pt-br"):
                list_categories += str(translated_dict[item]) + ", "
                continue
            list_categories += str(item) +", "
        if (self.language == "pt-br"):
            play_voice(list_categories, self.language)[:2]
            return
        play_voice(list_categories, self.language)

    def repeat(self, typeof):
        if (self.language == "pt-br"):
            play_voice(f"Qual {typeof} você quer?", self.language[:2])
            return
        play_voice("Which {} do you want?".format(typeof), self.language)

    def greetings(self):
        if (self.language == "pt-br"):
            play_voice("Modo Query ativado. Qual categoria você gostaria de procurar?", self.language[:2])
            return
        play_voice("Query mode activated. Which category do you want?", self.language)
    