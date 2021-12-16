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
            
    def list_elements(self, list_e, fila=None, translated_dict=None):
        list_elem = ""
        for item in list_e:
            if (self.language == "pt-br"):
                list_elem += str(translated_dict[item]) + ", "
                continue
            list_elem += str(item) +", "
        if (self.language == "pt-br"):
            play_voice(list_elem, self.language[:2])
            # fila.put(list_elem)
            return
        play_voice(list_elem, self.language)
        # fila.put(list_elem)
        
    def list_categories(self, dictionary, fila=None, translated_dict=None):
        list_categories = ""
        for item in list(dictionary.keys()):
            if (self.language == "pt-br"):
                list_categories += str(translated_dict[item]) + ", "
                continue
            list_categories += str(item) +", "
        if (self.language == "pt-br"):
            play_voice(list_categories, self.language)[:2]
            # fila.put(list_categories)
            return
        play_voice(list_categories, self.language)
        # fila.put(list_categories)

    def repeat(self, typeof, fila=None):
        if (self.language == "pt-br"):
            play_voice(f"Qual categoria você gostaria de buscar?", self.language[:2])
            # fila.put(f"Qual categoria você gostaria de buscar?")
            return
        play_voice("Which {} do you want?".format(typeof), self.language)
        # fila.put("Which {} do you want?".format(typeof))

    def greetings(self, fila=None):
        if (self.language == "pt-br"):
            play_voice("Modo Query ativado. Qual categoria você gostaria de procurar?", self.language[:2])
            # fila.put("Modo Query ativado. Qual categoria você gostaria de procurar?")
            return
        play_voice("Query mode activated. Which category do you want?", self.language)
        # fila.put("Query mode activated. Which category do you want?")
    