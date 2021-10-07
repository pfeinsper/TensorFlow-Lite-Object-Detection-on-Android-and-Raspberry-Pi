import speech_recognition as sr
import os
import time
from pydub import AudioSegment
from pydub.playback import play
from gtts import gTTS

class QueryMode():
    def __init__(self, language="en"):
        self.language = language

    def speech_recog(self):
        mic = sr.Recognizer()

        with sr.Microphone(device_index=2) as source:
            mic.adjust_for_ambient_noise(source)
            duration = 0.5
            freq = 440

           # os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

            audio = mic.listen(source, timeout=4)

            try:
                word = mic.recognize_google(audio, language=self.language)

                return word.lower()

            except:
                return None


    def list_categories(self, dictionary):
        list_categories = ""
        for item in list(dictionary.keys()):
            list_categories += str(item) +", "


    def play_voice(self, mText):
        tts_audio = gTTS(text=mText, lang=self.language, slow=False)
        tts_audio.save("voice.wav")
        play(AudioSegment.from_file("voice.wav"))
        os.remove("voice.wav")

    def greetings(self):
        self.play_voice("Query mode is activated. Which category do you want?")

qmode = QueryMode()
qmode.greetings()
print(qmode.speech_recog())
