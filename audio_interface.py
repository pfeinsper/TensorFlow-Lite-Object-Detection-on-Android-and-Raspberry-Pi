import pyttsx3
import speech_recognition as sr
# import pyaudio
from gtts import gTTS
from playsound import playsound
import os
import time


class QueryMode():
    def __init__(self, language):
        self.language = language

    # Função de HOLD do botão
    """def button_hold(self):"""

    def speak(self, output):
        myobj = gTTS(text=output, lang=self.language, slow=False)
        myobj.save("output.mp3")
        playsound("output.mp3")
        os.remove("output.mp3")

    def speech_recog(self):
        # Habilita o microfone do usuário
        mic = sr.Recognizer()

        # microphone handling
        with sr.Microphone() as source:
            mic.adjust_for_ambient_noise(source)
            # Beep sound
            duration = 0.5  # seconds
            freq = 440  # Hz
            os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

            # Armazena o que foi dito numa variavel
            audio = mic.listen(source)

            try:
                # Passa a variável para o algoritmo reconhecedor de padroes
                word = mic.recognize_google(audio, language='en')
                # Retorna a frase pronunciada
                return word.lower()

            # # Se nao reconheceu o padrao de fala, exibe a mensagem
            except:
                return None

    def greet(self):
        self.speak("Query mode is activated!")

    def shutdown(self):
        self.speak("Deactivating query mode.")
        exit()


if __name__ == "__main__":
    categories_elements = {
        'food': ['apple', 'orange', 'spaghetti'], 'automobiles': ['car', 'motorcycle']}

    querymode = QueryMode('en')

    # If query mode was pressed
    querymode.greet()
    category = None

    while category == None or category == 'list':
        querymode.speak("Which category do you want? Say after beep sound.")
        category = querymode.speech_recog()
        print(category)
        if category == 'list' or category == 'least':
            list_categories = categories_elements.keys()
            querymode.speak(', '.join(list(list_categories)))
        elif category == None:
            querymode.speak(f'Sorry, you did say nothing.')
            category = None
        else:
            if category in list(categories_elements.keys()):
                querymode.speak(f'Category {category} selected.')
            else:
                querymode.speak(f'Sorry, category does not exist.')
                category = None

    element = None
    while element == None or element == 'list':
        querymode.speak(
            f"Which element do you want into {category} category? Say after beep sound.")
        element = querymode.speech_recog()
        print(element)
        if element == 'list' or element == 'least':
            querymode.speak(', '.join(categories_elements[category]))
        elif element == None:
            querymode.speak(f'Sorry, you did say nothing.')
            element = None
        else:
            if element in categories_elements[category]:
                querymode.speak(f'element {element} selected.')
            else:
                querymode.speak(f'Sorry, element does not exist.')
                element = None

    # if word in list(categories_elements.keys()):


# sudo apt install sox
