from logging import captureWarnings
import os, time, argparse, json, queue

# Threading
import threading as Thread

# GPIO - Pi Buttons
from gpiozero import Button

# Audio
# from play_voice import play_voice
##############################################
from logging import FileHandler
import os
from threading import Thread

# Library text to audio
from gtts import gTTS

# Library to play audio files
from pydub import AudioSegment
from pydub.playback import play
#######################################################
from audio_recognition.recording import record_to_file
from audio_recognition.voice_recognition import VoiceRecognition

# TFLite detection
from TFLite_detection_webcam import initialize_detector, safari_mode, query_mode

# Text recognition
from text_recognition.opencv_text_detection.text_detection import main_text_detection

class VMobi:
    """Class that represents the system as a whole"""

    def __init__(self, args, ptbr_categ):
        self.args = args # Saving arguments
        self.MODEL_DIR = args.modeldir # Directory of the .tflite file and names file
        self.RESOLUTION = args.resolution # Camera resolution in pixels
        self.USE_EDGETPU = args.edgetpu # Flag to use the google coral tpu
        self.lang = args.lang # Language used on tts speech voice
        self.tts_lang = args.lang[:2]
        self.east_model_path = os.getcwd() + "/text_recognition/east_model_float16.tflite" # EAST .tflite path for text recognition
        self.ptbr_categ = ptbr_categ
        self.pt_to_en_categs = {v: k for k, v in ptbr_categ.items()}
        # Thread(target=self.multithreading_queue_checker).start()
        self.main() # Runs on the raspberry with buttons on the GPIO


    def main(self):
        """Main function that orchestrates the product"""
        print("On main function!")

        # Get a list of the categories as strings
        self.categories = self.get_all_categories()
        print(f"Got all categories: {self.categories}")

        # Conect button on GPIO2 and Ground
        # Watch out for connenctions in 'pin_layout.svg'
        self.query_button = Button(2, hold_time=2)

        # Running the safari mode to run on the background
        # thread_safari_mode = threading.Thread(target=initialize_detector, args=(self.args,))
        # thread_safari_mode.start()
        detector_args = initialize_detector(self.args)
            
        while (True):
            s = safari_mode(detector_args, self.query_button, fila=fila, lang=self.lang, ptbr_categ=self.ptbr_categ)
            if s > 0:
                # Enter Query Mode
                fila.queue.clear() # Clear the queue to get right on the query mode
                query_cat = self.query_mode_voice_type() # Get the category with voice command
                if (self.tts_lang == "pt"):
                    query_cat = self.pt_to_en_categs[query_cat]
                if query_cat == 'text':
                    main_text_detection(self.east_model_path, self.query_button)
                    continue
                else:
                    query_mode(detector_args, query_cat, query_btn=self.query_button, fila=fila, lang=self.lang, ptbr_categ=self.ptbr_categ)
                    continue
            

    def query_mode_voice_type(self):
        """Query  mode that uses voice recognition and only the query button"""
        print("Entering query mode with voice recognition. (Type 2)")
        qmode = VoiceRecognition(language=self.lang)
        qmode.greetings(fila)

        while eve.is_set():
            eve.wait(1)

        if (not eve.is_set()):
            record_to_file("audio_recognition/output.wav")
            categ = qmode.speech_recog()
        
        
        while categ == None or categ == "list" or categ == "least" or (categ not in self.categories) or (categ not in self.pt_to_en_categs.keys()):
            print(categ)
            if categ == None:
                qmode.repeat("category", fila)
                record_to_file("audio_recognition/output.wav")
                # categ = qmode.speech_recog()
            elif categ == "list" or categ == "least" or categ == "lista" or categ == "categorias":
                qmode.list_elements(self.categories, fila, self.ptbr_categ)
                categ = None
            elif categ == 'text' or categ == "texto":
                if (self.tts_lang == "pt"):
                    # play_voice("Você escolheu a categoria de texto. Iniciando o reconhecimento.")
                    fila.put("Você escolheu a categoria de texto. Iniciando o reconhecimento.")
                else:
                    # play_voice("You chose text category. Start recognizing")
                    fila.put("You chose text category. Start recognizing")
                return 'text'
            else:
                if (self.tts_lang == "pt"):
                    # play_voice("Categoria não está no dataset. Qual categoria você gostaria de procurar?")
                    fila.put("Categoria não está no dataset. Qual categoria você gostaria de procurar?")
                else:
                    # play_voice("Category not in dataset. Which category do you want?")
                    fila.put("Category not in dataset. Which category do you want?")
                    
                while eve.is_set():
                    eve.wait(1)

                if (not eve.is_set()):
                    record_to_file("audio_recognition/output.wav")

            categ = qmode.speech_recog()
                
        if (self.tts_lang == "pt"):
            # play_voice(f"Você escolheu a categoria: {self.ptbr_categ[categ]}", self.tts_lang)
            fila.put(f"Você escolheu a categoria: {self.ptbr_categ[categ]}")
        else:
            # play_voice(f"You chose the category: {categ}", self.lang)
            fila.put(f"You chose the category: {categ}")
        return categ
    

    def get_all_categories(self):
        """Function that get all available categories from model '.name' file"""
        for root, dir, files in os.walk(self.MODEL_DIR):
            for f in files:
                if "labelmap.txt" in f:
                    filename = f
                    break
        cat = []

        f = open(self.MODEL_DIR + filename, "r")
        for line in f.readlines():
            if "?" in line:
                continue
            cat.append(line.replace("\n", ""))
        return cat
    
def play_voice(mText, lang="en"):
    """Function used to play the string 'mText' in audio using tts"""
    print(f"[play_voice] now playing: '{mText}'")
    tts_audio = gTTS(text=mText, lang=lang, slow=False)

    tts_audio.save("audio_recognition/voice.wav")
    play(AudioSegment.from_file("audio_recognition/voice.wav"))
    os.remove("audio_recognition/voice.wav")

def multithreading_queue_checker(lang):
    global fila, sem
    while (True):
        if not fila.empty():
            eve.set()
            a = fila.get()
            print(f"[QUEUE CHECKER] Reading now: {a}")
            play_voice(a, lang)
        eve.clear()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in', 
                        default="Sample_TFLite_model/")
    parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                        default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                        default='labelmap.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                        default=0.5)
    parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                        default='1280x720')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                        action='store_true')
    parser.add_argument('--safari', help='Start Safari Mode', action='store_true')
    parser.add_argument('--query', help='Start Query Mode', default='?')
    parser.add_argument('--lang', help='Choose the speech language (e.g. "en", "pt-br", ...)', default='en')

    args = parser.parse_args()

    with open('locales/ptbr.json') as json_file:
        ptbr_categ = json.load(json_file)

    global fila
    fila = queue.Queue()

    global eve
    eve = Thread.Event()
    Thread(target=multithreading_queue_checker, args=(args.lang[:2],)).start()
    # Thread(target=thread_check, args=("pt",)).start()

    helper = VMobi(args, ptbr_categ)
