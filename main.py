from logging import captureWarnings
import os, time, argparse, subprocess

# GPIO - Pi Buttons
from gpiozero import Button

# Audio
# from play_voice import play_voice
from audio_recognition.recording import record_to_file
from audio_recognition.voice_recognition import VoiceRecognition

# TFLite detection
from TFLite_detection_webcam import initialize_detector, safari_mode, query_mode

# Text recognition
from text_recognition.opencv_text_detection.text_detection import main_text_detection

class VMobi:
    """Class that represents the system as a whole"""

    def __init__(self, args, lang = "en"):
        self.args = args # Saving arguments
        self.MODEL_DIR = args.modeldir # Directory of the .tflite file and names file
        self.RESOLUTION = args.resolution # Camera resolution in pixels
        self.USE_EDGETPU = args.edgetpu # Flag to use the google coral tpu
        self.lang = lang # Language used on tts speech voice
        self.east_model_path = os.getcwd() + "/text_recognition/east_model_float16.tflite" # EAST .tflite path for text recognition
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
            s = safari_mode(detector_args, self.query_button)
            if s > 0:
                # Enter Query Mode
                query_cat = self.query_mode_voice_type() # Get the category with voice command
                if query_cat == 'text':
                    main_text_detection(self.east_model_path, self.query_button)
                    continue
                else:
                    query_mode(detector_args, query_cat, query_btn=self.query_button)
                    continue
            

    def query_mode_voice_type(self):
        """Query  mode that uses voice recognition and only the query button"""
        print("Entering query mode with voice recognition. (Type 2)")
        qmode = VoiceRecognition()
        qmode.greetings()
        record_to_file("audio_recognition/output.wav")
        categ = qmode.speech_recog()
        
        while categ == None or categ == "list" or categ == "least" or (categ not in self.categories):
            print(categ)
            if categ == None:
                qmode.repeat("category")
                record_to_file("audio_recognition/output.wav")
                categ = qmode.speech_recog()
            elif categ == "list" or categ == "least":
                qmode.list_elements(self.categories)
                categ = None
            elif categ == 'text':
                subprocess.Popen(["python3", "play_voice.py", "--text='You chose text category. Start recognizing'"], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
                # play_voice("You chose text category. Start recognizing")
                return 'text'
            else:
                subprocess.Popen(["python3", "play_voice.py", "--text='Category not in dataset. Which category do you want?'"], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
                # play_voice("Category not in dataset. Which category do you want?")
                record_to_file("audio_recognition/output.wav")
                
        # play_voice(f"You chose the category: {categ}", self.lang)
        subprocess.Popen(["python3", "play_voice.py", f"--text='You chose the category: {categ}'"], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
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

    args = parser.parse_args()

    helper = VMobi(args)
