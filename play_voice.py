import os

# Library text to audio
from gtts import gTTS

# Library to play audio files
from pydub import AudioSegment
from pydub.playback import play

def play_voice(mText, lang="en"):
    """Function used to play the string 'mText' in audio using tts"""
    print(f"[play_voice] now playing: '{mText}'")
    tts_audio = gTTS(text=mText, lang=lang, slow=False)

    tts_audio.save("audio_recognition/voice.wav")
    play(AudioSegment.from_file("audio_recognition/voice.wav"))
    os.remove("audio_recognition/voice.wav")