import pyaudio
import wave
import os
import Properties as prop


def play_sound_wav(filename):

    # define stream chunk
    chunk = 1024
    audio_path = os.path.join(prop.dir_resource_base, filename)
    f = wave.open(audio_path,"rb")
    # instantiate PyAudio
    p = pyaudio.PyAudio()

    # open stream
    stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                    channels=f.getnchannels(),
                    rate=f.getframerate(),
                    output=True)
    # read data
    data = f.readframes(chunk)

    while data:
        stream.write(data)
        data = f.readframes(chunk)

    # stop stream and terminate
    stream.stop_stream()
    stream.close()
    p.terminate()
