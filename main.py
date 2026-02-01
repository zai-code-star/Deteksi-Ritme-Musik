import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile
import glob, os, shutil, shlex
import plotly.graph_objects as go

import subprocess
import numpy as np
import tqdm

from test import *
from compare import *
import soundfile as sf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from flask import *  
app = Flask(__name__)  

@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  

@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':
        PATH = 'audio/'
        STATIC_PATH = 'static/'

        f = request.files['file']
        AUDIO_FILE = f.filename 
        f.save(os.path.join(PATH, f.filename))

        if request.files['file2']:
            f2 = request.files['file2']
            f2.save(os.path.join(PATH, f2.filename))
            similarity = correlate(os.path.join(PATH, f.filename), os.path.join(PATH, f2.filename))
        else:
            similarity = 'tidak file yang dibandingkan'
        
        predicts = predict(os.path.join(PATH, f.filename))
        
        FPS = 30
        FFT_WINDOW_SECONDS = 0.25
        FREQ_MIN = 10
        FREQ_MAX = 1000
        TOP_NOTES = 3
        RESOLUTION = (1280, 720)
        SCALE = 1

        NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        fs, data = wavfile.read(os.path.join(PATH, AUDIO_FILE))
        if len(data.shape) > 1:
            audio = data.T[0]  # Ambil channel pertama jika stereo
        else:
            audio = data  # Jika mono, gunakan data langsung

        FRAME_STEP = (fs / FPS)
        FFT_WINDOW_SIZE = int(fs * FFT_WINDOW_SECONDS)
        AUDIO_LENGTH = len(audio) / fs

        if os.path.isdir('content'):
            shutil.rmtree('content')
        os.makedirs('content')

        window = 0.5 * (1 - np.cos(np.linspace(0, 2*np.pi, FFT_WINDOW_SIZE, False)))

        xf = np.fft.rfftfreq(FFT_WINDOW_SIZE, 1/fs)
        FRAME_COUNT = int(AUDIO_LENGTH*FPS)
        FRAME_OFFSET = int(len(audio)/FRAME_COUNT)

        mx = 0
        for frame_number in range(FRAME_COUNT):
            sample = extract_sample(audio, frame_number, FRAME_OFFSET, FFT_WINDOW_SIZE)

            fft = np.fft.rfft(sample * window)
            fft = np.abs(fft).real 
            mx = max(np.max(fft), mx)

        for frame_number in tqdm.tqdm(range(FRAME_COUNT)):
            sample = extract_sample(audio, frame_number, FRAME_OFFSET, FFT_WINDOW_SIZE)

            fft = np.fft.rfft(sample * window)
            fft = np.abs(fft) / mx 
            s = find_top_notes(fft, TOP_NOTES, xf)

            fig = plot_fft(fft.real, xf, fs, s, FREQ_MIN, FREQ_MAX, RESOLUTION)
            fig.write_image(f"content/frame{frame_number}.png", scale=1)
          
            finalresult = os.path.splitext(f.filename)[0] + '.mp4'
            AUDIO_FILE2 = os.path.join(PATH, AUDIO_FILE)

        ffmpeg_path = "C:/ffmpeg/bin/ffmpeg.exe"
        output_path = os.path.join(STATIC_PATH, finalresult)
        command = f"{ffmpeg_path} -y -r {FPS} -f image2 -s 1280x720 -i content/frame%d.png -i {AUDIO_FILE2} -c:v h264 -pix_fmt yuv420p -movflags +faststart {output_path}"

        result = subprocess.run(shlex.split(command), capture_output=True, text=True)
        if result.returncode != 0:
            print("FFmpeg error:", result.stderr)
        else:
            print("FFmpeg output:", result.stdout)

        return render_template("success.html", finalresult=finalresult, predicts=predicts, similarity=similarity)
   
    return render_template("file_upload_form.html")

def freq_to_number(f):
    return 69 + 12*np.log2(f/440.0)

def number_to_freq(n):
    return 440 * 2.0**((n-69)/12.0)

def note_name(n):
    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return NOTE_NAMES[n % 12] + str(int(n/12 - 1))

def plot_fft(pf, xf, fs, notes, FREQ_MIN, FREQ_MAX, dimensions=(960,540)):
    layout = go.Layout(
        title="frequency spectrum",
        autosize=False,
        width=dimensions[0],
        height=dimensions[1],
        xaxis_title="Frequency (note)",
        yaxis_title="Magnitude",
        font={'size' : 24}
    )

    fig = go.Figure(layout=layout,
                    layout_xaxis_range=[FREQ_MIN, FREQ_MAX],
                    layout_yaxis_range=[0,1]
                    )
    
    fig.add_trace(go.Scatter(
        x=xf,
        y=pf))
    
    for note in notes:
        fig.add_annotation(x=note[0]+10, y=note[2],
            text=note[1],
            font={'size' : 48},
            showarrow=False)

    return fig

def extract_sample(audio, frame_number, FRAME_OFFSET, FFT_WINDOW_SIZE):
    end = frame_number * FRAME_OFFSET
    begin = int(end - FFT_WINDOW_SIZE)

    if end == 0:
        return np.zeros((np.abs(begin)), dtype=float)
    elif begin < 0:
        return np.concatenate([np.zeros((np.abs(begin)), dtype=float), audio[0:end]])
    else:
        return audio[begin:end]

def find_top_notes(fft, num, xf):
    if np.max(fft.real) < 0.001:
        return []

    lst = [x for x in enumerate(fft.real)]
    lst = sorted(lst, key=lambda x: x[1], reverse=True)

    idx = 0
    found = []
    found_note = set()

    while (idx < len(lst)) and (len(found) < num):
        f = xf[lst[idx][0]]
        y = lst[idx][1]
        n = freq_to_number(f)
        n0 = int(round(n))
        name = note_name(n0)

        if name not in found_note:
            found_note.add(name)
            s = [f, note_name(n0), y]
            found.append(s)
        idx += 1
        
    return found

if __name__ == '__main__':  
    app.run(host= '127.0.0.1', debug=True)
