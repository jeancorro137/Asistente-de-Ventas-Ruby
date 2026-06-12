import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

print("Habla durante 5 segundos...")

fs = 16000
audio = sd.rec(
    int(5 * fs),
    samplerate=fs,
    channels=1,
    dtype="int16"
)

sd.wait()

write("audio.wav", fs, audio)

print("Transcribiendo...")

model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

segments, info = model.transcribe(
    "audio.wav",
    language="es"
)

texto = ""

for segment in segments:
    texto += segment.text

print("\nTexto detectado:")
print(texto)