import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import json
import os
import requests
import pyttsx3
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "tinyllama:latest"

USE_CLAUDE = False
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

engine = pyttsx3.init()

with open("stock.json", "r", encoding="utf-8") as f:
    stock = json.load(f)

whisper = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

SYSTEM_PROMPT = f"""
Eres Ruby, una asesora virtual de ventas de ropa.

IMPORTANTE:
- Debes responder SIEMPRE en español.
- Nunca respondas en inglés.
- Usa un lenguaje claro, sencillo y amigable.
- No mezcles idiomas bajo ninguna circunstancia.

Tu trabajo es responder SOLO con base en este inventario:

{json.dumps(stock, ensure_ascii=False, indent=2)}

Reglas:
- No inventes productos.
- Si el cliente pide algo que no existe, ofrece una alternativa parecida.
- Menciona precio, talla y color si están disponibles.
- Responde de forma breve, clara y amable.
- Cierra invitando a comprar.

Ejemplo de respuesta correcta:
"Tenemos una camiseta negra en talla M por $35.000. Es una excelente opción. ¿Te gustaría comprarla?"
"""

def escuchar():

    print("\n🎤 Habla ahora...")

    fs = 16000

    audio = sd.rec(
        int(5 * fs),
        samplerate=fs,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    write("audio.wav", fs, audio)

    segments, _ = whisper.transcribe(
        "audio.wav",
        language="es"
    )

    texto = ""

    for segment in segments:
        texto += segment.text

    print(f"\nCliente: {texto}")

    return texto

def hablar(texto: str):
    engine.say(texto)
    engine.runAndWait()

def preguntar_ollama(pregunta: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pregunta}
        ],
        "stream": False
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]

def mejorar_con_claude(texto: str) -> str:
    if not ANTHROPIC_API_KEY:
        return texto

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    payload = {
        "model": "claude-sonnet-4-0",
        "max_tokens": 250,
        "messages": [
            {
                "role": "user",
                "content": f"""Mejora esta respuesta de ventas.
Debe sonar clara, natural y comercial, sin inventar nada:

{texto}"""
            }
        ]
    }

    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["content"][0]["text"]

def main():
    print("Ruby está lista. Escribe 'salir' para terminar.\n")

    while True:
        #pregunta = input("Cliente: ").strip()
        pregunta = escuchar().strip()

        if pregunta.lower() == "salir":
            print("Fin del chat.")
            break

        try:
            respuesta = preguntar_ollama(pregunta)

            if USE_CLAUDE:
                respuesta = mejorar_con_claude(respuesta)

            print(f"\nRuby: {respuesta}\n")
            hablar(respuesta)

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
