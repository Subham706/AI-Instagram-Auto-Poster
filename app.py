import os
from pathlib import Path
import subprocess
import cv2
import streamlit as st
from instagrapi import Client
from random import choice
from PIL import Image
from io import BytesIO
import warnings

# Suppress unnecessary pydantic warnings
warnings.filterwarnings("ignore")

# Google Generative AI
from google import genai
from google.genai import types

# MusicGen
from audiocraft.models import MusicGen
import torchaudio

# ---------- Config ----------
API_KEY = ""
FFMPEG_BIN = "ffmpeg"
client = genai.Client(api_key=API_KEY)

# ---------- Helpers ----------
def ensure_ffmpeg():
    try:
        subprocess.run([FFMPEG_BIN, "-version"], capture_output=True, text=True, check=True)
    except Exception as e:
        st.error(f"FFmpeg check failed: {e}")
        st.stop()

def generate_image(prompt: str, out_file: Path) -> Path:
    response = client.models.generate_content(
        model="gemini-2.0-flash-preview-image-generation",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"]
        )
    )
    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data:
            img = Image.open(BytesIO(part.inline_data.data))
            img.save(out_file)
            return out_file
    return None

def generate_ai_song_musicgen(prompt: str, out_file: Path, duration_s: int = 15, model_size: str = "small"):
    model = MusicGen.get_pretrained(model_size)
    model.set_generation_params(duration=duration_s)
    wav = model.generate([prompt])
    audio = wav[0]
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)  # Ensure 2D tensor (channels, samples)
    torchaudio.save(str(out_file), audio.cpu(), 32000)

def image_to_silent_video(image_path: Path, out_file: Path, duration_s: int = 10, fps: int = 24):
    img = cv2.imread(str(image_path))
    height, width, _ = img.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))
    total_frames = duration_s * fps
    for _ in range(total_frames):
        vw.write(img)
    vw.release()

def mux_audio_video(video_in: Path, audio_in: Path, video_out: Path):
    cmd = [
        FFMPEG_BIN, "-y", "-i", str(video_in), "-i", str(audio_in),
        "-c:v", "copy", "-c:a", "aac", "-shortest", str(video_out)
    ]
    subprocess.run(cmd, check=True)

def generate_caption_with_gemini(img_prompt: str, music_prompt: str) -> str:
    fallback_captions = [
        "Night vibes. New AI track. #aiart #aimusic",
        "AI mood, AI beats. Check this out! #aiart #aimusic",
        "Creativity meets AI. Enjoy the soundscape! #aiart #aimusic"
    ]
    try:
        prompt = (
            "Write a short Instagram caption (max 20 words), no hashtags in text, "
            "then list 5-8 relevant hashtags on a new line.\n"
            f"Image vibe: {img_prompt}\n"
            f"Music vibe: {music_prompt}\n"
            "Tone: punchy, positive."
        )
        resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return resp.text.strip() if hasattr(resp, "text") and resp.text else choice(fallback_captions)
    except Exception:
        return choice(fallback_captions)

def instagram_upload(video_path: Path, caption: str, username: str, password: str):
    cl = Client()
    cl.login(username, password)
    
    candidates = [
        {
            "width": 640,
            "height": 640,
            "url": "https://example.com/dummy1.jpg",
            "scans_profile": "e35",
        },
        {
            "width": 640,
            "height": 640,
            "url": "https://example.com/dummy2.jpg",
            "scans_profile": "e35",
        }
    ]
    
    extra_data = {
        "image_versions2": {
            "candidates": candidates
        }
    }
    
    cl.video_upload(str(video_path), caption=caption, extra_data=extra_data)

# ---------- Streamlit UI ----------
st.title("AI Instagram Auto Poster")

IG_USER = st.text_input("Instagram Username")
IG_PASS = st.text_input("Instagram Password", type="password")

img_prompt = st.text_area("Describe the image you want:")
music_prompt = st.text_area("Describe the music vibe (style/instruments/mood):")
duration = st.slider("Video length (seconds)", 10, 30, 12)

if st.button("Generate & Post"):
    ensure_ffmpeg()

    with st.spinner("Generating AI image..."):
        img_file = Path("generated_img.png")
        generate_image(img_prompt, img_file)
        st.image(str(img_file))

    with st.spinner("Generating AI song..."):
        song_file = Path("ai_song.wav")
        generate_ai_song_musicgen(music_prompt, song_file, duration_s=duration)

    st.subheader("Generated Music")
    st.audio(str(song_file))

    with st.spinner("Creating video..."):
        silent_video = Path("temp_silent.mp4")
        final_video = Path("instagram_post.mp4")
        image_to_silent_video(img_file, silent_video, duration_s=duration)
        mux_audio_video(silent_video, song_file, final_video)
        st.video(str(final_video))

    with st.spinner("Generating caption..."):
        caption = generate_caption_with_gemini(img_prompt, music_prompt)
        st.write("**Caption Preview:**")
        st.write(caption)

    if IG_USER and IG_PASS:
        with st.spinner("Uploading to Instagram..."):
            try:
                instagram_upload(final_video, caption, IG_USER, IG_PASS)
                st.success("Image successfully posted on Instagram!")
            except Exception as e:
                if "scans_profile" in str(e):
                    st.warning("Upload succeeded but a known validation warning was suppressed.")
                else:
                    st.error(f"Upload failed: {e}")
    else:
        st.warning("Please provide Instagram credentials.")
