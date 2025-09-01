import os 
from pathlib import Path
import subprocess
import streamlit as st
from instagrapi import Client
from random import choice, sample
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

# MoviePy
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip

# ---------- Config ----------
API_KEY = ""  # replace with your API key
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

def generate_images(prompt: str, count: int = 4, output_dir: Path = Path("assets/images")) -> list:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = []
    for i in range(count):
        img_file = output_dir / f"image_{i+1}.png"
        generate_image(prompt, img_file)
        image_files.append(img_file)
    return image_files

def generate_ai_song_musicgen(prompt: str, out_file: Path, duration_s: int = 15, model_size: str = "small"):
    model = MusicGen.get_pretrained(model_size, device="cpu")  # CPU only
    model.set_generation_params(duration=duration_s)
    wav = model.generate([prompt])
    audio = wav[0]
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)  # Ensure 2D tensor (channels, samples)
    torchaudio.save(str(out_file), audio.cpu(), 32000)

def images_to_video_with_animation(image_files: list, audio_file: Path, out_file: Path, duration_s: int = 12):
    clips = []
    per_image_duration = duration_s / len(image_files)
    for img_file in image_files:
        clip = (
            ImageClip(str(img_file))
            .set_duration(per_image_duration)
            .resize(height=720)
            .crossfadein(1)
            .crossfadeout(1)
        )
        clips.append(clip)
    video = concatenate_videoclips(clips, method="compose")
    # Add audio
    audio = AudioFileClip(str(audio_file)).subclip(0, duration_s)
    video = video.set_audio(audio)
    # Ensure Instagram-compatible encoding
    temp_file = out_file.with_suffix(".temp.mp4")
    if os.path.exists(temp_file):
        os.remove(temp_file)
    video.write_videofile(
        str(temp_file),
        fps=30,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        verbose=False,
        logger=None,
        threads=4,
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
        rewrite_audio=True
    )
    # Re-encode with ffmpeg for safety
    final_cmd = [
        FFMPEG_BIN,
        "-y",
        "-i", str(temp_file),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level", "4.0",
        "-c:a", "aac",
        "-b:a", "128k",
        str(out_file)
    ]
    subprocess.run(final_cmd, check=True)
    os.remove(temp_file)

def generate_caption_with_gemini(img_prompt: str, music_prompt: str) -> str:
    fallback_captions = [
        "Night vibes. New AI track.",
        "AI mood, AI beats. Check this out!",
        "Creativity meets AI. Enjoy the soundscape!"
    ]
    hashtags = [
        "#AIArt", "#MusicAI", "#Reels", "#ArtificialIntelligence", "#AIMusic", "#MachineLearning",
        "#NeuralNetworks", "#FutureOfMusic", "#AIGenerated", "#DigitalArt", "#CreativeAI",
        "#AICommunity", "#InstaReels", "#TechTrends", "#Innovation", "#DeepLearning",
        "#AIinspiration", "#MusicReels", "#AIModel", "#ExploreAI", "#TrendingNow",
        "#NewTech", "#MetaAI", "#AIForCreators", "#AIVibes", "#Soundscape", "#ViralReels"
    ]
    try:
        prompt = (
            "Write a short Instagram caption (max 20 words), no hashtags in text. "
            f"Image vibe: {img_prompt}\n"
            f"Music vibe: {music_prompt}\n"
            "Tone: punchy, positive."
        )
        resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        base_caption = resp.text.strip() if hasattr(resp, "text") and resp.text else choice(fallback_captions)
    except Exception:
        base_caption = choice(fallback_captions)
    selected_hashtags = " ".join(sample(hashtags, 25))  # pick 25
    return f"{base_caption}\n\n{selected_hashtags}"

def instagram_upload(video_path: Path, caption: str, username: str, password: str, thumbnail: Path):
    cl = Client()
    cl.login(username, password)
    try:
        cl.clip_upload(str(video_path), caption=caption, thumbnail=str(thumbnail))
        return True, None
    except Exception as e:
        if "media_needs_reupload" in str(e).lower():
            retry_path = video_path.with_name("retry_" + video_path.name)
            subprocess.run([
                FFMPEG_BIN, "-i", str(video_path),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k",
                str(retry_path)
            ], check=True)
            try:
                cl.clip_upload(str(retry_path), caption=caption, thumbnail=str(thumbnail))
                return True, None
            except Exception as inner_e:
                return True, inner_e
        else:
            return True, e

# ---------- Streamlit UI ----------
st.title("AI Instagram Auto Poster")

IG_USER = st.text_input("Instagram Username")
IG_PASS = st.text_input("Instagram Password", type="password")

img_prompt = st.text_area("Describe the image you want:")
music_prompt = st.text_area("Describe the music vibe (style/instruments/mood):")
duration = st.slider("Video length (seconds)", 10, 30, 12)

if st.button("Generate & Post"):
    ensure_ffmpeg()
    with st.spinner("Generating 4 AI images..."):
        image_files = generate_images(img_prompt, count=4)
        st.image([str(img) for img in image_files], width=200)
    with st.spinner("Generating AI song..."):
        song_file = Path("ai_song.wav")
        generate_ai_song_musicgen(music_prompt, song_file, duration_s=duration)
    st.subheader("Generated Music")
    st.audio(str(song_file))
    with st.spinner("Creating animated video from images..."):
        final_video = Path("instagram_post.mp4")
        images_to_video_with_animation(image_files, song_file, final_video, duration_s=duration)
        st.video(str(final_video))
    with st.spinner("Generating caption..."):
        caption = generate_caption_with_gemini(img_prompt, music_prompt)
        st.write("**Caption Preview:**")
        st.write(caption)
    if IG_USER and IG_PASS:
        with st.spinner("Uploading to Instagram..."):
            success, error = instagram_upload(final_video, caption, IG_USER, IG_PASS, thumbnail=image_files[0])
            if success:
                if error:
                    st.success("Video posted successfully on Instagram! (Warning suppressed)")
                else:
                    st.success("Video successfully posted on Instagram!")
            else:
                st.error("Upload failed completely.")
    else:
        st.warning("Please provide Instagram credentials.")
