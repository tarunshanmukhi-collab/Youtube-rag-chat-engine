import os
import subprocess
import tempfile
from typing import List
from urllib.parse import urlparse, parse_qs
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from constants import CHUNK_SIZE, CHUNK_OVERLAP, INDEX_FOLDER,YOUTUBE_VIDEO_URLS

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

#GET VIDEO ID FROM YOUTUBE URL
def get_video_id(url: str) -> str:
    parsed = urlparse(url)

    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")

    if "youtube.com" in parsed.netloc:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        if parsed.path.startswith(("/embed/", "/shorts/")):
            return parsed.path.split("/")[2]

    raise ValueError(f"Invalid YouTube URL: {url}")


# VTT → CLEAN TEXT
def parse_vtt_file(vtt_path: str) -> str:
    import re
    lines = []

    with open(vtt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue
            if line.startswith(("WEBVTT", "Kind:", "Language:")):
                continue
            if "-->" in line:
                continue

            # Remove all tags and timestamps like <c>, <00:08:48.480>, etc.
            cleaned = re.sub(r"<[^>]+>", "", line)
            lines.append(cleaned)
    
    return " ".join(lines)


# FETCH TRANSCRIPT VIA yt-dlp
def fetch_transcript_ytdlp(url: str) -> str:
    """
    Downloads subtitles using yt-dlp and returns cleaned text.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_template = os.path.join(tmpdir, "%(id)s.%(ext)s")

        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-auto-sub",
            "--write-sub",
            "--sub-lang", "en,en-US",
            "--sub-format", "vtt",
            "-o", output_template,
            url
        ]

        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Read first .vtt file found
        for file in os.listdir(tmpdir):
            if file.endswith(".vtt"):
                return parse_vtt_file(os.path.join(tmpdir, file))

    raise RuntimeError("No subtitles found via yt-dlp")




# SPLIT TRANSCRIPT INTO CHUNKS
def split_transcript(
    transcript_text: str,
    video_id: str
) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_text(transcript_text)

    return [
        Document(
            page_content=chunk,
            metadata={
                "video_id": video_id,
                "chunk_id": idx + 1
            }
        )
        for idx, chunk in enumerate(chunks)
    ]



# FAISS INDEX CREATION
def create_or_update_index(youtube_urls: List[str]):

    if os.path.exists(INDEX_FOLDER):
        print(" Loading existing FAISS index...")
        vectorstore = FAISS.load_local(INDEX_FOLDER,embeddings,"index",allow_dangerous_deserialization=True)
    else:
        print("Creating new FAISS index...")
        vectorstore = None

    for url in youtube_urls:
        try:
            video_id = get_video_id(url)
            print(f"\n Processing video url:{url} with video id: {video_id}")

            transcript_text = fetch_transcript_ytdlp(url)

            transcript_path =  f"{video_id}.txt"
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)

            documents = split_transcript(
                transcript_text,
                video_id
            )

            video_db = FAISS.from_documents(documents, embeddings)

            if vectorstore is None:
                vectorstore = video_db
            else:
                vectorstore.merge_from(video_db)

            print(f"Indexed {len(documents)} chunks")

        except Exception as e:
            print(f"Skipping video: {e}")

    if vectorstore:
        vectorstore.save_local(INDEX_FOLDER, "index")
        print("\n FAISS index saved successfully.")
    else:
        print("\n No videos were indexed.")



if __name__ == "__main__":
    create_or_update_index(YOUTUBE_VIDEO_URLS)
