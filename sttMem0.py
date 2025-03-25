""" 💬 AI STT Companion with memory
Features:
- Mem0 with OpenAI embeddings + DeepSeek/OpenRouter LLM
- Faster-Whisper for STT
- Terminal-based text responses
- Pg-vector with PostgreSQL
"""

import os
import time
import numpy as np
from pydantic import BaseModel
from faster_whisper import WhisperModel
from mem0 import Memory
from openai import OpenAI
from datetime import datetime
from termcolor import cprint
import pyaudio
import wave
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === Configuration ===
class TextAgentConfig(BaseModel):
    llm_provider: str = os.getenv("LLM_PROVIDER", "openrouter")
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY")
    memory_size: int = 7

# === AI Companion Class ===
class AITextCompanion:
    def __init__(self):
        self.config = TextAgentConfig()
        self.setup_directories()
        self.setup_memory()
        self.setup_whisper()
        self.setup_llm_client()
        self.conversation_history = []
        cprint("💬 Hi mate, how are you today? (Speak to me or type 't' to enter text mode)", "green")

    def setup_directories(self):
        self.data_dir = Path("data/conversations")
        self.audio_dir = Path("data/audio")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def setup_memory(self):
        """Mem0 with pgvector/PostgreSQL"""
        if not self.config.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for embeddings")
        
        self.memory = Memory.from_config({
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-large",
                    "api_key": self.config.openai_api_key
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "api_key": self.config.openai_api_key
                }
            },
            "vector_store": {
                "provider": "pgvector",
                "config": {
                    "user": "postgres",
                    "password": "postgres",
                    "host": "127.0.0.1",
                    "port": "54329",
                    "dbname": "postgres"
                }
            }
        })
        cprint("🧠 Initialized PostgreSQL/pgvector storage", "cyan")

    def setup_whisper(self):
        self.whisper_model = WhisperModel(
            "medium",
            device="auto",
            compute_type="int8"
        )
        cprint("🎤 Whisper STT model loaded", "cyan")

    def setup_llm_client(self):
        """Double LLM setup for DeepSeek/OpenRouter"""
        if self.config.llm_provider == "deepseek":
            if not self.config.deepseek_api_key:
                raise ValueError("DEEPSEEK_API_KEY not set")
            self.llm_client = OpenAI(
                api_key=self.config.deepseek_api_key,
                base_url="https://api.deepseek.com"
            )
            self.llm_model = "deepseek-chat"
            cprint("🧠 Using DeepSeek LLM", "cyan")
        
        elif self.config.llm_provider == "openrouter":
            if not self.config.openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY not set")
            self.llm_client = OpenAI(
                api_key=self.config.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            self.llm_model = "deepseek/deepseek-chat-v3-0324:free"
            cprint("🧠 Using OpenRouter LLM", "cyan")
        
        else:
            raise ValueError("Invalid LLM provider")

    # === Memory Operations ===
    def add_to_memory(self, text, speaker="user"):
        try:
            if not text or len(text.strip()) < 3:
                cprint("⚠️ Text too short, not storing in memory", "yellow")
                return
                
            self.memory.add(
                messages=[{"role": speaker, "content": text}],
                user_id="default"
            )
            cprint(f"🧠 Stored memory: {text[:50]}...", "magenta")
        except Exception as e:
            cprint(f"❌ Memory storage error: {e}", "red")
            # Add to conversation history even if memory storage fails
            self.conversation_history.append({"role": speaker, "content": text})
        
    def check_memory_count(self):
        """Debug function to check memory count in database"""
        import psycopg2
        conn = psycopg2.connect(
            user="postgres",
            password="postgres",
            host="127.0.0.1",
            port="54329",
            database="postgres"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM mem0_memories")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        cprint(f"📊 Current memory count: {count}", "cyan")
        return count
        
    def retrieve_relevant(self, query):
        results = self.memory.search(
            query=query,
            user_id="default",
            limit=3
        )
        return [mem["memory"] for mem in results["results"]]

    # === Speech Processing ===
    def record_audio(self):
        """Record audio with manual start/stop via Enter key"""
        RATE = 16000
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        MAX_DURATION = 120  # Maximum recording duration in seconds (increased to 2 minutes)
        
        # Prompt user to start recording
        cprint("🎤 Press Enter to start recording...", "yellow")
        input()
        
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        cprint("🎙️ Recording... (Press Enter to stop)", "green")
        
        frames = []
        max_chunks = int(MAX_DURATION * RATE / CHUNK)
        
        # Start a separate thread to wait for Enter key
        import threading
        stop_recording = threading.Event()
        
        def wait_for_enter():
            input()
            stop_recording.set()
            cprint("🛑 Stopping recording...", "yellow")
        
        enter_thread = threading.Thread(target=wait_for_enter)
        enter_thread.daemon = True
        enter_thread.start()
        
        # Main recording loop
        chunk_count = 0
        while chunk_count < max_chunks and not stop_recording.is_set():
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            chunk_count += 1
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        if chunk_count >= max_chunks:
            cprint(f"⚠️ Reached maximum recording duration ({MAX_DURATION}s)", "yellow")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.audio_dir / f"input_{timestamp}.wav"
        wf = wave.open(str(filename), 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        duration = len(frames) * CHUNK / RATE
        cprint(f"🎙️ Recorded {duration:.1f} seconds of audio", "green")
        return filename

    def transcribe(self, audio_file):
        start_time = time.time()
        try:
            segments, _ = self.whisper_model.transcribe(
                audio_file,
                beam_size=5,
                language="en",
                vad_filter=True
            )
            transcription = " ".join([seg.text for seg in segments])
            cprint(f"🎤 Transcription completed in {time.time() - start_time:.2f} seconds.", "yellow")
            return transcription
        except Exception as e:
            cprint(f"❌ Transcription failed: {e}", "red")
            return None

    # === Response Generation ===
    def generate_response(self, user_input):
        context = self.retrieve_relevant(user_input)
        context_str = "\n".join(context) if context else "No relevant memories"
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": f"Context from memory:\n{context_str}"},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            cprint(f"❌ LLM Error: {e}", "red")
            return "I encountered an error generating a response. Please try again."

    # === Display Response ===
    def display_response(self, text):
        """Display the AI response in the terminal with formatting"""
        cprint("\n┌─────────────────────────────────────────────────", "blue")
        cprint("│ 🤖 AI Response:", "blue")
        
        # Split text into lines for better formatting
        lines = text.split('\n')
        for line in lines:
            # Wrap long lines
            while len(line) > 70:
                cprint(f"│ {line[:70]}", "cyan")
                line = line[70:]
            cprint(f"│ {line}", "cyan")
            
        cprint("└─────────────────────────────────────────────────\n", "blue")

    # === Main Loop ===
    def run(self):
        cprint("💬 AI Text Companion Ready - Speak or type 't' for text input", "cyan")
        memory_check_counter = 0
        try:
            while True:
                cprint("\n🔄 Waiting for input (speak or type 't' for text mode)...", "yellow")
                user_choice = input("Press Enter to speak or 't' to type: ").strip().lower()
                
                if user_choice == 't':
                    cprint("📝 Text input mode:", "yellow")
                    user_input = input("You: ")
                else:
                    audio_file = self.record_audio()
                    user_input = self.transcribe(audio_file)
                
                if user_input:
                    cprint(f"🧑 You: {user_input}", "green")
                    self.add_to_memory(user_input)
                    
                    cprint("🤔 Thinking...", "yellow")
                    response = self.generate_response(user_input)
                    self.add_to_memory(response, "assistant")
                    
                    # Display the response in the terminal
                    self.display_response(response)
                    
                    # Check memory count every 5 interactions
                    memory_check_counter += 1
                    if memory_check_counter % 5 == 0:
                        self.check_memory_count()
                else:
                    cprint("❌ No input detected. Please try again.", "red")
                    
        except KeyboardInterrupt:
            cprint("\n👋 Shutting down...", "yellow")

if __name__ == "__main__":
    try:
        companion = AITextCompanion()
        companion.run()
    except Exception as e:
        cprint(f"❌ Initialization failed: {e}", "red")