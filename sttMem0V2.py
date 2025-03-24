""" 💬 AI Text Companion V2
Features:
- Mem0 with OpenAI embeddings + DeepSeek/OpenRouter LLM
- Faster-Whisper for STT
- Terminal-based text responses
- Pg-vector with PostgreSQL
- Shares memory with Voice Companion
- Improved conversation flow with short-term context window
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
from dotenv import load_dotenv
import pyaudio
import wave
import torch
from pathlib import Path
import json

load_dotenv()

# === Configuration ===
class TextAgentConfig(BaseModel):
    llm_provider: str = os.getenv("LLM_PROVIDER", "openrouter")
    openai_api_key: str = os.getenv("OPENAI_API_KEY")  # Required for embeddings
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY")
    memory_size: int = 7
    context_window_size: int = 5  # Number of recent messages to keep in context window

# === AI Companion Class ===
class AITextCompanion:
    def __init__(self):
        self.config = TextAgentConfig()
        self.setup_directories()
        self.setup_memory()
        self.setup_whisper()
        self.setup_llm_client()
        self.conversation_history = []  # Short-term conversation context
        cprint("💬 Hello Mate, wanna remember something? (Speak to me or type 't' to enter text mode)", "green")

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
            
            # Add to short-term conversation history
            self.conversation_history.append({"role": speaker, "content": text})
            
            # Keep only the most recent messages in the conversation history
            if len(self.conversation_history) > self.config.context_window_size * 2:  # *2 to account for pairs of messages
                self.conversation_history = self.conversation_history[-self.config.context_window_size * 2:]
                
            # Store in long-term memory (Mem0)
            self.memory.add(
                messages=[{"role": speaker, "content": text}],
                user_id="default"
            )
            cprint(f"🧠 Stored memory: {text[:50]}...", "magenta")
            
            # Every few exchanges, also store the entire recent conversation as a single memory
            # This helps maintain conversation flow in future retrievals
            if speaker == "assistant" and len(self.conversation_history) >= 4 and len(self.conversation_history) % 4 == 0:
                recent_convo = self.conversation_history[-4:]  # Get the last 2 exchanges (4 messages)
                convo_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_convo])
                
                self.memory.add(
                    messages=[{"role": "system", "content": f"Conversation exchange: {convo_text}"}],
                    user_id="default",
                    metadata={"type": "conversation_chunk", "timestamp": datetime.now().isoformat()}
                )
                cprint("🧠 Stored conversation chunk in memory", "magenta")
                
        except Exception as e:
            cprint(f"❌ Memory storage error: {e}", "red")
    
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
        cursor.execute("SELECT COUNT(*) FROM mem0")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        cprint(f"📊 Current memory count: {count}", "cyan")
        return count
        
    def retrieve_relevant(self, query):
        """Retrieve relevant memories based on the query"""
        results = self.memory.search(
            query=query,
            user_id="default",
            limit=5  # Increased from 3 to get more context
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
        # Get relevant memories from long-term storage
        relevant_memories = self.retrieve_relevant(user_input)
        memory_context = "\n".join(relevant_memories) if relevant_memories else "No relevant memories"
        
        # Prepare messages for the LLM
        messages = [
            {"role": "system", "content": f"""You are a helpful AI assistant with access to both long-term memories and recent conversation context.
            
Long-term memories:
{memory_context}

Be conversational, friendly, and maintain the flow of conversation. Use the long-term memories for factual information, but prioritize the recent conversation context for continuity."""}
        ]
        
        # Add the recent conversation history to provide continuity
        if self.conversation_history:
            # Add the most recent messages from conversation history
            messages.extend(self.conversation_history[-self.config.context_window_size*2:])
        
        # Add the current user input if it's not already included
        if not self.conversation_history or self.conversation_history[-1]["role"] != "user" or self.conversation_history[-1]["content"] != user_input:
            messages.append({"role": "user", "content": user_input})
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
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

    # === Conversation Management ===
    def save_conversation(self):
        """Save the current conversation to a JSON file"""
        if not self.conversation_history:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.data_dir / f"conversation_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "messages": self.conversation_history
            }, f, indent=2)
            
        cprint(f"💾 Conversation saved to {filename}", "green")

    # === Main Loop ===
    def run(self):
        cprint("💬 AI Text Companion V2 Ready - Speak or type 't' for text input", "cyan")
        cprint("🔄 This version maintains conversation flow with a hybrid memory approach", "cyan")
        memory_check_counter = 0
        try:
            while True:
                cprint("\n🔄 Waiting for input (speak or type 't' for text mode)...", "yellow")
                user_choice = input("Press Enter to speak or 't' to type: ").strip().lower()
                
                if user_choice == 't':
                    cprint("📝 Text input mode:", "yellow")
                    user_input = input("You: ")
                elif user_choice == 'q':
                    cprint("👋 Exiting...", "yellow")
                    self.save_conversation()
                    break
                elif user_choice == 'm':
                    # Debug: Show memory count
                    self.check_memory_count()
                    continue
                elif user_choice == 'h':
                    # Show conversation history
                    cprint("\n📜 Recent Conversation History:", "magenta")
                    for i, msg in enumerate(self.conversation_history[-10:]):
                        role_color = "green" if msg["role"] == "user" else "cyan"
                        cprint(f"{i+1}. {msg['role']}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}", role_color)
                    continue
                else:
                    audio_file = self.record_audio()
                    user_input = self.transcribe(audio_file)
                
                if user_input:
                    cprint(f"🧑 You: {user_input}", "green")
                    self.add_to_memory(user_input, "user")
                    
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
            self.save_conversation()

if __name__ == "__main__":
    try:
        companion = AITextCompanion()
        companion.run()
    except Exception as e:
        cprint(f"❌ Initialization failed: {e}", "red")