"""
This implementation provides:

Complete video processing pipeline - Download → Transcript → Chunk → Vectorize
Intelligent chunking - Based on actions, content size, and topic changes
Fallback mechanisms - YouTube API → Whisper for transcripts
Vector storage - FAISS with metadata preservation
Search capabilities - Semantic search with similarity scores
Error handling - Comprehensive logging and error management
Data persistence - Saves processed data and vector stores
"""

import json
import re
from pathlib import Path
from typing import List, Dict

# Video processing imports
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import whisper

# LangChain imports
from langchain_community.vectorstores import FAISS

from src.utils import embeddings as embeddings_mod

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.utils.logging_setup import get_logger

logger = get_logger(__name__, log_file="agents.log")

def _brief(exc, limit: int = 240) -> str:
    """A one-glance reason, for showing a user.

    ffmpeg fails by printing its whole build banner and putting the real message on
    the last line, so a naive str(exc) buries the cause under 25 lines of configure
    flags. Keep the first line (what failed) and the last two (why). The untrimmed
    text is still in the log -- every caller logs it before calling this.
    """
    lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
    if not lines:
        return ""
    kept = lines if len(lines) <= 3 else [lines[0], "...", *lines[-2:]]
    brief = " ".join(kept)
    return brief if len(brief) <= limit else brief[:limit - 1] + "…"


# Why transcription fell back, as a stable code. Classified against the *full*
# exception text: _brief() trims the middle out for display, and the decisive line
# ("does not contain any stream") is exactly what it drops.
NO_AUDIO = "no_audio"
FFMPEG_MISSING = "ffmpeg_missing"


DEFAULT_STORE_KEY = "default"


def store_key_for(source: str) -> str:
    """Directory name for one source's vectors: the YouTube id, or the file's stem.

    Every ingestion used to `save_local` over the same `vector_store/` directory, so
    the last video processed silently replaced every earlier one and no video could
    be revisited without re-embedding it. Keying by source keeps them side by side.

    Human-readable on purpose -- `vector_store/iBTWBhODSU0/`, `vector_store/demo-1/` --
    so the directory can be inspected and deleted by hand. Identity is only *where*
    the vectors live; whether they are still current is `content_hash` in the meta.
    """
    text = str(source or "").strip()
    youtube = re.search(r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#/]+)", text)
    if youtube:
        return youtube.group(1)
    if not text.lower().startswith(("http://", "https://")):
        text = Path(text).stem
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.")
    return slug[:60] or DEFAULT_STORE_KEY


def chunks_fingerprint(chunks: List[Dict]) -> str:
    """Hash of exactly what would be embedded.

    The honest reuse key: identical chunk text means identical vectors, so a run
    whose fingerprint matches the stored one can skip re-embedding entirely. File
    mtimes and sizes only approximate this.
    """
    import hashlib

    joined = "\n".join(str(c.get("text", "")) for c in chunks)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def _classify_transcript_failure(exc) -> str | None:
    text = str(exc)
    if "does not contain any stream" in text or "Output file #0 does not contain" in text:
        return NO_AUDIO
    if "No such file or directory: 'ffmpeg'" in text or "ffmpeg not found" in text:
        return FFMPEG_MISSING
    return None


class DataIngestionAgent:
    def __init__(self, data_dir: str = "src/data"):
        """Initialize the Data Ingestion Agent"""
        self.data_dir = Path(data_dir)
        self.videos_dir = self.data_dir / "videos"
        self.transcripts_dir = self.data_dir / "transcripts"
        
        # Create directories
        self._create_directories()
        
        # Vectors are produced in-process by a local model (src/utils/embeddings.py);
        # the gateway serves chat only. An existing store pins the vector width --
        # mixing widths would corrupt it, so pass it through.
        store_meta = embeddings_mod.read_store_meta(self.data_dir / "vector_store") or {}
        self.embeddings = embeddings_mod.Qwen3Embeddings(
            expected_dimensions=store_meta.get("dimensions")
        )
        self.vector_store = None
        self.last_store_key = DEFAULT_STORE_KEY
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Action keywords for intelligent chunking
        self.action_keywords = [
            "click", "select", "enter", "type", "navigate", "open", 
            "close", "save", "upload", "download", "scroll", "drag",
            "button", "menu", "form", "field", "login", "signup"
        ]
    
    def _validate_transcript_result(self, transcript_result: Dict) -> bool:
        """Validate transcript result structure"""
        if not isinstance(transcript_result, dict):
            return False
        
        if not transcript_result.get("success", False):
            return False
        
        transcript_text = transcript_result.get("transcript", "")
        if not transcript_text or not isinstance(transcript_text, str):
            return False
        
        return True
    
    def _create_directories(self):
        """Create necessary directories"""
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directories: {self.videos_dir}, {self.transcripts_dir}")
    
    def process_video_content(self, video_url: str) -> Dict:
        """Process video and store in vector database"""
        try:
            logger.info(f"Processing video: {video_url}")
            
            # Step 1: Download video
            download_result = self._download_video(video_url)
            if not download_result.get("success", False):
                return download_result
            
            # Step 2: Extract transcript
            video_id = download_result.get("video_id")
            if not video_id:
                return {"success": False, "error": "No video ID found"}
            
            video_path = download_result.get("video_path")
            transcript_result = self._extract_transcript(video_id, video_path)
            
            if not transcript_result.get("success", False):
                return transcript_result
            
            # Step 3: Chunk and vectorize transcript
            # Pass the full transcript result to chunking
            chunks = self._intelligent_chunking(transcript_result)
            vectorize_result = self.chunk_and_vectorize(
                chunks, store_key=store_key_for(video_url), source=video_url
            )
            
            # Step 4: Save processed data
            output_data = {
                "video_info": download_result,
                "transcript": transcript_result.get("transcript", ""),
                "transcript_method": transcript_result.get("method"),
                "transcript_reason": transcript_result.get("reason"),
                "transcript_cause": transcript_result.get("cause"),
                "chunks": chunks,
                "vector_store_info": vectorize_result
            }
            
            output_file = self._save_processed_data(video_id, output_data)
            
            # Return structured result that matches expected format
            return {
                "success": True,
                "video_id": video_id,
                "video_info": download_result,
                "transcript": transcript_result.get("transcript", ""),
                "transcript_method": transcript_result.get("method"),
                "transcript_reason": transcript_result.get("reason"),
                "transcript_cause": transcript_result.get("cause"),
                "chunks": chunks,
                "chunks_count": len(chunks),
                "vector_store_info": vectorize_result,
                "vector_store_key": vectorize_result.get("store_key"),
                "output_file": str(output_file)
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}
    
    def _download_video(self, video_url: str) -> Dict:
        """Download video with multiple fallback methods"""
        try:
            logger.info(f"Attempting to download video: {video_url}")
            
            # Extract video ID
            video_id = self._extract_video_id(video_url)
            if not video_id:
                return {"success": False, "error": "Could not extract video ID from URL"}
            
            # Try Method 1: pytube with updated settings
            try:
                return self._download_with_pytube(video_url, video_id)
            except Exception as e:
                logger.warning(f"Pytube download failed: {e}")
            
            # Try Method 2: yt-dlp (more reliable alternative)
            try:
                return self._download_with_ytdlp(video_url, video_id)
            except Exception as e:
                logger.warning(f"yt-dlp download failed: {e}")
            
            # Method 3: Skip download and use transcript only
            logger.info("Skipping video download, using transcript-only approach")
            return {
                "success": True,
                "video_id": video_id,
                "video_path": None,  # No video file
                "title": "Unknown Video",
                "description": "Video download skipped",
                "method": "transcript_only"
            }
            
        except Exception as e:
            logger.error(f"All download methods failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def _download_with_pytube(self, video_url: str, video_id: str) -> Dict:
        """Download using pytube with updated settings"""
        from pytube import YouTube
        
        # Create YouTube object with custom settings
        yt = YouTube(
            video_url,
            use_oauth=False,
            allow_oauth_cache=False
        )
        
        # Get video info
        title = yt.title
        description = yt.description or ""
        
        # Download video (lowest quality to save space and time)
        video_stream = yt.streams.filter(
            adaptive=True, 
            file_extension='mp4',
            only_video=True
        ).order_by('resolution').first()
        
        if not video_stream:
            video_stream = yt.streams.filter(
                file_extension='mp4'
            ).order_by('resolution').first()
        
        if not video_stream:
            raise Exception("No suitable video stream found")
        
        # Download to videos directory
        video_path = self.videos_dir / f"{video_id}.mp4"
        video_stream.download(
            output_path=str(self.videos_dir),
            filename=f"{video_id}.mp4"
        )
        
        return {
            "success": True,
            "video_id": video_id,
            "video_path": str(video_path),
            "title": title,
            "description": description,
            "method": "pytube"
        }

    def _download_with_ytdlp(self, video_url: str, video_id: str) -> Dict:
        """Download using yt-dlp (more reliable)"""
        import subprocess
        import json
        
        # Install yt-dlp if not available
        try:
            import yt_dlp
        except ImportError:
            logger.info("Installing yt-dlp...")
            subprocess.check_call(["pip", "install", "yt-dlp"])
            import yt_dlp
        
        # Set output path
        video_path = self.videos_dir / f"{video_id}.%(ext)s"
        
        # yt-dlp options
        ydl_opts = {
            'outtmpl': str(video_path),
            'format': 'worst[ext=mp4]/worst',  # Download lowest quality
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info
            info = ydl.extract_info(video_url, download=False)
            title = info.get('title', 'Unknown')
            description = info.get('description', '')
            
            # Download video
            ydl.download([video_url])
            
            # Find the actual downloaded file
            downloaded_file = None
            for file in self.videos_dir.glob(f"{video_id}.*"):
                downloaded_file = str(file)
                break
            
            return {
                "success": True,
                "video_id": video_id,
                "video_path": downloaded_file,
                "title": title,
                "description": description,
                "method": "yt-dlp"
            }

    def _extract_video_id(self, video_url: str) -> str:
        """Extract YouTube video ID from URL"""
        import re
        
        # Handle different YouTube URL formats
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
            r'youtube\.com/watch\?.*v=([^&\n?#]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, video_url)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_transcript(self, video_id: str, video_path: str = None) -> Dict:
        """Extract transcript with fallback methods"""
        try:
            # Why each real method failed, so the synthetic fallback can say so
            # instead of being indistinguishable from a genuine transcript.
            reasons = []
            cause = None

            # Method 1: Try YouTube transcript API first
            try:
                logger.info("Attempting to get transcript from YouTube API...")
                transcript_data = self._get_youtube_transcript(video_id)
                if transcript_data["success"]:
                    return transcript_data
                reasons.append(f"YouTube captions: {_brief(transcript_data.get('error', 'unavailable'))}")
            except Exception as e:
                logger.warning(f"YouTube transcript API failed: {e}")
                reasons.append(f"YouTube captions: {_brief(e)}")
            
            # Method 2: Try Whisper if video file exists
            if video_path and Path(video_path).exists():
                try:
                    logger.info("Attempting Whisper transcription...")
                    return self._get_whisper_transcript(video_path)
                except Exception as e:
                    logger.warning(f"Whisper transcription failed: {e}")
                    reasons.append(f"Whisper: {_brief(e)}")
                    cause = cause or _classify_transcript_failure(e)
            else:
                reasons.append("Whisper: the video could not be downloaded")
            
            # Method 3: Create basic transcript from video metadata
            logger.info("Creating basic transcript from available data...")
            return self._create_basic_transcript(video_id, "; ".join(reasons), cause)
            
        except Exception as e:
            logger.error(f"All transcript methods failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def _get_youtube_transcript(self, video_id: str) -> Dict:
        """Get transcript using YouTube Transcript API"""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            
            # Get transcript data directly
            transcript_data = YouTubeTranscriptApi.get_transcript(
                video_id, 
                languages=['en', 'en-US', 'en-GB', 'auto']
            )
            
            # Format transcript - handle both old and new response formats
            full_transcript = ""
            segments = []
            
            for entry in transcript_data:
                # Handle different response formats
                try:
                    if isinstance(entry, dict):
                        text = entry.get('text', '')
                        start = entry.get('start', 0)
                        duration = entry.get('duration', 0)
                        end = start + duration
                    else:
                        # Handle FetchedTranscriptSnippet objects
                        text = getattr(entry, 'text', str(entry))
                        start = getattr(entry, 'start', 0)
                        duration = getattr(entry, 'duration', 0)
                        end = start + duration
                    
                    # Clean up text
                    text = str(text).strip()
                    if text:
                        full_transcript += text + " "
                        segments.append({
                            'text': text,
                            'start': float(start),
                            'end': float(end),
                            'duration': float(duration)
                        })
                        
                except Exception as entry_error:
                    logger.warning(f"Error processing transcript entry: {entry_error}")
                    continue
            
            if not full_transcript.strip():
                raise Exception("No transcript text extracted")
            
            return {
                "success": True,
                "transcript": full_transcript.strip(),
                "segments": segments,
                "method": "youtube_api"
            }
            
        except Exception as e:
            logger.error(f"YouTube transcript error: {str(e)}")
            raise Exception(f"YouTube transcript failed: {str(e)}")

    def _get_whisper_transcript(self, video_path: str) -> Dict:
        """Get transcript using Whisper"""
        try:
            from ..utils.video_processor import VideoProcessor
            
            processor = VideoProcessor()
            
            if not hasattr(processor, 'whisper_model') or processor.whisper_model is None:
                raise Exception("Whisper model not available")
            
            logger.info(f"Transcribing video file: {video_path}")
            result = processor.whisper_model.transcribe(video_path)
            
            # Format segments
            segments = []
            if 'segments' in result:
                for segment in result['segments']:
                    segments.append({
                        'text': segment.get('text', ''),
                        'start': segment.get('start', 0),
                        'end': segment.get('end', 0),
                        'duration': segment.get('end', 0) - segment.get('start', 0)
                    })
            
            return {
                "success": True,
                "transcript": result.get("text", ""),
                "segments": segments,
                "method": "whisper"
            }
            
        except Exception as e:
            logger.error(f"Whisper transcription error: {str(e)}")
            raise Exception(f"Whisper model not available: {str(e)}")

    def _create_basic_transcript(self, video_id: str, reason: str = "", cause: str | None = None) -> Dict:
        """Create a basic transcript when other methods fail"""
        try:
            basic_transcript = f"""
            This is a video analysis for video ID: {video_id}
            
            Video Content Analysis:
            - This appears to be a tutorial or demonstration video
            - The video likely contains user interface interactions
            - Common elements may include navigation, form filling, and button clicks
            - The video demonstrates a typical user workflow or process
            
            Key Interaction Points:
            - User navigates to application
            - User interacts with interface elements
            - User completes tasks or workflows
            - Application responds to user actions
            
            Testing Focus Areas:
            - Navigation functionality
            - Form validation and submission
            - User interface responsiveness
            - Error handling and edge cases
            """
            
            return {
                "success": True,
                "transcript": basic_transcript.strip(),
                "segments": [{"text": basic_transcript.strip(), "start": 0, "end": 120}],
                "method": "basic_fallback",
                "reason": reason,
                "cause": cause
            }
        except Exception as e:
            logger.error(f"Error creating basic transcript: {e}")
            return {
                "success": False,
                "error": f"Failed to create basic transcript: {str(e)}"
            }
    
    def _intelligent_chunking(self, transcript_data) -> List[Dict]:
        """Intelligent chunking based on content and actions"""
        chunks = []
        
        # Handle different input formats
        if isinstance(transcript_data, str):
            # If transcript is a string, create basic chunks
            return self._chunk_string_transcript(transcript_data)
        elif isinstance(transcript_data, dict) and 'segments' in transcript_data:
            # If transcript has segments
            segments = transcript_data['segments']
        elif isinstance(transcript_data, list):
            # If transcript is already a list of segments
            segments = transcript_data
        else:
            # Fallback: treat as string
            return self._chunk_string_transcript(str(transcript_data))
        
        current_chunk = {
            "text": "",
            "start_time": 0,
            "end_time": 0,
            "actions": [],
            "chunk_type": "content"
        }
        
        for i, segment in enumerate(segments):
            # Handle different segment formats
            if isinstance(segment, dict):
                text = segment.get("text", "")
                start = segment.get("start", 0)
                duration = segment.get("duration", 0)
                end = segment.get("end", start + duration)
            else:
                # Handle string segments
                text = str(segment)
                start = i * 10  # Arbitrary timestamps
                end = (i + 1) * 10
            
            text_lower = text.lower()
            
            # Check for action keywords
            actions_found = [keyword for keyword in self.action_keywords if keyword in text_lower]
            
            # Determine if this should start a new chunk
            should_split = (
                len(current_chunk["text"]) > 800 or  # Size-based split
                (actions_found and current_chunk["text"]) or  # Action-based split
                self._is_topic_change(current_chunk["text"], text)  # Topic change
            )
            
            if should_split and current_chunk["text"]:
                # Finalize current chunk
                current_chunk["chunk_id"] = len(chunks)
                chunks.append(current_chunk.copy())
                
                # Start new chunk
                current_chunk = {
                    "text": text,
                    "start_time": start,
                    "end_time": end,
                    "actions": actions_found,
                    "chunk_type": "action" if actions_found else "content"
                }
            else:
                # Add to current chunk
                if not current_chunk["text"]:
                    current_chunk["start_time"] = start
                current_chunk["text"] += " " + text if current_chunk["text"] else text
                current_chunk["actions"].extend(actions_found)
                current_chunk["end_time"] = end
        
        # Add final chunk
        if current_chunk["text"]:
            current_chunk["chunk_id"] = len(chunks)
            chunks.append(current_chunk)
        
        logger.info(f"Created {len(chunks)} intelligent chunks")
        return chunks

    def _chunk_string_transcript(self, transcript_text: str) -> List[Dict]:
        """Chunk a plain text transcript"""
        # Split text into sentences or by length
        sentences = transcript_text.split('. ')
        chunks = []
        current_chunk = ""
        
        for i, sentence in enumerate(sentences):
            if len(current_chunk) + len(sentence) > 800 and current_chunk:
                # Create chunk
                actions_found = [keyword for keyword in self.action_keywords if keyword in current_chunk.lower()]
                chunks.append({
                    "chunk_id": len(chunks),
                    "text": current_chunk.strip(),
                    "start_time": len(chunks) * 30,  # Arbitrary timestamps
                    "end_time": (len(chunks) + 1) * 30,
                    "actions": actions_found,
                    "chunk_type": "action" if actions_found else "content"
                })
                current_chunk = sentence
            else:
                current_chunk += ". " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk:
            actions_found = [keyword for keyword in self.action_keywords if keyword in current_chunk.lower()]
            chunks.append({
                "chunk_id": len(chunks),
                "text": current_chunk.strip(),
                "start_time": len(chunks) * 30,
                "end_time": (len(chunks) + 1) * 30,
                "actions": actions_found,
                "chunk_type": "action" if actions_found else "content"
            })
        
        return chunks
    
    def _is_topic_change(self, previous_text: str, current_text: str) -> bool:
        """Simple topic change detection"""
        topic_indicators = [
            "now let's", "next step", "moving on", "let's switch",
            "another way", "alternatively", "on the other hand"
        ]
        return any(indicator in current_text.lower() for indicator in topic_indicators)
    
    def process_video_file(self, file_path: str) -> Dict:
        """Process uploaded video file and store in vector database"""
        try:
            logger.info(f"Processing uploaded video file: {file_path}")
            
            file_name = Path(file_path).stem
            # Deterministic: the same file must yield the same id, or every re-upload
            # writes another transcript and overwrites the previous vector store.
            video_id = f"uploaded_{store_key_for(file_path)}"
            
            # Step 1: Create video info for uploaded file
            video_info = {
                "success": True,
                "video_id": video_id,
                "video_path": file_path,
                "title": f"Uploaded Video: {file_name}",
                "description": f"Uploaded video file: {Path(file_path).name}",
                "method": "file_upload"
            }
            
            # Step 2: Extract transcript using Whisper
            transcript_result = self._extract_transcript_from_file(file_path)
            
            if not transcript_result.get("success", False):
                return transcript_result
            
            # Step 3: Chunk and vectorize transcript
            chunks = self._intelligent_chunking(transcript_result)
            vectorize_result = self.chunk_and_vectorize(
                chunks, store_key=store_key_for(file_path), source=file_path
            )
            
            # Step 4: Save processed data
            output_data = {
                "video_info": video_info,
                "transcript": transcript_result.get("transcript", ""),
                "transcript_method": transcript_result.get("method"),
                "transcript_reason": transcript_result.get("reason"),
                "transcript_cause": transcript_result.get("cause"),
                "chunks": chunks,
                "vector_store_info": vectorize_result
            }
            
            output_file = self._save_processed_data(video_id, output_data)
            
            # Return structured result
            return {
                "success": True,
                "video_id": video_id,
                "video_info": video_info,
                "transcript": transcript_result.get("transcript", ""),
                "transcript_method": transcript_result.get("method"),
                "transcript_reason": transcript_result.get("reason"),
                "transcript_cause": transcript_result.get("cause"),
                "chunks": chunks,
                "chunks_count": len(chunks),
                "vector_store_info": vectorize_result,
                "vector_store_key": vectorize_result.get("store_key"),
                "output_file": str(output_file)
            }
            
        except Exception as e:
            logger.error(f"Error processing uploaded video file: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}

    def _extract_transcript_from_file(self, file_path: str) -> Dict:
        """Extract transcript from uploaded video file"""
        try:
            # Method 1: Try Whisper transcription
            reason, cause = "", None
            try:
                logger.info(f"Attempting Whisper transcription for uploaded file: {file_path}")
                return self._get_whisper_transcript_direct(file_path)
            except Exception as e:
                logger.warning(f"Whisper transcription failed: {e}")
                reason, cause = _brief(e), _classify_transcript_failure(e)
            
            # Method 2: Create basic transcript based on file
            logger.info("Creating basic transcript for uploaded video...")
            return self._create_basic_transcript_for_file(file_path, reason, cause)
            
        except Exception as e:
            logger.error(f"All transcript methods failed for uploaded file: {str(e)}")
            return {"success": False, "error": str(e)}

    def _get_whisper_transcript_direct(self, file_path: str) -> Dict:
        """Get transcript using Whisper directly (without VideoProcessor dependency)"""
        try:
            # Try to use Whisper directly
            try:
                import whisper
                
                # Load Whisper model (use base model for balance of speed/accuracy)
                logger.info("Loading Whisper model...")
                model = whisper.load_model("base")
                
                logger.info(f"Transcribing video file: {file_path}")
                result = model.transcribe(file_path)
                
                # Format segments
                segments = []
                if 'segments' in result:
                    for segment in result['segments']:
                        segments.append({
                            'text': segment.get('text', ''),
                            'start': segment.get('start', 0),
                            'end': segment.get('end', 0),
                            'duration': segment.get('end', 0) - segment.get('start', 0)
                        })
                
                return {
                    "success": True,
                    "transcript": result.get("text", ""),
                    "segments": segments,
                    "method": "whisper_direct"
                }
                
            except ImportError:
                logger.warning("Whisper not installed, trying alternative method...")
                raise Exception("Whisper model not available")
                
        except Exception as e:
            logger.error(f"Direct Whisper transcription error: {str(e)}")
            raise Exception(f"Whisper transcription failed: {str(e)}")

    def _create_basic_transcript_for_file(self, file_path: str, reason: str = "",
                                          cause: str | None = None) -> Dict:
        """Create a basic transcript for uploaded video file when other methods fail.

        The text below is a fixed placeholder: it knows the filename and nothing else
        about the video. Downstream it is indistinguishable from a real transcript, so
        `method` and `reason` travel with it and `app.py::warn_if_synthetic_transcript`
        surfaces them -- otherwise a silent video yields confident, unrelated tests.
        """
        try:
            file_name = Path(file_path).name
            
            basic_transcript = f"""
            This is an uploaded video file analysis: {file_name}
            
            Video File Analysis:
            - This appears to be a demonstration or tutorial video
            - The uploaded video likely contains user interface interactions
            - Common elements may include navigation, form interactions, and workflows
            - The video demonstrates application functionality or user processes
            
            Key Interaction Areas:
            - Application navigation and menu interactions
            - Form filling and data entry processes
            - Button clicks and user interface elements
            - Workflow completion and task execution
            
            Testing Focus Areas:
            - User interface functionality testing
            - Form validation and error handling
            - Navigation and workflow testing
            - Cross-browser compatibility verification
            - Responsive design and accessibility testing
            """
            
            return {
                "success": True,
                "transcript": basic_transcript.strip(),
                "segments": [{"text": basic_transcript.strip(), "start": 0, "end": 180}],
                "method": "basic_fallback_file",
                "reason": reason,
                "cause": cause
            }
            
        except Exception as e:
            logger.error(f"Error creating basic transcript for file: {e}")
            return {
                "success": False,
                "error": f"Failed to create basic transcript: {str(e)}"
            }
    
    def store_path_for(self, store_key: str | None = None) -> Path:
        """Where one source's vectors live: src/data/vector_store/<key>/."""
        return self.data_dir / "vector_store" / (store_key or self.last_store_key)

    def chunk_and_vectorize(self, chunks: List[Dict], store_key: str | None = None,
                            source: str | None = None) -> Dict:
        """Embed chunks into the store for `store_key`, reusing it when unchanged."""
        try:
            # Prepare documents for vectorization
            documents = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk["text"],
                    metadata={
                        "chunk_id": chunk["chunk_id"],
                        "start_time": chunk["start_time"],
                        "end_time": chunk["end_time"],
                        "actions": chunk["actions"],
                        "chunk_type": chunk["chunk_type"]
                    }
                )
                documents.append(doc)
            
            # Create vector store
            if documents:
                store_key = store_key or DEFAULT_STORE_KEY
                self.last_store_key = store_key
                vector_store_path = self.store_path_for(store_key)
                fingerprint = chunks_fingerprint(chunks)

                reused = self._reuse_store(vector_store_path, fingerprint)
                if reused:
                    logger.info("Reusing vectors for '%s' -- chunks unchanged", store_key)
                    return reused

                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                logger.info(f"Created vector store with {len(documents)} documents")

                # Save vector store
                vector_store_path.mkdir(parents=True, exist_ok=True)
                self.vector_store.save_local(str(vector_store_path))

                attribution = self.embeddings.last_attribution
                if attribution:
                    embeddings_mod.write_store_meta(
                        vector_store_path, attribution, self.embeddings.dimensions,
                        source=source, content_hash=fingerprint,
                    )
                
                return {
                    "success": True,
                    "documents_count": len(documents),
                    "store_key": store_key,
                    "reused": False,
                    "vector_store_path": str(vector_store_path),
                    "embedding_attribution": attribution.to_dict() if attribution else None,
                    "embedding_dimensions": self.embeddings.dimensions
                }
            else:
                return {"success": False, "error": "No documents to vectorize"}
                
        except Exception as e:
            logger.error(f"Error in vectorization: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _reuse_store(self, vector_store_path: Path, fingerprint: str) -> Dict | None:
        """Load an existing store instead of re-embedding, when nothing has changed.

        Requires the *same* chunks (fingerprint) and the *same* embedding model and
        width -- a store built by a different model is not interchangeable even if
        the source text is identical.
        """
        stored = embeddings_mod.read_store_meta(vector_store_path) or {}
        if not stored.get("content_hash") or stored["content_hash"] != fingerprint:
            return None
        if stored.get("model") != self.embeddings.model:
            return None
        configured = embeddings_mod.model_dimensions(self.embeddings.model)
        if configured and stored.get("dimensions") and stored["dimensions"] != configured:
            return None
        try:
            self.vector_store = FAISS.load_local(
                str(vector_store_path), self.embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            logger.warning("Could not reuse %s, rebuilding: %s", vector_store_path, e)
            return None
        return {
            "success": True,
            "documents_count": self.vector_store.index.ntotal,
            "store_key": vector_store_path.name,
            "reused": True,
            "vector_store_path": str(vector_store_path),
            "embedding_attribution": {
                "provider": stored.get("provider"), "model": stored.get("model"),
                "display": f"Reused — {stored.get('model')} ({stored.get('written_at')})",
            },
            "embedding_dimensions": stored.get("dimensions"),
        }

    def setup_retrieval_chain(self, store_key: str | None = None):
        """Setup RAG retrieval chain for one source's store."""
        if store_key and store_key != self.last_store_key:
            self.vector_store = None  # a different video: do not serve the cached one
            self.last_store_key = store_key
        if self.vector_store is None:
            # Try to load existing vector store
            vector_store_path = self.store_path_for(store_key)
            if vector_store_path.exists():
                stored = embeddings_mod.read_store_meta(vector_store_path) or {}
                configured = embeddings_mod.model_dimensions(self.embeddings.model)
                if stored.get("dimensions") and configured and stored["dimensions"] != configured:
                    logger.error(
                        "Vector store was built with %s (%s dims) but %s produces %s dims. "
                        "Vectors of different widths are not comparable -- clear "
                        "src/data/vector_store/ and re-ingest.",
                        stored.get("model"), stored["dimensions"],
                        self.embeddings.model, configured,
                    )
                    return None
                try:
                    # langchain-community 0.3.27 requires this opt-in. Without it
                    # every cold-start load raised, the except below swallowed it,
                    # and a store was never reused across sessions.
                    # ponytail: trusts src/data/vector_store/ (pickle). It is written
                    # by this app into a gitignored local dir; if a store ever arrives
                    # from anywhere else, move the index to a non-pickle format.
                    self.vector_store = FAISS.load_local(
                        str(vector_store_path),
                        self.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    logger.info("Loaded existing vector store")
                except Exception as e:
                    logger.error(f"Failed to load vector store: {str(e)}")
                    return None
            else:
                logger.warning("No vector store found. Process a video first.")
                return None
        
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        return retriever
    
    def _save_processed_data(self, video_id: str, data: Dict) -> Path:
        """Save processed data to JSON file"""
        output_file = self.transcripts_dir / f"{video_id}_processed.json"
        
        # Remove non-serializable objects
        save_data = data.copy()
        if "vector_store" in save_data:
            del save_data["vector_store"]
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved processed data to: {output_file}")
        return output_file
    
    def search_content(self, query: str, k: int = 5) -> List[Dict]:
        """Search content using vector similarity"""
        if self.vector_store is None:
            retriever = self.setup_retrieval_chain()
            if retriever is None:
                return []
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in content search: {str(e)}")
            return []