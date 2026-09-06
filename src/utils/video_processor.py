import re
import json
from typing import Dict, List
from pathlib import Path
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import nltk

# Download required NLTK data
try:
    import nltk
    
    # Try to find punkt data, if not found, download it
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
    
    # Try to find stopwords data, if not found, download it  
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)
        
except Exception as e:
    print(f"Warning: NLTK setup failed: {e}")

from src.utils.logging_setup import get_logger

logger = get_logger(__name__, log_file="utils.log")

class VideoProcessor:
    def __init__(self):
        """Initialize Video Processor with Whisper model"""
        try:
            import whisper
            # Use the correct method name for loading Whisper model
            self.whisper_model = whisper.load_model("base")
            logger.info("✅ Whisper model loaded successfully")
        except AttributeError as e:
            logger.warning(f"⚠️ Whisper load_model method not found: {e}")
            self.whisper_model = None
        except Exception as e:
            logger.warning(f"⚠️ Could not load Whisper model: {e}")
            self.whisper_model = None
        
        # Action keywords for flow detection
        self.action_keywords = [
            "click", "select", "choose", "enter", "type", "input", "fill",
            "navigate", "go to", "open", "close", "save", "submit", "upload",
            "download", "scroll", "drag", "drop", "hover", "press", "tap",
            "login", "logout", "signin", "signup", "register", "search"
        ]
        
        # UI element keywords
        self.ui_keywords = [
            "button", "link", "menu", "dropdown", "form", "field", "input",
            "checkbox", "radio", "toggle", "slider", "tab", "modal", "popup",
            "dialog", "card", "panel", "sidebar", "header", "footer", "navbar"
        ]
        
        # Flow transition keywords
        self.transition_keywords = [
            "then", "next", "after", "now", "subsequently", "following",
            "once", "when", "if", "finally", "lastly", "first", "second"
        ]
        
        logger.info("VideoProcessor initialized successfully")
    
    def download_video(self, youtube_url: str, output_dir: str = "src/data/videos") -> Dict:
        """Download video from YouTube URL"""
        try:
            yt = YouTube(youtube_url)
            
            # Create safe filename
            safe_title = re.sub(r'[^\w\s-]', '', yt.title).strip()
            safe_title = re.sub(r'[-\s]+', '-', safe_title)
            
            # Download highest quality video
            stream = yt.streams.get_highest_resolution()
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            video_path = stream.download(
                output_path=str(output_path),
                filename=f"{safe_title}_{yt.video_id}.mp4"
            )
            
            logger.info(f"Downloaded video: {yt.title}")
            
            return {
                "success": True,
                "video_path": video_path,
                "title": yt.title,
                "video_id": yt.video_id,
                "duration": yt.length,
                "description": yt.description,
                "views": yt.views
            }
            
        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def extract_transcript(self, video_path: str, video_id: str = None) -> Dict:
        """Extract transcript using YouTube API or Whisper"""
        transcript_data = {"success": False, "transcript": "", "method": ""}
        
        # Try YouTube Transcript API first if video_id is available
        if video_id:
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_text = " ".join([entry['text'] for entry in transcript_list])
                
                transcript_data = {
                    "success": True,
                    "transcript": transcript_text,
                    "segments": transcript_list,
                    "method": "youtube_api"
                }
                logger.info("Transcript extracted using YouTube API")
                return transcript_data
                
            except Exception as e:
                logger.warning(f"YouTube API failed: {str(e)}, trying Whisper...")
        
        # Fallback to Whisper
        try:
            if self.whisper_model is None:
                raise Exception("Whisper model not available - install with: pip install openai-whisper")
            
            if not video_path:
                raise Exception("Video path required for Whisper transcription")
            
            logger.info("Attempting Whisper transcription...")
            result = self.whisper_model.transcribe(video_path)
            
            segments = []
            for segment in result.get("segments", []):
                segments.append({
                    "text": segment["text"].strip(),
                    "start": segment["start"],
                    "end": segment["end"]
                })
            
            transcript_data = {
                "success": True,
                "transcript": result["text"],
                "segments": segments,
                "method": "whisper"
            }
            logger.info("✅ Transcript extracted using Whisper")
            
        except Exception as e:
            logger.error(f"❌ Whisper transcription failed: {str(e)}")
            transcript_data = {
                "success": False, 
                "error": f"All transcription methods failed. Last error: {str(e)}"
            }
        
        return transcript_data
    
    def segment_transcript(self, transcript_data: Dict) -> List[Dict]:
        """Segment transcript into logical steps and flows"""
        if not transcript_data.get("success"):
            return []
        
        segments = transcript_data.get("segments", [])
        if not segments:
            # If no segments, create them from full transcript
            segments = self._create_segments_from_text(transcript_data["transcript"])
        
        logical_segments = []
        current_segment = {
            "text": "",
            "start_time": 0,
            "end_time": 0,
            "actions": [],
            "ui_elements": [],
            "step_type": "unknown",
            "confidence": 0.0
        }
        
        for i, segment in enumerate(segments):
            text = segment["text"].lower().strip()
            
            # Detect actions and UI elements
            actions = self._detect_actions(text)
            ui_elements = self._detect_ui_elements(text)
            
            # Check if this should start a new logical segment
            should_split = (
                self._is_new_action_sequence(text, current_segment["text"]) or
                self._is_topic_change(text) or
                len(current_segment["text"]) > 800
            )
            
            if should_split and current_segment["text"]:
                # Finalize current segment
                current_segment["confidence"] = self._calculate_confidence(current_segment)
                current_segment["step_type"] = self._classify_step_type(current_segment)
                logical_segments.append(current_segment.copy())
                
                # Start new segment
                current_segment = {
                    "text": segment["text"],
                    "start_time": segment.get("start", 0),
                    "end_time": segment.get("end", 0),
                    "actions": actions,
                    "ui_elements": ui_elements,
                    "step_type": "unknown",
                    "confidence": 0.0,
                    "segment_id": len(logical_segments)
                }
            else:
                # Extend current segment
                if not current_segment["text"]:
                    current_segment["start_time"] = segment.get("start", 0)
                
                current_segment["text"] += " " + segment["text"]
                current_segment["end_time"] = segment.get("end", 0)
                current_segment["actions"].extend(actions)
                current_segment["ui_elements"].extend(ui_elements)
        
        # Add final segment
        if current_segment["text"]:
            current_segment["confidence"] = self._calculate_confidence(current_segment)
            current_segment["step_type"] = self._classify_step_type(current_segment)
            current_segment["segment_id"] = len(logical_segments)
            logical_segments.append(current_segment)
        
        logger.info(f"Created {len(logical_segments)} logical segments")
        return logical_segments
    
    def extract_key_flows(self, segments: List[Dict]) -> List[Dict]:
        """Identify main user flows and actions from segments"""
        flows = []
        current_flow = {
            "flow_name": "",
            "description": "",
            "steps": [],
            "flow_type": "user_journey",
            "complexity": "medium",
            "estimated_time": 0
        }
        
        for segment in segments:
            # Identify flow boundaries
            if self._is_flow_start(segment):
                # Save current flow if it has steps
                if current_flow["steps"]:
                    current_flow = self._finalize_flow(current_flow)
                    flows.append(current_flow.copy())
                
                # Start new flow
                current_flow = {
                    "flow_name": self._extract_flow_name(segment),
                    "description": segment["text"][:100] + "...",
                    "steps": [self._segment_to_step(segment)],
                    "flow_type": self._classify_flow_type(segment),
                    "complexity": "medium",
                    "estimated_time": segment["end_time"] - segment["start_time"]
                }
            else:
                # Add step to current flow
                step = self._segment_to_step(segment)
                current_flow["steps"].append(step)
                current_flow["estimated_time"] += (segment["end_time"] - segment["start_time"])
        
        # Add final flow
        if current_flow["steps"]:
            current_flow = self._finalize_flow(current_flow)
            flows.append(current_flow)
        
        # Post-process flows
        flows = self._merge_related_flows(flows)
        flows = self._rank_flows_by_importance(flows)
        
        logger.info(f"Extracted {len(flows)} key user flows")
        return flows
    
    def _create_segments_from_text(self, text: str) -> List[Dict]:
        """Create segments from plain text using sentence tokenization"""
        from nltk.tokenize import sent_tokenize
        
        sentences = sent_tokenize(text)
        segments = []
        current_time = 0
        
        for sentence in sentences:
            # Estimate timing (rough approximation)
            duration = len(sentence.split()) * 0.5  # ~0.5 seconds per word
            
            segments.append({
                "text": sentence,
                "start": current_time,
                "end": current_time + duration
            })
            current_time += duration
        
        return segments
    
    def _detect_actions(self, text: str) -> List[str]:
        """Detect action keywords in text"""
        detected_actions = []
        text_lower = text.lower()
        
        for action in self.action_keywords:
            if action in text_lower:
                detected_actions.append(action)
        
        return list(set(detected_actions))  # Remove duplicates
    
    def _detect_ui_elements(self, text: str) -> List[str]:
        """Detect UI element keywords in text"""
        detected_elements = []
        text_lower = text.lower()
        
        for element in self.ui_keywords:
            if element in text_lower:
                detected_elements.append(element)
        
        return list(set(detected_elements))
    
    def _is_new_action_sequence(self, current_text: str, previous_text: str) -> bool:
        """Check if current text starts a new action sequence"""
        current_actions = self._detect_actions(current_text)
        
        # Check for transition words
        has_transition = any(keyword in current_text for keyword in self.transition_keywords)
        
        # Check for new actions
        has_new_action = bool(current_actions)
        
        return has_transition or (has_new_action and len(previous_text) > 200)
    
    def _is_topic_change(self, text: str) -> bool:
        """Detect topic changes in the text"""
        topic_indicators = [
            "now let's", "next we", "moving on", "switching to",
            "another way", "alternatively", "different approach",
            "let's try", "now we're going to"
        ]
        
        return any(indicator in text.lower() for indicator in topic_indicators)
    
    def _calculate_confidence(self, segment: Dict) -> float:
        """Calculate confidence score for a segment"""
        score = 0.0
        
        # Action presence increases confidence
        if segment["actions"]:
            score += 0.4
        
        # UI elements increase confidence
        if segment["ui_elements"]:
            score += 0.3
        
        # Reasonable length increases confidence
        text_length = len(segment["text"])
        if 50 <= text_length <= 500:
            score += 0.2
        
        # Clear structure increases confidence
        if any(word in segment["text"].lower() for word in ["step", "first", "then", "next"]):
            score += 0.1
        
        return min(score, 1.0)
    
    def _classify_step_type(self, segment: Dict) -> str:
        """Classify the type of step"""
        actions = segment["actions"]
        ui_elements = segment["ui_elements"]
        text = segment["text"].lower()
        
        if "login" in actions or "signin" in actions:
            return "authentication"
        elif "navigate" in actions or "go to" in actions:
            return "navigation"
        elif any(action in actions for action in ["click", "select", "choose"]):
            return "interaction"
        elif any(action in actions for action in ["type", "enter", "fill"]):
            return "data_entry"
        elif "form" in ui_elements or "input" in ui_elements:
            return "form_handling"
        elif any(word in text for word in ["verify", "check", "confirm", "validate"]):
            return "validation"
        else:
            return "general"
    
    def _is_flow_start(self, segment: Dict) -> bool:
        """Check if segment marks the start of a new flow"""
        text = segment["text"].lower()
        
        flow_starters = [
            "let's start", "first", "begin by", "to get started",
            "the first step", "initially", "we'll start"
        ]
        
        return any(starter in text for starter in flow_starters) or segment["step_type"] == "authentication"
    
    def _extract_flow_name(self, segment: Dict) -> str:
        """Extract a meaningful flow name from segment"""
        text = segment["text"]
        actions = segment["actions"]
        
        if "login" in actions:
            return "User Login Flow"
        elif "signup" in actions or "register" in actions:
            return "User Registration Flow"
        elif "search" in actions:
            return "Search Functionality"
        elif segment["step_type"] == "form_handling":
            return "Form Submission Flow"
        else:
            # Extract from first few words
            words = text.split()[:5]
            return " ".join(words).title() + " Flow"
    
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
            
            # Step 3: Chunk and vectorize
            transcript_text = transcript_result.get("transcript", "")
            if not transcript_text:
                return {"success": False, "error": "No transcript text found"}
            
            chunks = self._intelligent_chunking(transcript_text)
            vectorize_result = self.chunk_and_vectorize(chunks)
            
            # Step 4: Save processed data
            output_data = {
                "video_info": download_result,
                "transcript": transcript_text,
                "chunks": chunks,
                "vector_store_info": vectorize_result
            }
            
            output_file = self._save_processed_data(video_id, output_data)
            
            # Return structured result
            return {
                "success": True,
                "video_info": download_result,
                "transcript": transcript_text,
                "chunks": chunks,
                "vector_store_info": vectorize_result,
                "output_file": output_file
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}

    def _classify_flow_type(self, segment: Dict) -> str:
        """Classify the type of user flow"""
        actions = segment["actions"]
        step_type = segment["step_type"]
        
        if step_type == "authentication":
            return "authentication_flow"
        elif step_type == "form_handling":
            return "data_entry_flow"
        elif step_type == "navigation":
            return "navigation_flow"
        elif "search" in actions:
            return "search_flow"
        else:
            return "user_journey"
    
    def _segment_to_step(self, segment: Dict) -> Dict:
        """Convert segment to a test step format"""
        return {
            "step_number": segment.get("segment_id", 0) + 1,
            "description": segment["text"].strip(),
            "actions": segment["actions"],
            "ui_elements": segment["ui_elements"],
            "step_type": segment["step_type"],
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "confidence": segment["confidence"]
        }
    
    def _finalize_flow(self, flow: Dict) -> Dict:
        """Finalize flow with calculated metrics"""
        steps_count = len(flow["steps"])
        
        # Calculate complexity
        if steps_count <= 3:
            flow["complexity"] = "simple"
        elif steps_count <= 7:
            flow["complexity"] = "medium"
        else:
            flow["complexity"] = "complex"
        
        # Calculate average confidence
        if flow["steps"]:
            avg_confidence = sum(step["confidence"] for step in flow["steps"]) / len(flow["steps"])
            flow["confidence"] = avg_confidence
        
        return flow
    
    def _merge_related_flows(self, flows: List[Dict]) -> List[Dict]:
        """Merge flows that are closely related"""
        # Simple implementation - could be more sophisticated
        merged_flows = []
        
        for flow in flows:
            # Check if this flow can be merged with existing ones
            merged = False
            for existing_flow in merged_flows:
                if self._are_flows_related(flow, existing_flow):
                    # Merge flows
                    existing_flow["steps"].extend(flow["steps"])
                    existing_flow["estimated_time"] += flow["estimated_time"]
                    existing_flow["description"] += f" {flow['description']}"
                    merged = True
                    break
            
            if not merged:
                merged_flows.append(flow)
        
        return merged_flows
    
    def _are_flows_related(self, flow1: Dict, flow2: Dict) -> bool:
        """Check if two flows are related and can be merged"""
        # Simple heuristic - same flow type and similar timing
        same_type = flow1["flow_type"] == flow2["flow_type"]
        time_gap = abs(flow1["estimated_time"] - flow2["estimated_time"]) < 30  # 30 seconds
        
        return same_type and time_gap
    
    def _rank_flows_by_importance(self, flows: List[Dict]) -> List[Dict]:
        """Rank flows by importance/complexity"""
        def importance_score(flow):
            score = 0
            
            # More steps = higher importance
            score += len(flow["steps"]) * 2
            
            # Authentication and form flows are important
            if flow["flow_type"] in ["authentication_flow", "data_entry_flow"]:
                score += 10
            
            # Higher confidence = higher importance
            score += flow.get("confidence", 0) * 5
            
            return score
        
        return sorted(flows, key=importance_score, reverse=True)
    
    def process_video_complete(self, youtube_url: str) -> Dict:
        """Complete video processing pipeline"""
        try:
            logger.info(f"Starting complete video processing for: {youtube_url}")
            
            # Step 1: Download video
            download_result = self.download_video(youtube_url)
            if not download_result["success"]:
                return download_result
            
            # Step 2: Extract transcript
            transcript_result = self.extract_transcript(
                download_result["video_path"],
                download_result.get("video_id")
            )
            if not transcript_result["success"]:
                return transcript_result
            
            # Step 3: Segment transcript
            segments = self.segment_transcript(transcript_result)
            
            # Step 4: Extract flows
            flows = self.extract_key_flows(segments)
            
            # Compile results
            result = {
                "success": True,
                "video_info": download_result,
                "transcript": transcript_result,
                "segments": segments,
                "flows": flows,
                "statistics": {
                    "total_segments": len(segments),
                    "total_flows": len(flows),
                    "processing_time": transcript_result.get("estimated_time", 0),
                    "confidence_avg": sum(s["confidence"] for s in segments) / len(segments) if segments else 0
                }
            }
            
            # Save processed data
            self._save_processed_data(result, download_result.get("video_id", "unknown"))
            
            logger.info(f"Video processing completed: {len(segments)} segments, {len(flows)} flows")
            return result
            
        except Exception as e:
            logger.error(f"Error in complete video processing: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _save_processed_data(self, data: Dict, video_id: str):
        """Save processed video data"""
        output_dir = Path("src/data/transcripts")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{video_id}_processed.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processed data saved to: {output_file}")