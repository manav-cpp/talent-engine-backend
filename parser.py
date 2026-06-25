import os
import json
import urllib.request
import urllib.error
import logging
import time
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class ResumeParser:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("CRITICAL: GEMINI_API_KEY missing from environment.")
            
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"

    def _call_gemini_with_retries(self, payload: dict, max_retries: int = 3) -> str:
        """Handles internet requests with an enterprise retry loop."""
        req = urllib.request.Request(
            self.url, 
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )

        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=15) as response:
                    response_data = json.loads(response.read().decode('utf-8'))
                    return response_data['candidates'][0]['content']['parts'][0]['text']
            
            except urllib.error.HTTPError as e:
                error_info = e.read().decode('utf-8')
                logger.warning(f"Attempt {attempt + 1} Failed: HTTP {e.code} - {error_info}")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} Failed: {str(e)}")
            
            # Wait before trying again (Exponential backoff: 2s, 4s, 8s)
            if attempt < max_retries - 1:
                sleep_time = 2 ** (attempt + 1)
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                
        raise ConnectionError("Failed to communicate with AI Engine after multiple attempts.")

    def parse_resume(self, pdf_text: str, target_role: str, required_skills: str, target_experience: str) -> dict:
        
        # We explicitly tell Gemini to output application/json for guaranteed structure
        prompt = f"""
        You are an elite AI technical recruiter. Analyze the candidate resume against the Job Parameters.
        
        --- JOB PARAMETERS ---
        Target Role: {target_role}
        Required Skills: {required_skills}
        Minimum Experience Required: {target_experience} years
        
        --- CANDIDATE RESUME ---
        {pdf_text}
        
        --- SCORING RUBRIC ---
        Calculate a 'recent_activity_score' (0.00 to 1.00). 
        Perfect match = 0.85-1.00. Unqualified = 0.10-0.30.
        
        Respond ONLY with a raw JSON object. Do NOT wrap it in ```json blocks.
        {{
            "full_name": "Extracted Name",
            "current_role": "Their actual current job title",
            "skills": ["Skill 1", "Skill 2"],
            "years_experience": 5,
            "industry": "Extracted Industry",
            "recent_activity_score": 0.85 
        }}
        """

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0, # 0.0 enforces maximum determinism in production
                "response_mime_type": "application/json" # Google's native JSON mode
            }
        }
        
        try:
            raw_text = self._call_gemini_with_retries(payload)
            logger.debug(f"AI Raw Output: {raw_text}")
            
            # Since we used response_mime_type, it should be perfect JSON
            return json.loads(raw_text)
            
        except json.JSONDecodeError as e:
            logger.error(f"AI returned invalid JSON: {e} | Raw: {raw_text}")
            raise ValueError("AI Engine returned corrupted data.")
        except Exception as e:
            logger.error(f"Parser encountered a fatal error: {e}")
            raise