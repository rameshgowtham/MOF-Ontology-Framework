import os
import tempfile
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple, Any
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from threading import Lock
from pathlib import Path
import json
from io import BytesIO
import time
import random
import toml
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Move set_page_config to the top level, before any other Streamlit calls
st.set_page_config(page_title="MOF Insight  v2.0", layout="wide")

# Add global CSS for professional styling 
st.markdown("""
<style>
    /* Global professional font styling */
    body, h1, h2, h3, h4, h5, h6, p, li, span, button, input, textarea, select, label, .stButton>button, .stTextInput>div>div>input, .stSelectbox>div>div>select {
        font-family: 'Segoe UI', Arial, sans-serif !important;
        letter-spacing: 0.3px;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    h3 {
        margin-top: 8px;
    }
    
    /* Animation keyframes for all glowing elements */
    @keyframes glow {
        from {
            text-shadow: 0 0 5px #4299e1, 0 0 10px #4299e1;
        }
        to {
            text-shadow: 0 0 10px #4299e1, 0 0 20px #4299e1, 0 0 30px #4299e1;
        }
    }
    
    @keyframes glow-gray {
        from {
            text-shadow: 0 0 5px #cbd5e0, 0 0 10px #cbd5e0;
        }
        to {
            text-shadow: 0 0 10px #cbd5e0, 0 0 20px #cbd5e0, 0 0 30px #cbd5e0;
        }
    }
    
    @keyframes glow-orange {
        from {
            text-shadow: 0 0 5px #ed8936, 0 0 10px #ed8936;
        }
        to {
            text-shadow: 0 0 10px #ed8936, 0 0 20px #ed8936, 0 0 30px #ed8936;
        }
    }
    
    @keyframes glow-lite-blue {
        from {
            text-shadow: 0 0 5px #90cdf4, 0 0 10px #90cdf4;
        }
        to {
            text-shadow: 0 0 10px #90cdf4, 0 0 20px #90cdf4, 0 0 30px #90cdf4;
        }
    }
    
    @keyframes glow-yellow {
        from {
            text-shadow: 0 0 5px #ffd700, 0 0 10px #ffd700;
        }
        to {
            text-shadow: 0 0 10px #ffd700, 0 0 20px #ffd700, 0 0 30px #ffd700;
        }
    }
    
    /* Custom notification and prompt styles */
    .question-notification {
        background: #edf8fe !important;
        border-left: 4px solid #3182ce !important;
        padding: 7px 12px !important;
        margin: 8px 0 !important;
        border-radius: 6px !important;
        color: #000000 !important;
    }
    
    .question-notification * {
        color: #000000 !important;
    }
    
    .prompt-enhancement-panel {
        background-color: #3182ce !important;
        padding: 10px !important;
        border-radius: 5px !important;
        margin: 10px 0 !important;
    }
    
    .prompt-enhancement-panel strong {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

class APIKeyManager:
    """Thread-safe API key manager with rotation, failure handling, and key locking."""    
    def __init__(self, api_keys: List[str]):
        if not api_keys:
            raise ValueError("At least one API key must be provided")
        self.api_keys = api_keys
        self.current_index = 0
        self._failed_keys = set()
        self._lock = Lock()
        self._key_in_use = {key: False for key in api_keys}
    
    def acquire_key(self) -> Optional[str]:
        """Acquires and locks an available API key."""
        with self._lock:
            attempts = 0
            while attempts < len(self.api_keys):
                key = self.api_keys[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.api_keys)
                
                if key not in self._failed_keys and not self._key_in_use[key]:
                    self._key_in_use[key] = True
                    return key
                
                attempts += 1
            return None
    
    def release_key(self, key: str) -> None:
        """Releases a locked API key."""
        with self._lock:
            if key in self._key_in_use:
                self._key_in_use[key] = False
    
    def mark_key_failed(self, key: str) -> None:
        """Marks an API key as failed and releases it."""
        with self._lock:
            self._failed_keys.add(key)
            self._key_in_use[key] = False
            logger.warning(f"API key {key} marked as failed. Remaining keys: {len(self.api_keys) - len(self._failed_keys)}")

    def reset(self) -> None:
        """Resets the failed keys list and usage status."""
        with self._lock:
            self._failed_keys.clear()
            self._key_in_use = {key: False for key in self.api_keys}
            self.current_index = 0

class PromptEnhancer:
    """Enhances user prompts for more accurate MOF property extraction."""    
    def __init__(self, api_keys: List[str]):
        self.key_manager = APIKeyManager(api_keys)
        self._lock = Lock()
        self.current_key = None
        
        # Initialize the enhancement prompt template
        self.enhancement_prompt = PromptTemplate(
            input_variables=["user_prompt", "question"],
            template="""
            I want you to enhance this prompt for extracting specific information about Metal-Organic Frameworks (MOFs) from scientific papers.

            Original question: {question}
            
            User-provided prompt instructions: {user_prompt}
            
            Your task is to improve this prompt to make it more precise and effective for LLM-based information extraction. Focus on these aspects:
            
            1. Clarity: Make the extraction target crystal clear
            2. Format specification: Specify exact output format (units, numeric precision, separator style)
            3. Edge cases: Address how to handle missing information, multiple values, or conflicting data
            4. Context awareness: Add instructions for considering tables, figures, and specific sections
            5. Units standardization: Add instructions for converting to standard units
            6. Value consolidation: Instructions for handling multiple values (list separately, prioritize, etc.)
            
            Return only the enhanced prompt without any explanations or meta-commentary.
            """
        )
        
        self.initialize_chain()
    
    def initialize_chain(self) -> None:
        """Initializes the LLM chain with a valid API key."""
        key = self.key_manager.acquire_key()
        if key:
            with self._lock:
                self.current_key = key
                os.environ["GOOGLE_API_KEY"] = key
                self.llm = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def enhance_prompt(self, user_prompt: str, question: str) -> str:
        """Enhances a user-provided prompt with retry logic."""
        try:
            response = self.llm.invoke(
                self.enhancement_prompt.format(
                    user_prompt=user_prompt,
                    question=question
                )
            )
            
            if not isinstance(response, str):
                return user_prompt
            
            enhanced_prompt = response.strip()
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Error enhancing prompt: {e}")
            if "quota exceeded" in str(e).lower():
                if self.current_key:
                    self.key_manager.mark_key_failed(self.current_key)
                    self.key_manager.release_key(self.current_key)
                self.initialize_chain()
                raise
            return user_prompt
        finally:
            if self.current_key:
                self.key_manager.release_key(self.current_key)

def sanitize_text(text: str) -> str:
    """Remove markdown formatting from text."""
    if not text:
        return text
        
    # Remove markdown syntax commonly used for formatting
    # Remove asterisks (bold/italic)
    text = re.sub(r'\*+', '', text)
    # Remove underscores (italic/bold)
    text = re.sub(r'_+', '', text)
    # Remove backticks (code)
    text = re.sub(r'`+', '', text)
    # Remove hash symbols at start of lines (headers)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # Remove > for blockquotes
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^={3,}$', '', text, flags=re.MULTILINE)
    
    return text

class MOFResearchAnalyzer:
    """Thread-safe MOF research paper analyzer."""    
    def __init__(self, api_keys: List[str]):
        self.key_manager = APIKeyManager(api_keys)
        self._lock = Lock()
        self.current_key = None
        
        # Load predefined optimized questions
        self.optimized_questions = self._load_optimized_questions()
        
        # Initialize unified prompt before setting up the model
        self.unified_prompt = PromptTemplate(
            input_variables=["context", "question", "answer_type", "schema", "custom_prompt"],
            template="""
            Given the following MOF ontology schema:
                {schema}

                And the following context from a research paper:
                {context}

                Question: {question}
                Answer Type: {answer_type}
                
                {custom_prompt}

                Analyze both the schema and the document context to provide the answer.

            Rules for answer formatting:
            You are a research assistant analyzing a scientific paper. Provide only the specific answer to the question based on the given context. Do not include any additional explanations or unnecessary words.
            If a value is requested, return only the value with its unit. If the answer includes units, convert them to SI units before returning, ensuring all values in the column have the same units.
            The question has two parts:
            part 1: actual question
            part 2: Either String or Integer/Float
            once the answer is fetched, depending upon the part 2, modify the answer accordingly before returning.
            For example: 

            for the question 
            metal : What are all the metal ions in the ZIF-8 compound? (String),
            the answer should be Zn²⁺, Cu²⁺..

            for the question 
            porosity_nature : What is the porous nature of the ZIF-8 compound? (String)
            Extract the exact porous classification of ZIF-8 (microporous/mesoporous/macroporous) from this paper.
            - Return only the explicit classification(s) mentioned ("Microporous" or "Mesoporous" or "Macroporous").
        

            For the question:
            Porosity: What is the pore size of the ZIF-8 compound? (Integer/Float)

            Extract the exact pore size (in nanometers, nm) of the primary ZIF-8 compound from the specified paper. Pore size refers to the cavity diameter or aperture size of the pores, as determined by methods like X-ray diffraction, gas adsorption, or positron annihilation lifetime spectroscopy, and must be reported in nanometers (nm).

            - Only report the pore size for the primary ZIF-8 discussed in the paper, not reference samples, unrelated compounds, or other metrics (e.g., particle size, crystallite size).
            - Check tables, figures, and text in the paper for explicit pore size values.
            - Report only numeric values in nanometers (nm). If values are given in other units (e.g., Ångströms, Å), convert them to nm (e.g., 21 Å = 2.1 nm).
            - If variations of ZIF-8 (e.g., ZIF-8@CNF, mesoporous ZIF-8) are mentioned in the paper with distinct pore sizes, include their pore sizes in nm by mentioning "variations".
            - if multiple values are mentioned for different conditions then return all values separated by commas.

            for the question
            surface_area : What is the surface area of the ZIF-8 compound? (Integer/Float)
            - Extract surface area values of ZIF-8 and its variations from this paper.
            - Return only numeric values with their units (e.g., "1171.3 m²/g, 1867 m² g⁻¹")
            - Include all distinct values for different conditions (temperatures, synthesis methods, etc.), Separate multiple values with commas
            
            For the question:
            Dimension: What is the dimension of the ZIF-8 compound (say either 2D or 3D)? (String)

            Extract the dimension of the primary ZIF-8 compound from the specified paper, where dimension refers to whether the ZIF-8 structure is two-dimensional (2D) or three-dimensional (3D).

            - Only report the dimension for the primary ZIF-8 discussed in the paper, not reference samples, variations (e.g., ZIF-8@CNF, mesoporous ZIF-8), or unrelated compounds.
            - Check tables, figures, and text in the paper for an explicit statement of ZIF-8’s dimension as either "2D" or "3D" (e.g., described as a 2D layered structure or 3D framework).
            - Respond with "2D" if the paper explicitly states ZIF-8 is two-dimensional, or "3D" if it explicitly states ZIF-8 is three-dimensional.
            - Do not report numerical values (e.g., pore size, particle size, crystallite size in nm) or other metrics.

            for the question
            morphology : What is the morphology of the ZIF-8 compound? (String)
            Focus on explicitly stated morphological descriptions (e.g., flower-shaped, disc, rod, cube, nanosheet, nanoplates). Avoid general terms like "crystalline" unless specifically mentioned

            for the question 
            size : What is the size of the ZIF-8 compound? (Integer/Float)
            the answer should be a value like 270 nm, if any other unit is fetched, then change it to SI unit. Size and Pore size both are different. if size not found return "-". 
            If multiple values exist (due to different temperatures, synthesis methods,adsorbents or material types) list each of it separately by commas.

            Application: Identify the primary application from the title first, then verify within the document. If unclear, return "-",

            for the question
            Achievement: What is the achievement from the paper? (String)
            - Extract the specific quantitative achievements and applications of ZIF-8 from this paper's conclusion/results.
            - Focus on measured performance metrics (efficiency %, selectivity ratios, capacity values, etc.)
            - Include numerical improvements compared to benchmarks or previous work
            - Report specific application domains (gas separation, catalysis, sensing, etc.)
    

            For each question, follow these guidelines when extracting values:

            If multiple values exist (due to different temperatures, synthesis methods,adsorbents or material types), list each separately by commas, do not combine them into a range.
            If a value is given as a range, explicitly mention it as a "range" (e.g., "Range: 50 - 200 m²/g").
            If a value changes under a single condition, indicate the trend using:
            "INC" for increasing, "DEC" for decreasing,Example: INC 22.21 m²/g - 409.74 m²/g
            If multiple values exist from different conditions (e.g., different synthesis methods or material types), do not use "INC" or "DEC"—just list them separately.
    
            Ignore any information from supplementary material and only take data from the main paper.  

            Identifying Pure or Composite ZIF-8:
            - If the research paper mentions only "ZIF-8" without any other material, classify it as 'Pure". 
            - If terms like "composite," "hybrid," "ZIF-8/..." (e.g., ZIF-8/GO, ZIF-8@MOF)," "functionalized," "doped," or "ZIF-8-derived" appear, classify it as "Composite".  
            - If it is unclear, return "unclear".

            - Functionalization: If modifications like doping, grafting, or post-synthetic changes are mentioned, return "Functionalized", else "Non-functionalized".  

            If a value with a new unit is fetched, then try to convert it to the unit which is widely used for that question.

            Answer in only one word, except for multiple synthesis cases where multiple values are listed separately give all values separated by commas.
            
            Final Answer: """
            )
        
        self.initialize_chain()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len
        )
        
        self.documents = None
        self.texts = None
        self._schema = self._load_schema()
    
    @staticmethod
    def _load_optimized_questions() -> Dict:
        """Loads predefined optimized questions and prompts."""
        # Define optimized questions with descriptions
        optimized_questions = {
            "Metal Ions": {
                "question": "What are all the metal ions in the ZIF-8 compound?",
                "type": "String",
                "description": "Extracts metal ion components like Zn²⁺, Cu²⁺"
            },
            "Porosity Nature": {
                "question": "What is the porous nature of the ZIF-8 compound?",
                "type": "String",
                "description": "Identifies if MOF is microporous, mesoporous, or macroporous"
            },
            "Porosity": {
                "question": "What is the porosity of the ZIF-8 compound?",
                "type": "Integer/Float",
                "description": "Extracts porosity values in nanometers (nm)"
            },
            "Surface Area": {
                "question": "What is the surface area of the ZIF-8 compound?",
                "type": "Integer/Float",
                "description": "Extracts surface area with units (typically m²/g)"
            },
            "Dimension": {
                "question": "What is the dimension of the ZIF-8 compound (say either 2D or 3D)?",
                "type": "String",
                "description": "Identifies if MOF structure is 2D or 3D"
            },
            "Morphology": {
                "question": "What is the morphology of the ZIF-8 compound?",
                "type": "String",
                "description": "Extracts shape descriptions (e.g., flower-shaped, cube, rod)"
            },
            "Size": {
                "question": "What is the size of the ZIF-8 compound?",
                "type": "Integer/Float",
                "description": "Extracts particle size values in nanometers (nm)"
            },
            "Application": {
                "question": "What is the application of the ZIF-8 compound?",
                "type": "String",
                "description": "Identifies primary applications (e.g., gas separation, sensing)"
            },
            "Achievement": {
                "question": "What is the achievement from the paper?",
                "type": "String",
                "description": "Extracts key performance metrics and innovations"
            },
        }
        return optimized_questions
    
    
def main():
    # Load keys from .streamlit/secrets.toml
    try:
        with open(".streamlit/secrets.toml", "r") as f:
            secrets = toml.load(f)
        if "GOOGLE_API_KEYS" in secrets:
            api_keys = secrets["GOOGLE_API_KEYS"]
        else:
            # Fallback to dummy keys if GOOGLE_API_KEYS not found in secrets
            api_keys = ["dummy_key_1", "dummy_key_2"]
            st.warning("⚠️ GOOGLE_API_KEYS not found in secrets.toml. Using dummy keys for development.")
    except Exception as e:
        # For development testing only, use dummy keys (remove in production)
        api_keys = ["dummy_key_1", "dummy_key_2"]
        st.warning(f"⚠️ Error loading secrets.toml: {e}. Using dummy API keys for development.")
    
    analyzer = MOFResearchAnalyzer(api_keys)
    prompt_enhancer = PromptEnhancer(api_keys)
    
    # Add custom CSS for animations including page fade-in
    st.markdown("""
        <style>
        /* Page fade-in animation */
        @keyframes fadeIn {
            0% {
                opacity: 0;
                transform: translateY(20px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Apply fade-in to main container */
        .main {
            animation: fadeIn 1.5s ease-out;
        }

        /* Stagger children animations */
        .main > * {
            opacity: 0;
            animation: fadeIn 1.5s ease-out forwards;
        }

        .main > *:nth-child(1) { animation-delay: 0.2s; }
        .main > *:nth-child(2) { animation-delay: 0.4s; }
        .main > *:nth-child(3) { animation-delay: 0.6s; }
        .main > *:nth-child(4) { animation-delay: 0.8s; }
        .main > *:nth-child(5) { animation-delay: 1.0s; }

        /* Title sparkle effect */
        .title-sparkle {
            position: relative;
            display: inline-block;
        }
        
        .title-sparkle::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 50%;
            height: 100%;
            background: linear-gradient(
                90deg,
                transparent,
                rgba(255, 255, 255, 0.8),
                transparent
            );
            animation: none;
        }
        
        @keyframes sparkle {
            0% { left: -100%; }
            100% { left: 200%; }
        }
        
        h1 {
            overflow: hidden;
        }
        
        h1:hover {
            cursor: default;
        }
        
        h1:hover::before {
            animation: sparkle 2s infinite linear;
        }

        /* Professional question addition animation */
        @keyframes slideIn {
            0% {
                opacity: 0;
                transform: translateX(-20px);
            }
            100% {
                opacity: 1;
                transform: translateX(0);
            }
        }

        .question-added {
            background: #edf8fe; /* Light background color */
            border-left: 4px solid #3182ce; /* Left border color */
            margin: 8px 0; /* Reduced margin for closer spacing */
            height: auto; /* Allow height to adjust based on content */
            min-height: 2px; /* Set a minimum height for consistency */
            border-radius: 8px; /* Increased border-radius for a cuter look */
            box-shadow: 0 1px 2px rgba(0,0,0,0.1); /* Softer shadow for a subtle effect */
            animation: slideIn 0.5s ease-out forwards; /* Keep the animation */
            color: #161515; /* Text color */
            padding-left: 10px; /* Add padding to move text right */
        }

        .question-type {
            color: #3182ce;
            font-size: 1em;
            margin-top: 4px;
        }

        /* Enhanced question notification */
        @keyframes smoothAppear {
            0% {
                opacity: 0;
                transform: translateY(-10px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .question-notification {
            background: #edf8fe;
            border-left: 4px solid #3182ce;
            padding: 7px 12px;
            margin: 8px 0;
            border-radius: 6px;
            animation: smoothAppear 0.4s ease-out forwards;
            display: flex;
            align-items: center;
            gap: 11px;
            box-shadow: 0 2px 8px rgba(56, 161, 105, 0.1);
            max-width: 500px;
        }

        /* Tab List and Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
        }

        /* Remove default red line from tabs */
        .stTabs [data-baseweb="tab-highlight"] {
            display: none !important;
        }

        /* Base tab styling */
        .stTabs [data-baseweb="tab"] {
            height: 45px; 
            white-space: pre-wrap;
            background-color: #e3f2fd;
            border-radius: 15px;
            gap: 12px;
            padding: 10px 16px;
            width: auto;
            min-width: 160px;
            max-width: 300px;
            text-align: center;
            transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
            text-decoration: none;
            border: none;
            color: #2c5282;
            font-weight: bold;
            box-shadow: 0 2px 4px rgba(44, 82, 130, 0.15);
            position: relative;
            overflow: hidden;
        }

        /* Tab background animation */
        .stTabs [data-baseweb="tab"]::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                90deg,
                rgba(255, 255, 255, 0),
                rgba(255, 255, 255, 0.4),
                rgba(255, 255, 255, 0)
            );
            transition: all 0.6s;
        }

        /* Hover effect with animation */
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #b3e0ff;
            transform: translateY(-4px) scale(1.05);
            box-shadow: 0 5px 15px rgba(44, 82, 130, 0.25);
        }

        .stTabs [data-baseweb="tab"]:hover::before {
            left: 100%;
            transition: all 1s;
        }

        /* Selected (active) tab with glow effect - remove blue underline */
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #90caf9;
            font-weight: bold;
            color: #1e3a5f;
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(44, 82, 130, 0.3),
                        0 0 15px rgba(65, 105, 225, 0.5);
            border-bottom-color: transparent !important;
        }

        /* Remove the custom blue underline by removing the ::after content */
        .stTabs [data-baseweb="tab"][aria-selected="true"]::after {
            content: none;
        }

        /* Tab content fade in animation */
        .stTabs [role="tabpanel"] {
            animation: fadeIn 0.8s ease-out;
        }
        
        /* Results animation */
        [data-testid="stDataFrame"] {
            animation: fadeIn 1s ease-out;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
            border-radius: 10px;
            overflow: hidden;
        }
        
        /* Download button animation */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        button:has(div:contains("Download")) {
            animation: pulse 2s infinite;
        }

        /* Enhanced Optimized Question Card */
        .optimized-question-card {
            background: linear-gradient(135deg, #f5f9ff 0%, #edf7ff 100%);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.8);
            padding: 20px;
            margin: 15px 0;
            transition: all 0.3s ease;
            box-shadow: 
                0 4px 15px rgba(49, 130, 206, 0.1),
                inset 0 0 15px rgba(255, 255, 255, 0.5);
            position: relative;
            overflow: hidden;
        }
        
        .optimized-question-card:hover {
            transform: translateY(-5px);
            box-shadow: 
                0 8px 25px rgba(49, 130, 206, 0.15),
                inset 0 0 20px rgba(255, 255, 255, 0.8);
        }
        
        .optimized-question-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                90deg,
                transparent,
                rgba(255, 255, 255, 0.6),
                transparent
            );
            transition: 0.5s;
        }
        
        .optimized-question-card:hover::before {
            left: 100%;
        }
        
        .optimized-question-title {
            font-size: 1.1em;
            font-weight: 600;
            color: #2c5282;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 10px;
            border-bottom: 2px solid rgba(49, 130, 206, 0.1);
            padding-bottom: 8px;
        }
        
        .optimized-question-title .emoji {
            font-size: 1.2em;
            background: rgba(49, 130, 206, 0.1);
            padding: 5px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        .badge {
            background: linear-gradient(135deg, #3182ce 0%, #63b3ed 100%);
            color: white;
            font-size: 0.7em;
            padding: 3px 8px;
            border-radius: 20px;
            box-shadow: 0 2px 5px rgba(49, 130, 206, 0.2);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            animation: badgePulse 2s infinite;
        }
        
        @keyframes badgePulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .optimized-question-description {
            font-size: 0.95em;
            color: #4a5568;
            margin: 10px 0;
            line-height: 1.5;
            padding-left: 10px;
            border-left: 3px solid rgba(49, 130, 206, 0.2);
        }
        
        .question-text {
            background: rgba(49, 130, 206, 0.05);
            border-radius: 8px;
            padding: 10px 15px;
            margin-top: 10px;
            font-size: 0.9em;
            color: #2d3748;
            position: relative;
            transition: all 0.3s ease;
        }
        
        .question-text:hover {
            background: rgba(49, 130, 206, 0.1);
            transform: scale(1.01);
        }
        
        .question-text::before {
            content: '❝';
            position: absolute;
            left: -5px;
            top: -5px;
            font-size: 1.5em;
            color: rgba(49, 130, 206, 0.3);
        }
        
        .question-text::after {
            content: '❞';
            position: absolute;
            right: -5px;
            bottom: -5px;
            font-size: 1.5em;
            color: rgba(49, 130, 206, 0.3);
        }

        /* Add Button Enhancement */
        .add-button {
            background: linear-gradient(135deg, #3182ce 0%, #63b3ed 100%);
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(49, 130, 206, 0.2);
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        
        .add-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(49, 130, 206, 0.3);
        }
        
        .add-button:active {
            transform: translateY(0);
        }

        /* Question Icons */
        .question-icon {
            display: inline-block;
            width: 24px;
            height: 24px;
            background: rgba(49, 130, 206, 0.1);
            border-radius: 50%;
            text-align: center;
            line-height: 24px;
            margin-right: 8px;
        }

        /* Update the toast notification CSS */
        @keyframes slideInRight {
            0% {
                transform: translateX(100%);
                opacity: 0;
            }
            10% {
                transform: translateX(0);
                opacity: 1;
            }
            90% {
                transform: translateX(0);
                opacity: 1;
            }
            100% {
                transform: translateX(100%);
                opacity: 0;
            }
        }

        .toast-notification {
            position: fixed;
            top: 60px;  /* Increased from 20px to avoid Streamlit header */
            right: 20px;
            background: linear-gradient(135deg, #3182ce 0%, #63b3ed 100%);
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            z-index: 999999;  /* Increased z-index */
            animation: slideInRight 0.3s ease-in;
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 16px;  /* Added explicit font size */
            min-width: 200px;  /* Added minimum width */
            max-width: 400px;  /* Added maximum width */
        }

        .toast-notification .icon {
            font-size: 1.2em;
        }

        /* Add container for better positioning */
        .toast-container {
            position: relative;
            width: 100%;
            height: 0;
            overflow: visible;
        }

        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        @keyframes slideInLeft {
            0% { transform: translateX(-100px); opacity: 0; }
            100% { transform: translateX(0); opacity: 1; }
        }
        
        @keyframes slideInRight {
            0% { transform: translateX(100px); opacity: 0; }
            100% { transform: translateX(0); opacity: 1; }
        }
        
        @keyframes fadeInUp {
            0% { transform: translateY(20px); opacity: 0; }
            100% { transform: translateY(0); opacity: 1; }
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        @keyframes gradientFlow {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .hero-section {
            background: linear-gradient(-45deg, #1a365d, #2c5282, #2b6cb0, #3182ce);
            background-size: 400% 400%;
            animation: gradientFlow 15s ease infinite;
            padding: 2.5rem;
            border-radius: 1rem;
            margin-bottom: 2.5rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.12);
            position: relative;
            overflow: hidden;
        }
        
        .floating-icon {
            animation: float 3s ease-in-out infinite;
        }
        
        .slide-in-left {
            animation: slideInLeft 1s ease-out forwards;
        }
        
        .slide-in-right {
            animation: slideInRight 1s ease-out forwards;
        }
        
        .fade-in-up {
            animation: fadeInUp 1s ease-out forwards;
        }
        
        .pulse-effect {
            animation: pulse 2s infinite;
        }
        
        .metric-card {
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        }

        .text-gradient {
            background: linear-gradient(120deg, #ffffff 0%, #e2e8f0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }

        .glass-effect {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }

        @keyframes techPulse {
            0% { box-shadow: 0 0 0 0 rgba(49, 130, 206, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(49, 130, 206, 0); }
            100% { box-shadow: 0 0 0 0 rgba(49, 130, 206, 0); }
        }

        @keyframes dataFlow {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

        .tech-timeline {
                position: relative;
            padding: 2rem;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 1rem;
                overflow: hidden;
            }
            
        .tech-timeline::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(45deg, 
                rgba(49, 130, 206, 0.1) 0%,
                rgba(49, 130, 206, 0) 50%,
                rgba(49, 130, 206, 0.1) 100%);
            animation: dataFlow 3s linear infinite;
        }

        .tech-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .tech-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg,
                transparent,
                rgba(255, 255, 255, 0.2),
                transparent);
            transform: translateX(-100%);
            transition: 0.5s;
        }

        .tech-card:hover::before {
            transform: translateX(100%);
        }

        .tech-stat {
            position: relative;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            animation: techPulse 2s infinite;
        }

        .matrix-bg {
            position: relative;
            overflow: hidden;
        }

        .matrix-bg::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(45deg,
                transparent 0%,
                rgba(49, 130, 206, 0.1) 50%,
                transparent 100%);
            animation: dataFlow 20s linear infinite;
        }

        .system-architecture {
                    display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            padding: 1rem;
            background: rgba(0, 0, 0, 0.05);
            border-radius: 1rem;
        }

        .architecture-node {
            background: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }

        .architecture-node:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
        }

        .tech-timeline-item {
            position: relative;
            padding-left: 2rem;
            margin-bottom: 2rem;
            opacity: 0;
            transform: translateX(-20px);
            animation: slideInLeft 0.5s ease forwards;
        }

        .tech-timeline-item::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #3182ce;
            box-shadow: 0 0 0 4px rgba(49, 130, 206, 0.2);
        }

        .tech-timeline-item::after {
            content: '';
            position: absolute;
            left: 5px;
            top: 12px;
            width: 2px;
            height: calc(100% + 10px);
            background: linear-gradient(to bottom, #3182ce 0%, transparent 100%);
        }

        /* File Upload Button Animations */
        [data-testid="stFileUploader"] {
            transition: all 0.3s ease;
        }

        /* Container styling */
        [data-testid="stFileUploader"] > section {
            border: none;
            border-radius: 20px;
            padding: 25px;
            background: linear-gradient(145deg, #1a365d 0%, #2c5282 100%);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }

        /* Hover effect */
        [data-testid="stFileUploader"] > section:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(49, 130, 206, 0.25);
            background: linear-gradient(145deg, #2c5282 0%, #3182ce 100%);
        }

        /* Click/Active effect */
        [data-testid="stFileUploader"] > section:active {
            transform: translateY(0px);
            box-shadow: 0 4px 15px rgba(49, 130, 206, 0.15);
        }

        /* Ripple effect on click */
        [data-testid="stFileUploader"] > section::after {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            background: rgba(66, 153, 225, 0.3);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            transition: width 0.6s ease-out, height 0.6s ease-out, opacity 0.6s ease-out;
            opacity: 0;
        }

        [data-testid="stFileUploader"] > section:active::after {
            width: 200%;
            height: 200%;
            opacity: 1;
        }

        /* Browse files button styling */
        [data-testid="stFileUploader"] button[kind="secondary"] {
            background: linear-gradient(135deg, #4299e1 0%, #63b3ed 100%);
            color: #e2e8f0;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 25px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(49, 130, 206, 0.2);
            font-weight: 500;
        }

        [data-testid="stFileUploader"] button[kind="secondary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(49, 130, 206, 0.3);
            background: linear-gradient(135deg, #3182ce 0%, #4299e1 100%);
        }

        [data-testid="stFileUploader"] button[kind="secondary"]:active {
            transform: translateY(0);
            box-shadow: 0 4px 12px rgba(49, 130, 206, 0.2);
        }

        /* Drag and drop text styling */
        [data-testid="stFileUploader"] div[data-testid="stMarkdownContainer"] p {
            transition: all 0.3s ease;
            position: relative;
            color: #e2e8f0;
            font-size: 1.1em;
        }

        [data-testid="stFileUploader"]:hover div[data-testid="stMarkdownContainer"] p {
            color: #f7fafc;
            transform: scale(1.02);
        }

        /* File upload progress animation */
        [data-testid="stFileUploader"] progress {
            height: 8px;
            border-radius: 4px;
            background-color: rgba(226, 232, 240, 0.2);
            transition: all 0.3s ease;
        }

        [data-testid="stFileUploader"] progress::-webkit-progress-value {
            background: linear-gradient(90deg, #4299e1 0%, #63b3ed 100%);
            border-radius: 4px;
            transition: all 0.3s ease;
        }

        [data-testid="stFileUploader"] progress::-webkit-progress-bar {
            background-color: rgba(226, 232, 240, 0.2);
            border-radius: 4px;
        }

        /* File list animation and styling */
        [data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"] > div {
            animation: fadeInUp 0.5s ease-out forwards;
            color: #e2e8f0;
            background: rgba(66, 153, 225, 0.1);
            border-radius: 15px;
            padding: 8px 12px;
            margin: 4px 0;
        }

        /* Uploaded file name styling */
        [data-testid="stFileUploader"] span {
            color: #e2e8f0;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Add glow effect on hover */
        [data-testid="stFileUploader"] > section:hover {
            box-shadow: 0 8px 25px rgba(66, 153, 225, 0.3),
                        0 0 50px rgba(66, 153, 225, 0.1);
        }

        /* Remove default file icon color */
        [data-testid="stFileUploader"] svg {
            color: #63b3ed !important;
        }

        /* Style the close/remove button */
        [data-testid="stFileUploader"] button[data-testid="stFileUploadedDeleteButton"] {
            color: #e2e8f0;
            background: rgba(66, 153, 225, 0.2);
            border-radius: 50%;
            padding: 4px;
            transition: all 0.3s ease;
        }

        [data-testid="stFileUploader"] button[data-testid="stFileUploadedDeleteButton"]:hover {
            background: rgba(66, 153, 225, 0.4);
            transform: scale(1.1);
        }

        /* Global professional font styling */
        body, h1, h2, h3, h4, h5, h6, p, li, span, button, input, textarea, select, label {
            font-family: 'Segoe UI', Arial, sans-serif !important;
            letter-spacing: 0.3px;
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        
        .custom-question-container {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .emoji-container {
            position: relative;
            margin-right: 10px;
        }
        .glowing-bulb {
            font-size: 24px;
            position: relative;
            z-index: 1;
        }
        .glowing-bulb::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(255, 255, 0, 0.3);  /* Reduced opacity from 0.5 to 0.3 */
            border-radius: 50%;
            filter: blur(4px);  /* Reduced blur from 8px to 4px */
            z-index: -1;
            animation: bulbGlow 3s infinite;  /* Slowed down animation */
        }
        @keyframes bulbGlow {
            0% { opacity: 0.2; transform: scale(0.9); }  /* Less dramatic scale changes */
            50% { opacity: 0.4; transform: scale(1.1); }  /* Reduced maximum glow */
            100% { opacity: 0.2; transform: scale(0.9); }
        }
        </style>
    """, unsafe_allow_html=True)

    # Wrap the entire content in a main div for fade-in
    st.markdown('<div class="main">', unsafe_allow_html=True)

    st.markdown("""
<div style="position: relative; margin-bottom: 20px;">
    <h1 class="title-sparkle" style="position: relative; display: inline-block; padding: 0 10px; text-align: left; cursor: default;">
        MOF Insight
        <div style="
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, transparent, rgba(49, 130, 206, 0.7), transparent);
            animation: footerGlow 3s infinite;
        "></div>
    </h1>
</div>

<style>
    /* Animation for the underline glow */
    @keyframes footerGlow {
        0% { width: 0; left: 0; }
        50% { width: 100%; left: 0; }
        100% { width: 0; left: 100%; }
    }

    /* Base styles for title-sparkle with high specificity */
    h1.title-sparkle {
        cursor: default !important;
        text-decoration: none !important;
        color: inherit !important;
        pointer-events: none !important; /* Prevents any click/hover events */
    }

    /* Explicitly disable hover effects */
    h1.title-sparkle:hover {
        cursor: default !important;
        text-decoration: none !important;
        color: inherit !important;
        background: none !important;
    }

    /* Ensure no link behavior from parent or Streamlit overrides */
    div h1.title-sparkle, div h1.title-sparkle:hover {
        cursor: default !important;
        text-decoration: none !important;
        color: inherit !important;
    }

    /* Override any potential Streamlit link styling */
    a, a:hover {
        text-decoration: none !important;
        cursor: default !important;
    }
</style>
""", unsafe_allow_html=True)
    
    # Tabs for the UI sections with icons
    tab1, tab2, tab3 = st.tabs([
        "🕹️ Parameters Configuration", 
        "📄 Paper Processing & Analysis", 
        "ℹ️ About MOF Insight"
    ])

    with tab1:
        # Add some blank space at the top
        st.write("")  # This adds one line of space
        
        # Initialize session state variables if they don't exist
        if "questions" not in st.session_state:
            st.session_state.questions = []
            st.session_state.custom_questions = []
            
        # Two column layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Place the styled heading in the first column
            st.markdown("""
            <h4 style="position: relative; display: inline-block; text-align: left; margin-top: 10px; margin-bottom: 20px;">
                <span style="
                    color: #63b3ed; 
                    margin-right: 8px;
                    text-shadow: 0 0 5px #63b3ed, 0 0 10px #4299e1, 0 0 15px #3182ce;
                    animation: starPulse 2s infinite ease-in-out;
                ">✦</span>Set Up Your Analysis Parameters
            </h4>
            <style>
                @keyframes starPulse {
                    0% { text-shadow: 0 0 5px #63b3ed, 0 0 10px #4299e1, 0 0 15px #3182ce; }
                    50% { text-shadow: 0 0 10px #63b3ed, 0 0 15px #4299e1, 0 0 25px #3182ce; }
                    100% { text-shadow: 0 0 5px #63b3ed, 0 0 10px #4299e1, 0 0 15px #3182ce; }
                }
            </style>
            """, unsafe_allow_html=True)
            
            with st.expander("View Essential Queries", expanded=False):
                # Styled text for the introduction
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #f0f9ff 0%, #e6f2ff 100%);
                    border-left: 3px solid #63b3ed;
                    border-radius: 8px;
                    padding: 12px 15px;
                    margin: 15px 0;
                    box-shadow: 0 2px 5px rgba(49, 130, 206, 0.1);
                    font-size: 0.95em;
                    line-height: 1.5;
                    color: #2d3748;
                    font-style: italic;
                    position: relative;
                    overflow: hidden;
                ">
                    <span style="
                        font-weight: 500;
                        background: linear-gradient(90deg, #3182ce, #63b3ed);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        display: inline;
                    ">✨ Insight:</span> 
                    The following questions are strategically refined for high-value insights and precise extraction—you can always edit them anytime!
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("✔️ Add All Questions", key="add_all"):
                    for key, q_info in analyzer.optimized_questions.items():
                        if not any(q.get("key") == key for q in st.session_state.questions):
                            st.session_state.questions.append({
                                "key": key,
                                "column_name": key,
                                "question": q_info["question"],
                                "type": q_info["type"],
                                "optimized": True
                            })
                    # Use st.empty() to create placeholders for both toasts
                    first_toast = st.empty()
                    second_toast = st.empty()
                    
                    # Show first toast
                    first_toast.markdown(f"""
                        <div class="toast-container">
                            <div class="toast-notification">
                                <span class="icon">✨</span>
                                <span>Added all optimized questions</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    time.sleep(3)  # Wait for first toast
                    
                    # Clear first toast
                    first_toast.empty()
                    
                    # Add second toast notification
                    second_toast.markdown("""
                        <div class="toast-container">
                            <div class="toast-notification" style="background: linear-gradient(135deg, #805AD5 0%, #B794F4 100%);">
                                <span class="icon">💡</span>
                                <span>Feel free to ask Customised questions too!</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    time.sleep(2)  # Wait for second toast
                    
                    # Clear the second toast
                    second_toast.empty()
                
                # Dictionary of emojis for each question type
                question_emojis = {
                        "Metal Ions": "🧪",         # Represents chemistry and metal ions
                        "Porosity Nature": "🌫️",   # Represents the nature of porous structures
                        "Porosity": "🕳️",         # Represents holes or porosity
                        "Surface Area": "📏",       # Represents measurement and area
                        "Dimension": "📐",          # Represents different dimensions
                        "Morphology": "🔬",        # Represents microscopic structure analysis
                        "Size": "📊",              # Represents size comparison
                        "Application": "🚀",       # Represents practical usage and innovation
                        "Achievement": "🥇"        # Represents success and accomplishments
                }
                
                for key, q_info in analyzer.optimized_questions.items():
                    st.markdown(f"""
                        <div class="optimized-question-card">
                            <div class="optimized-question-title">
                                <span class="emoji">{question_emojis.get(key, '🔹')}</span>
                                {key}
                                <span class="badge">{q_info['type']}</span>
                            </div>
                            <div class="optimized-question-description">
                                {q_info['description']}
                            </div>
                            <div class="question-text">
                                {q_info['question']}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Add button in a more subtle way
                    if st.button("➕ Add", key=f"add_opt_{key}"):
                        if not any(q.get("key") == key for q in st.session_state.questions):
                            st.session_state.questions.append({
                                "key": key,
                                "column_name": key,
                                "question": q_info["question"],
                                "type": q_info["type"],
                                "optimized": True
                            })
                            st.success(f"Added '{key}' to your analysis parameters")
        
        with col2:
            # Add the same amount of spacing before Custom Questions
            st.markdown("""
            <div class="custom-question-container">
                <div class="emoji-container">
                    <div class="glowing-bulb">💡</div>
                </div>
                <h3>Custom Questions</h3>
            </div>
            
            <style>
                /* Global professional font styling */
                body, h1, h2, h3, h4, h5, h6, p, li, span, button, input, textarea, select, label {
                    font-family: 'Segoe UI', Arial, sans-serif !important;
                    letter-spacing: 0.3px;
                }
                
                h1, h2, h3, h4, h5, h6 {
                    font-weight: 600;
                    letter-spacing: 0.5px;
                }
                
                h3 {
                    margin-top: 8px;
                }
                
                .custom-question-container {
                    display: flex;
                    align-items: center;
                    margin-bottom: 10px;
                }
                .emoji-container {
                    position: relative;
                    margin-right: 10px;
                }
                .glowing-bulb {
                    font-size: 24px;
                    position: relative;
                    z-index: 1;
                }
                .glowing-bulb::after {
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(255, 255, 0, 0.3);  /* Reduced opacity from 0.5 to 0.3 */
                    border-radius: 50%;
                    filter: blur(4px);  /* Reduced blur from 8px to 4px */
                    z-index: -1;
                    animation: bulbGlow 3s infinite;  /* Slowed down animation */
                }
                @keyframes bulbGlow {
                    0% { opacity: 0.2; transform: scale(0.9); }  /* Less dramatic scale changes */
                    50% { opacity: 0.4; transform: scale(1.1); }  /* Reduced maximum glow */
                    100% { opacity: 0.2; transform: scale(0.9); }
                }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown("Add your own custom questions with enhanced prompts:")

            # Add custom question form
            with st.form("custom_question_form"):
                custom_key = st.text_input("Parameter Name:")
                custom_question = st.text_area("Your Question:")
                custom_type = st.selectbox("Answer Type:", ["String", "Integer/Float"])
                custom_prompt = st.text_area("Custom Prompt Instructions:", 
                                   value=st.session_state.get("custom_prompt", ""),
                                   help="Give a hint at extraction and format—AI handles the rest!")
                
                # Random Pro Tip with only glowing border, original background
                prompt_tips = [
                    "Be specific about output format, including units and precision",
                    "Specify how to handle missing or conflicting information",
                    "Instruct to check tables and figures, not just main text",
                    "Include instructions for converting to standard units",
                    "Specify how to handle multiple values (list separately, average, etc.)",
                    "Be explicit about required context (e.g., 'only consider the primary MOF')",
                    "Clarify how to handle variations or derivatives of the material",
                    "Include examples of expected outputs for clarity",
                    "Specify numeric precision requirements when applicable",
                    "Include instructions for handling ambiguous terms"
                ]
                
                import random
                # Get a truly random tip
                random_tip = random.choice(prompt_tips)
                
                # Pro tip with glowing border only, original background and refresh button
                st.markdown(f"""
                    <style>
                        @keyframes borderGlow {{
                            0% {{ box-shadow: 0 0 5px #3182CE, 0 0 10px #3182CE; }}
                            50% {{ box-shadow: 0 0 10px #3182CE, 0 0 15px #3182CE; }}
                            100% {{ box-shadow: 0 0 5px #3182CE, 0 0 10px #3182CE; }}
                        }}

                        .pro-tip {{
                            background: rgba(49, 130, 206, 0.08);
                            border-left: 3px solid #3182CE;
                            border-radius: 10px;
                            padding: 12px 16px;
                            margin: 12px 0;
                            animation: borderGlow 2s infinite;
                            transition: transform 0.3s ease, box-shadow 0.3s ease;
                        }}

                        .pro-tip:hover {{
                            transform: scale(1.03);
                            box-shadow: 0 4px 12px rgba(49, 130, 206, 0.3);
                        }}
                        
                        .refresh-button {{
                            background: rgba(49, 130, 206, 0.2);
                            border: none;
                            color: #3182CE;
                            border-radius: 50%;
                            width: 24px;
                            height: 24px;
                            font-size: 14px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            cursor: pointer;
                            transition: all 0.3s ease;
                            margin-left: auto;
                        }}
                        
                        .refresh-button:hover {{
                            background: rgba(49, 130, 206, 0.4);
                            transform: rotate(180deg);
                        }}
                    </style>

                    <div class="pro-tip">
                        <div style="display: flex; align-items: flex-start; justify-content: space-between;">
                            <div style="display: flex; align-items: flex-start; gap: 10px;">
                                <div style="color: #3182CE; font-size: 16px; margin-top: 1px;">💡</div>
                                <div>
                                    <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; color: #3182CE; margin-bottom: 4px;">PRO TIP</div>
                                    <div style="color: white; font-size: 14px; line-height: 1.4;">{random_tip}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Add the refresh button outside of the form (as a regular Streamlit button)
                _,_, refresh_col2 = st.columns([5,2,0.75])
                with refresh_col2:
                    if st.form_submit_button("↻", help="Get a new tip for better insights!"):
                        pass
                
                # Check for tip refresh in query params - using the updated API
                if "tip_refresh" in st.query_params:
                    try:
                        new_index = int(st.query_params["tip_refresh"])
                        if new_index > st.session_state.tip_index:
                            st.session_state.tip_index = new_index
                            # Clear the query parameter to avoid infinite refreshes
                            st.query_params.clear()
                    except:
                        pass
                
                submit_col1, submit_col2 = st.columns([1, 1])
                with submit_col1:
                    submitted = st.form_submit_button(" ➕ Add Custom Question")
                with submit_col2:
                    enhance_prompt = st.form_submit_button("Enhance with AI 🌟")
                
                if submitted and custom_key and custom_question:
                    # Use enhanced prompt if available, otherwise use the regular custom prompt
                    if "enhanced_prompt" in st.session_state:
                        prompt_to_use = st.session_state.enhanced_prompt
                        # Clear the enhanced prompt from session state
                        del st.session_state.enhanced_prompt
                    else:
                        prompt_to_use = custom_prompt
                    
                    # Sanitize the prompt
                    clean_prompt = sanitize_text(prompt_to_use)
                    
                    # Add to custom questions list
                    st.session_state.questions.append({
                        "key": custom_key,
                        "column_name": custom_key,
                        "question": custom_question,
                        "type": custom_type,
                        "custom_prompt": clean_prompt,
                        "optimized": False
                    })
                    
                    st.markdown(f"""
                        <div class="question-notification">
                            <div class="notification-icon">✅</div>
                            <div>
                                <div class="success-label">Question added successfully</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                if enhance_prompt and custom_key and custom_question:
                    with st.spinner("Enhancing prompt with AI..."):
                        try:
                            enhanced_text = prompt_enhancer.enhance_prompt(
                                custom_prompt if custom_prompt else "Extract information about this property",
                                custom_question
                            )

                            # Sanitize the enhanced text to remove markdown formatting
                            enhanced_text_clean = sanitize_text(enhanced_text)

                            st.markdown("""
                                <div class="prompt-enhancement-panel">
                                    <strong>✨ The prompt has been enhanced for better results!</strong>
                                </div>
                            """, unsafe_allow_html=True)

                            # Display Enhanced Prompt with sanitized text and store in session state
                            st.text_area("Enhanced Prompt", enhanced_text_clean, height=300, key="enhanced_text")
                            
                            # Store the enhanced text in session state for later use
                            st.session_state.enhanced_prompt = enhanced_text_clean

                        except Exception as e:
                            st.error(f"Error enhancing prompt: {e}")

            
            # Display current questions for analysis
            if st.session_state.questions:
                st.markdown("### Current Analysis Parameters")
                for i, q in enumerate(st.session_state.questions):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.markdown(f"""
                            <div class="question-added">
                                <strong>{q["column_name"]}</strong> ({q["type"]})
                                <div>{q["question"]}</div>
                                {"<div style='color:#3182ce;margin-top:2px;font-size:0.8em;'><strong>✓ Optimized</strong></div>" if q.get("optimized") else ""}
                            </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        if st.button("✏️ Edit", key=f"edit_{i}"):
                            st.session_state.edit_index = i

                    with col3:
                        if st.button("🗑️ Remove", key=f"remove_{i}"):
                            st.session_state.questions.pop(i)
                
                # Edit question modal (simplified version with a form)
                if "edit_index" in st.session_state:
                    edit_idx = st.session_state.edit_index
                    if edit_idx < len(st.session_state.questions):
                        q = st.session_state.questions[edit_idx]
                        st.markdown("### Edit Question")
                        
                        with st.form("edit_question_form"):
                            edited_key = st.text_input("Parameter Name:", q["column_name"])
                            edited_question = st.text_area("Question:", q["question"])
                            edited_type = st.selectbox("Answer Type:", ["String", "Integer/Float"], 
                                                    index=0 if q["type"] == "String" else 1)
                            edited_prompt = st.text_area("Custom Prompt Instructions:", 
                                                    q.get("custom_prompt", ""))
                            
                            save_col, cancel_col = st.columns(2)
                            with save_col:
                                save = st.form_submit_button("Save Changes")
                            with cancel_col:
                                cancel = st.form_submit_button("Cancel")
                            
                            if save:
                                # Sanitize the edited prompt before saving
                                clean_prompt = sanitize_text(edited_prompt)
                                st.session_state.questions[edit_idx] = {
                                    "key": edited_key,
                                    "column_name": edited_key,
                                    "question": edited_question,
                                    "type": edited_type,
                                    "custom_prompt": clean_prompt,
                                    "optimized": q.get("optimized", False)
                                }
                                del st.session_state.edit_index
                                
                            if cancel:
                                del st.session_state.edit_index
                        
                
                # Additional actions for question list
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Clear All Questions"):
                        st.session_state.questions = []
                
                
                with col2:
                    pass
    
    with tab2:
        st.markdown("<h3 style=\"font-family: 'Segoe UI', Arial, sans-serif; font-weight: 600; letter-spacing: 0.5px;\"><span style=\"text-shadow: 0 0 5px #90cdf4, 0 0 10px #90cdf4, 0 0 15px #90cdf4; animation: glow-lite-blue 1.5s ease-in-out infinite alternate;\">🔍</span> Process Research Articles</h3>", unsafe_allow_html=True)
        
        # Add custom CSS for the glow effect
        st.markdown("""
            <style>
        .glow {
            text-shadow: 0 0 5px #8AB4F8, 0 0 10px #4285F4, 0 0 15px #1a73e8;
            animation: glow 2s ease-in-out infinite alternate;
            font-size: 0.85em;
            display: inline-block;
            vertical-align: middle;
        }
        
        @keyframes glow {
            from {
                text-shadow: 0 0 3px #8AB4F8, 0 0 8px #4285F4, 0 0 12px #1a73e8;
            }
            to {
                text-shadow: 0 0 6px #8AB4F8, 0 0 12px #4285F4, 0 0 18px #1a73e8;
            }
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Check if questions are added before showing the file uploader
        if not st.session_state.questions:
            st.warning("⚠️ Please add at least one question before uploading PDF files.")
        else:
            # File upload section
            uploaded_files = st.file_uploader("Upload PDF research papers", type="pdf", accept_multiple_files=True)
            
            # Process files
            if uploaded_files:
                if st.button("Start Analysis"):
                    # Start tracking total process time
                    process_start_time = time.time()
                    
                    with st.spinner("Processing files..."):
                        try:
                            # Process files in parallel
                            results = []
                            
                            # Prepare questions data with column_name included
                            questions_data = []
                            for q in st.session_state.questions:
                                q_data = {
                                    "column_name": q["column_name"],
                                    "question": q["question"],
                                    "type": q["type"]
                                }
                                if "custom_prompt" in q:
                                    q_data["custom_prompt"] = q["custom_prompt"]
                                questions_data.append(q_data)
                            
                            # Create a progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            with ThreadPoolExecutor(max_workers=min(5, len(uploaded_files))) as executor:
                                # Submit all tasks
                                future_to_file = {
                                    executor.submit(
                                        process_file, analyzer, file, questions_data
                                    ): file.name for file in uploaded_files
                                }
                                
                                # Process results as they complete
                                for i, future in enumerate(as_completed(future_to_file)):
                                    file_name = future_to_file[future]
                                    try:
                                        result = future.result()
                                        if result:
                                            results.append(result)
                                            status_text.text(f"Processed {i+1}/{len(uploaded_files)}: {file_name}")
                                        else:
                                            st.warning(f"Could not extract data from {file_name}")
                                    except Exception as e:
                                        st.error(f"Error processing {file_name}: {e}")
                                    
                                    # Update progress
                                    progress_bar.progress((i + 1) / len(uploaded_files))

                            if results:
                                # Convert to DataFrame
                                df = pd.DataFrame(results)
                                
                                # Display results
                                st.subheader("Analysis Results")
                                st.dataframe(df)

                                # Calculate and display total process time
                                process_end_time = time.time()
                                process_total_time = process_end_time - process_start_time
                                
                                # Display only the total process time
                                st.markdown(f"**Total Process Time:** {process_total_time:.2f} seconds")
                            
                                # Excel download
                                excel_buffer = pd.ExcelWriter("mof_analysis_results.xlsx", engine='xlsxwriter')
                                df.to_excel(excel_buffer, index=False, sheet_name='Results')
                                excel_data = BytesIO()
                                df.to_excel(excel_data, index=False)
                                excel_data.seek(0)
                                
                                st.download_button(
                                    "Download Excel Results",
                                    excel_data,
                                    "mof_analysis_results.xlsx",
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key='download-excel'
                                )
                                
                        except Exception as e:
                            st.error(f"Error during analysis: {e}")
                            
                            # Still display total time even if there was an error
                            process_end_time = time.time()
                            process_total_time = process_end_time - process_start_time
                            st.markdown(f"**Total Process Time:** {process_total_time:.2f} seconds")
                        
                            # Excel download
                            excel_buffer = pd.ExcelWriter("mof_analysis_results.xlsx", engine='xlsxwriter')
                            df.to_excel(excel_buffer, index=False, sheet_name='Results')
                            excel_data = BytesIO()
                            df.to_excel(excel_data, index=False)
                            excel_data.seek(0)
                            
                            st.download_button(
                                "Download Excel Results",
                                excel_data,
                                "mof_analysis_results.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key='download-excel'
                            )
                            
                        except Exception as e:
                            st.error(f"Error during analysis: {e}")
                            
                            # Still display total time even if there was an error
                            process_end_time = time.time()
                            process_total_time = process_end_time - process_start_time
                            st.markdown(f"**Total Process Time:** {process_total_time:.2f} seconds")
    
    
  

    with tab3:
        # Add the new creative technical timeline CSS
        st.markdown("""
            <style>
            /* ... existing animations ... */

            @keyframes techPulse {
                0% { box-shadow: 0 0 0 0 rgba(49, 130, 206, 0.4); }
                70% { box-shadow: 0 0 0 10px rgba(49, 130, 206, 0); }
                100% { box-shadow: 0 0 0 0 rgba(49, 130, 206, 0); }
            }

            @keyframes dataFlow {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

            .tech-timeline {
                position: relative;
                padding: 2rem;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 1rem;
                overflow: hidden;
            }

            .tech-timeline::before {
                content: '';
                position: absolute;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(45deg, 
                    rgba(49, 130, 206, 0.1) 0%,
                    rgba(49, 130, 206, 0) 50%,
                    rgba(49, 130, 206, 0.1) 100%);
                animation: dataFlow 3s linear infinite;
            }

            .tech-card {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 1rem;
                padding: 1.5rem;
                margin-bottom: 1rem;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }

            .tech-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg,
                    transparent,
                    rgba(255, 255, 255, 0.2),
                    transparent);
                transform: translateX(-100%);
                transition: 0.5s;
            }

            .tech-card:hover::before {
                transform: translateX(100%);
            }

            .tech-stat {
                position: relative;
                padding: 1rem;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 0.5rem;
                margin: 0.5rem 0;
                animation: techPulse 2s infinite;
            }

            .matrix-bg {
                position: relative;
                overflow: hidden;
            }

            .matrix-bg::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(45deg,
                    transparent 0%,
                    rgba(49, 130, 206, 0.1) 50%,
                    transparent 100%);
                animation: dataFlow 20s linear infinite;
            }

            .system-architecture {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1rem;
                padding: 1rem;
                background: rgba(0, 0, 0, 0.05);
                border-radius: 1rem;
            }

            .architecture-node {
                background: white;
                padding: 1rem;
                border-radius: 0.5rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
            }

            .architecture-node:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
            }

            .tech-timeline-item {
                position: relative;
                padding-left: 2rem;
                margin-bottom: 2rem;
                opacity: 0;
                transform: translateX(-20px);
                animation: slideInLeft 0.5s ease forwards;
            }

            .tech-timeline-item::before {
                content: '';
                position: absolute;
                left: 0;
                top: 0;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: #3182ce;
                box-shadow: 0 0 0 4px rgba(49, 130, 206, 0.2);
            }

            .tech-timeline-item::after {
                content: '';
                position: absolute;
                left: 5px;
                top: 12px;
                width: 2px;
                height: calc(100% + 10px);
                background: linear-gradient(to bottom, #3182ce 0%, transparent 100%);
            }
            </style>
        """, unsafe_allow_html=True)

        # Hero Section with Matrix-like background
        st.markdown("""
            <div class="hero-section matrix-bg">
                <div class="floating-icon">
                    <span style="
                        background: rgba(255,255,255,0.1);
                        padding: 0.5rem 1rem;
                        border-radius: 2rem;
                        color: #ffffff;
                    ">⚡ Version 2.0</span>
                </div>
                <h1 class="text-gradient">Advanced MOF Analytics Engine</h1>
                <div class="tech-card" style="max-width: 800px; margin: 2rem auto;">
                    <div style="color: #e2e8f0; font-family: 'Courier New', monospace;">
                        <span style="color: #4299e1;">class</span> <span style="color: #48bb78;">MOF Insight</span> {
                        <br>&nbsp;&nbsp;<span style="color: #4299e1;">function</span> extract() {
                        <br>&nbsp;&nbsp;&nbsp;&nbsp;return <span style="color: #f6ad55;">"State-of-the-art LLM-powered extraction"</span>;
                        <br>&nbsp;&nbsp;}
                        <br>}
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # System Architecture with Interactive Components
        st.markdown("""
            <div class="slide-in-left">
                <h2 style="
                    color:#4299e1;
                    margin-bottom: 1.5rem;
                    font-size: 2rem;
                    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
                ">
                    <span style="
                        text-shadow: 0 0 5px #cbd5e0, 0 0 10px #cbd5e0, 0 0 15px #cbd5e0;
                        animation: glow-gray 1.5s ease-in-out infinite alternate;
                    ">🔬</span> System Architecture
                </h2>
                <div class="system-architecture" style="
                    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                    padding: 2rem;
                    border-radius: 1rem;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                ">
                    <div class="architecture-node" style="
                        background: white;
                        padding: 1.5rem;
                        border-radius: 1rem;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    ">
                        <h4 style="
                            color: #2c5282;
                            font-size: 1.25rem;
                            margin-bottom: 1rem;
                            border-bottom: 2px solid #3182ce;
                            padding-bottom: 0.5rem;
                        ">Input Layer</h4>
                        <div class="tech-stat" style="
                            background: rgba(49, 130, 206, 0.1);
                            color: #2d3748;
                            font-weight: 500;
                        ">
                            <div style="margin-bottom: 0.5rem;">📄 Document Ingestion</div>
                            <ul style="
                                color: #4a5568;
                                list-style-type: none;
                                padding-left: 0;
                                font-size: 0.9rem;
                            ">
                                <li>• Dynamic PDF parsing</li>
                                <li>• Concurrent uploads</li>
                                <li>• Robust error logging</li>
                            </ul>
                        </div>
                        <div class="tech-stat" style="
                            background: rgba(49, 130, 206, 0.1);
                            color: #2d3748;
                            font-weight: 500;
                            margin-top: 1rem;
                        ">
                            <div style="margin-bottom: 0.5rem;">📊 Content Segmentation</div>
                            <ul style="
                                color: #4a5568;
                                list-style-type: none;
                                padding-left: 0;
                                font-size: 0.9rem;
                            ">
                                <li>• Adaptive text chunking</li>
                                <li>• Retains sectional context</li>
                                <li>• Supports diverse character encodings</li>
                            </ul>
                        </div>
                    </div>
                    <div class="architecture-node" style="
                        background: white;
                        padding: 1.5rem;
                        border-radius: 1rem;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    ">
                        <h4 style="
                            color: #2c5282;
                            font-size: 1.25rem;
                            margin-bottom: 1rem;
                            border-bottom: 2px solid #3182ce;
                            padding-bottom: 0.5rem;
                        ">Processing Core</h4>
                        <div class="tech-stat" style="
                            background: rgba(49, 130, 206, 0.1);
                            color: #2d3748;
                            font-weight: 500;
                        ">
                            <div style="margin-bottom: 0.5rem;">🧠 Language Processing Unit</div>
                            <ul style="
                                color: #4a5568;
                                list-style-type: none;
                                padding-left: 0;
                                font-size: 0.9rem;
                            ">
                                <li>• Seamless Gemini AI integration</li>
                                <li>• Context-aware prompt extraction</li>
                                <li>• Robust API handling</li>
                            </ul>
                        </div>
                        <div class="tech-stat" style="
                            background: rgba(49, 130, 206, 0.1);
                            color: #2d3748;
                            font-weight: 500;
                            margin-top: 1rem;
                        ">              
                            <div style="margin-bottom: 0.5rem;">🤖 Analysis Engine</div>
                            <ul style="
                                color: #4a5568;
                                list-style-type: none;
                                padding-left: 0;
                                font-size: 0.9rem;
                            ">
                                <li>• AI-driven prompt optimization</li>
                                <li>• Multi-threaded key management</li>
                                <li>• Strict data fidelity controls</li>
                            </ul>
                        </div>
                    </div>
                    <div class="architecture-node" style="
                        background: white;
                        padding: 1.5rem;
                        border-radius: 1rem;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    ">
                        <h4 style="
                            color: #2c5282;
                            font-size: 1.25rem;
                            margin-bottom: 1rem;
                            border-bottom: 2px solid #3182ce;
                            padding-bottom: 0.5rem;
                        ">Output Layer</h4>
                        <div class="tech-stat" style="
                            background: rgba(49, 130, 206, 0.1);
                            color: #2d3748;
                            font-weight: 500;
                        ">
                            <div style="margin-bottom: 0.5rem;">🎯 Result Compilation</div>
                            <ul style="
                                color: #4a5568;
                                list-style-type: none;
                                padding-left: 0;
                                font-size: 0.9rem;
                            ">
                                <li>• Structured output generation</li>
                                <li>• Standardized unit normalization</li>
                                <li>• Ontological framework alignment</li>
                            </ul>
                        </div>
                        <div class="tech-stat" style="
                            background: rgba(49, 130, 206, 0.1);
                            color: #2d3748;
                            font-weight: 500;
                            margin-top: 1rem;
                        ">
                            <div style="margin-bottom: 0.5rem;">📤 Data Delivery</div>
                            <ul style="
                                color: #4a5568;
                                list-style-type: none;
                                padding-left: 0;
                                font-size: 0.9rem;
                            ">
                                <li>• High-speed parallel processing</li>
                                <li>• Flexible exports with spreadsheet</li>
                                <li>• Real-time monitoring</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Technical Timeline with Animation
        st.markdown("""
            <div class="tech-timeline">
                <h2 style="color: #4299e1; margin-bottom: 2rem;">
                    <span style="
                        text-shadow: 0 0 5px #4299e1, 0 0 10px #4299e1, 0 0 15px #4299e1;
                        animation: glow 1.5s ease-in-out infinite alternate;
                    ">📈</span> Development Timeline
                </h2>
                <div class="tech-timeline-item" style="animation-delay: 0.2s;">
                    <div class="tech-card">
                        <h4 style="color: #3182ce;">Version 2.0 (Current)</h4>
                        <ul style="color: #4a5568; list-style-type: none; padding-left: 0;">
                            <li>• Implemented parallel processing architecture</li>
                            <li>• Enhanced Prompt Engineering integration</li>
                            <li>• Advanced error handling system</li>
                        </ul>
                        <div style="font-family: monospace; color: #718096; margin-top: 0.5rem;">
                            Status: <span style="color: #48bb78;">ACTIVE</span>
                        </div>
                    </div>
                </div>
                <div class="tech-timeline-item" style="animation-delay: 0.4s;">
                    <div class="tech-card">
                        <h4 style="color: #3182ce;">Version 1.5</h4>
                        <ul style="color: #4a5568; list-style-type: none; padding-left: 0;">
                            <li>• Optimized text processing</li>
                            <li>• Improved data extraction accuracy</li>
                        </ul>
                        <div style="font-family: monospace; color: #718096; margin-top: 0.5rem;">
                            Status: <span style="color: #ecc94b;">ARCHIVED</span>
                        </div>
                    </div>
                </div>
                <div class="tech-timeline-item" style="animation-delay: 0.6s;">
                    <div class="tech-card">
                        <h4 style="color: #3182ce;">Version 1.0</h4>
                        <ul style="color: #4a5568; list-style-type: none; padding-left: 0;">
                            <li>• Initial release</li>
                            <li>• Basic PDF processing</li>
                        </ul>
                        <div style="font-family: monospace; color: #718096; margin-top: 0.5rem;">
                            Status: <span style="color: #718096;">DEPRECATED</span>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Performance Metrics with Technical Style
        st.markdown("""
            <div class="slide-in-right" style="
                background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
                padding: 2rem;
                border-radius: 1rem;
                margin: 2rem 0;
            ">
                <h2 style="color: white; margin-bottom: 1.5rem;">
                    <span style="
                        text-shadow: 0 0 5px #ed8936, 0 0 10px #ed8936, 0 0 15px #ed8936;
                        animation: glow-orange 1.5s ease-in-out infinite alternate;
                    ">⚡</span> System Metrics
                </h2>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem;">
                    <div class="tech-card">
                        <div class="pulse-effect" style="font-size: 2rem; color: #48bb78;">
                            ~2s
                        </div>
                        <div style="color: #e2e8f0; font-family: monospace;">
                            Processing Time/Page
                        </div>
                        <div class="tech-stat" style="font-size: 0.8rem; color: #a0aec0;">
                            Parallel Processing Optimization
                        </div>
                    </div>
                    <div class="tech-card">
                        <div class="pulse-effect" style="font-size: 2rem; color: #4299e1;">
                            99.5%
                        </div>
                        <div style="color: #e2e8f0; font-family: monospace;">
                            System Uptime
                        </div>
                        <div class="tech-stat" style="font-size: 0.8rem; color: #a0aec0;">
                            Robust API Retry Mechanism
                        </div>
                    </div>
                    <div class="tech-card">
                        <div class="pulse-effect" style="font-size: 2rem; color: #f6ad55;">
                            4-6x
                        </div>
                        <div style="color: #e2e8f0; font-family: monospace;">
                            Speed Improvement
                        </div>
                        <div class="tech-stat" style="font-size: 0.8rem; color: #a0aec0;">
                            vs Sequential Manual Extraction
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Add professional courtesy footer at the end of the About section with bold styling and animation, no divider or background
        st.markdown("""
        <div style="
            margin-top: 40px;
            padding: 15px 0;
            text-align: center;
            font-size: 0.9rem;
            color: #2d3748;
            font-weight: 600;
            animation: fadeIn 1.5s ease-out;
            padding: 20px 0;
            text-shadow: 0 1px 2px rgba(0,0,0,0.05);
            ">
            <div style="
                display: inline-block;
                position: relative;
                overflow: hidden;
                padding: 0 10px;
            ">
                <span style="position: relative; z-index: 2;">© 2025 Amrita CBE. All Rights Reserved.</span>
                <div style="
                    position: absolute;
                    bottom: 0;
                    left: 0;
                    width: 100%;
                    height: 2px;
                    background: linear-gradient(90deg, transparent, rgba(49, 130, 206, 0.7), transparent);
                    animation: footerGlow 3s infinite;
                "></div>
            </div>
        </div>
        
        <style>
            @keyframes footerGlow {
                0% { width: 0; left: 0; }
                50% { width: 100%; left: 0; }
                100% { width: 0; left: 100%; }
            }
        </style>
        """, unsafe_allow_html=True)

    # Close the main div wrapper at the end of the function
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    

