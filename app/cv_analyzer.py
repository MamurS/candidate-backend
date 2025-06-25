#!/usr/bin/env python3
"""
CV Analyzer Module

A comprehensive tool for analyzing candidate résumés in PDF format and extracting
structured data using OpenAI's function calling capabilities.

Usage:
    from cv_analyzer import CVAnalyzer

    analyzer = CVAnalyzer()
    profile = analyzer.analyze("resume.pdf")

Or via CLI:
    python cv_analyzer.py ./cv.pdf --out ./profile.json --model gpt-4o-mini
"""

import argparse
import json
import logging
import os
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract
from pydantic import BaseModel, Field, EmailStr, ValidationError, TypeAdapter

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class WorkExperience(BaseModel):
    """Represents a work experience entry."""
    company: str
    position: str
    employer_location: Optional[str] = None  # e.g. "Berlin, DE"
    remote: bool
    start_date: date
    end_date: Optional[date] = None
    description: Optional[str] = None


class EducationItem(BaseModel):
    """Represents an education entry."""
    institution: str
    degree: str
    field_of_study: str
    graduation_date: Optional[date] = None


class CandidateProfile(BaseModel):
    """Complete candidate profile extracted from CV."""
    full_name: str
    email: EmailStr
    phone: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None
    summary: Optional[str] = None
    languages: Optional[List[str]] = Field(default_factory=list)
    skills: Optional[List[str]] = Field(default_factory=list)
    experience: Optional[List[WorkExperience]] = Field(default_factory=list)
    education: Optional[List[EducationItem]] = Field(default_factory=list)
    certifications: Optional[List[str]] = Field(default_factory=list)
    linkedin_url: Optional[str] = None


class CVAnalyzer:
    """
    Main class for analyzing candidate CVs using OpenAI function calling.

    This class handles PDF extraction, text processing, OpenAI API calls,
    and data validation to produce structured candidate profiles.
    """

    MAX_TOKENS = 15000  # Token limit before chunking

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the CV analyzer.

        Args:
            api_key: OpenAI API key. If None, will use OPENAI_API_KEY environment variable.
        """
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        logger.info("CVAnalyzer initialized")

    def analyze(self, pdf_path: str, *, model: str = "gpt-4o-mini") -> CandidateProfile:
        """
        Analyze a candidate's résumé from PDF and return structured profile data.

        Args:
            pdf_path: Path to the PDF résumé file
            model: OpenAI model to use for analysis

        Returns:
            CandidateProfile: Structured candidate data

        Raises:
            ValueError: If input is not a PDF file
            FileNotFoundError: If PDF file doesn't exist
            ValidationError: If extracted data doesn't match schema
            Exception: For other processing errors
        """
        logger.info(f"Starting analysis of {pdf_path} with model {model}")

        # Validate input
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError("Input file must be a PDF")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            # Extract text from PDF
            text = self._extract_pdf_text(pdf_path)
            logger.info(f"Extracted {len(text)} characters from PDF")

            # Process with OpenAI
            profile_data = self._process_with_openai(text, model)
            logger.info("Successfully processed text with OpenAI")

            # Validate and create profile
            profile = self._validate_and_create_profile(profile_data)
            logger.info("Successfully created and validated candidate profile")

            # Save to file
            self._save_profile(profile)
            logger.info("Profile saved to profile.json")

            return profile

        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            raise

    def _extract_pdf_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF using pdfplumber with pdfminer fallback.

        Args:
            pdf_path: Path to PDF file

        Returns:
            str: Extracted text content
        """
        logger.info("Attempting text extraction with pdfplumber")

        try:
            # Primary: pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

                text = "\n".join(text_parts)

                if text.strip():
                    logger.info("Successfully extracted text with pdfplumber")
                    return self._clean_text(text)

        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")

        # Fallback: pdfminer.six
        logger.info("Falling back to pdfminer.six")
        try:
            text = pdfminer_extract(pdf_path)
            if text.strip():
                logger.info("Successfully extracted text with pdfminer.six")
                return self._clean_text(text)
        except Exception as e:
            logger.error(f"pdfminer.six extraction failed: {e}")
            raise Exception("Failed to extract text from PDF with both pdfplumber and pdfminer.six")

        raise Exception("No text could be extracted from the PDF")

    def _clean_text(self, text: str) -> str:
        """
        Clean and sanitize extracted text.

        Args:
            text: Raw extracted text

        Returns:
            str: Cleaned text
        """
        # Remove embedded JavaScript (security requirement)
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)

        return text.strip()

    def _estimate_tokens(self, text: str) -> int:
        """
        Rough estimation of token count (approximately 4 chars per token).

        Args:
            text: Text to estimate

        Returns:
            int: Estimated token count
        """
        return len(text) // 4

    def _process_with_openai(self, text: str, model: str) -> Dict[str, Any]:
        """
        Process text with OpenAI using function calling.

        Args:
            text: Extracted CV text
            model: OpenAI model to use

        Returns:
            Dict: Extracted profile data
        """
        # Check if we need to chunk the text
        estimated_tokens = self._estimate_tokens(text)

        if estimated_tokens > self.MAX_TOKENS:
            logger.info(f"Text too long ({estimated_tokens} tokens), chunking required")
            return self._process_chunked_text(text, model)
        else:
            return self._make_openai_call(text, model)

    def _process_chunked_text(self, text: str, model: str) -> Dict[str, Any]:
        """
        Process large text by chunking and aggregating results.

        Args:
            text: Large text to process
            model: OpenAI model to use

        Returns:
            Dict: Aggregated profile data
        """
        # Simple chunking by splitting text
        chunk_size = self.MAX_TOKENS * 4  # Rough character estimate
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        logger.info(f"Processing {len(chunks)} chunks")

        results = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i + 1}/{len(chunks)}")
            result = self._make_openai_call(chunk, model)
            results.append(result)

        # Aggregate results (simple strategy: merge lists, keep first non-empty values)
        return self._aggregate_chunk_results(results)

    def _aggregate_chunk_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from multiple chunks.

        Args:
            results: List of individual chunk results

        Returns:
            Dict: Aggregated profile data
        """
        if not results:
            return {}

        # Start with first result
        aggregated = results[0].copy()

        # Merge subsequent results
        for result in results[1:]:
            # For lists, extend them
            for key in ['languages', 'skills', 'experience', 'education', 'certifications']:
                if key in result and isinstance(result[key], list):
                    if key not in aggregated:
                        aggregated[key] = []
                    aggregated[key].extend(result[key])

            # For strings, use first non-empty value
            for key in ['full_name', 'email', 'phone', 'country', 'city', 'summary', 'linkedin_url']:
                if key in result and result[key] and (key not in aggregated or not aggregated[key]):
                    aggregated[key] = result[key]

        # Deduplicate lists
        for key in ['languages', 'skills', 'certifications']:
            if key in aggregated and isinstance(aggregated[key], list):
                aggregated[key] = list(dict.fromkeys(aggregated[key]))  # Preserve order

        return aggregated

    def _make_openai_call(self, text: str, model: str, retry_count: int = 0) -> Dict[str, Any]:
        """
        Make OpenAI API call with function calling.

        Args:
            text: Text to process
            model: OpenAI model to use
            retry_count: Current retry attempt

        Returns:
            Dict: Extracted profile data
        """
        # Get the schema for function calling
        schema = CandidateProfile.model_json_schema()

        # Prepare function definition
        function_def = {
            "name": "extract_profile",
            "description": "Extract candidate profile information from CV text",
            "parameters": schema
        }

        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": "You are an expert CV analyzer. Extract candidate information from the provided CV text. "
                           "Return only the requested data in the exact format specified. "
                           "If information is missing, use null/empty values. "
                           "For dates, use YYYY-MM-DD format. "
                           "Be accurate and thorough."
            },
            {
                "role": "user",
                "content": f"Please analyze this CV and extract the candidate profile:\n\n{text}"
            }
        ]

        # Add retry context if this is a retry
        if retry_count > 0:
            messages.append({
                "role": "user",
                "content": "The previous extraction had validation errors. Please ensure all data types and formats are correct."
            })

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                functions=[function_def],
                function_call={"name": "extract_profile"},
                temperature=0.0
            )

            # Extract function call result
            function_call = response.choices[0].message.function_call
            if not function_call or function_call.name != "extract_profile":
                raise Exception("OpenAI did not call the expected function")

            # Parse the JSON response
            result = json.loads(function_call.arguments)
            logger.info("Successfully received OpenAI response")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response as JSON: {e}")
            if retry_count < 1:
                logger.info("Retrying OpenAI call due to JSON parse error")
                return self._make_openai_call(text, model, retry_count + 1)
            raise Exception("Failed to get valid JSON response from OpenAI after retry")

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise Exception(f"OpenAI processing failed: {str(e)}")

    def _normalize_data(self, data: Dict[str, Any]) -> None:
        """
        Normalize data to handle None values for list fields.

        Args:
            data: Raw profile data to normalize in-place
        """
        list_fields = ['languages', 'skills', 'experience', 'education', 'certifications']
        for field in list_fields:
            if field in data and data[field] is None:
                data[field] = []
                logger.info(f"Normalized None value to empty list for field: {field}")

    def _validate_and_create_profile(self, data: Dict[str, Any]) -> CandidateProfile:
        """
        Validate data against Pydantic schema and create profile.

        Args:
            data: Raw profile data from OpenAI

        Returns:
            CandidateProfile: Validated profile instance
        """
        try:
            # Handle None values for list fields
            self._normalize_data(data)

            adapter = TypeAdapter(CandidateProfile)
            profile = adapter.validate_python(data)
            logger.info("Profile validation successful")
            return profile
        except ValidationError as e:
            logger.error(f"Profile validation failed: {e}")
            raise e

    def _save_profile(self, profile: CandidateProfile) -> None:
        """
        Save profile to JSON file.

        Args:
            profile: CandidateProfile instance to save
        """
        try:
            with open("profile.json", "w", encoding="utf-8") as f:
                json.dump(profile.model_dump(), f, indent=2, default=str, ensure_ascii=False)
            logger.info("Profile saved to profile.json")
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")
            # Don't raise - saving is not critical for the main functionality


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Analyze CV and extract candidate profile")
    parser.add_argument("pdf_path", help="Path to PDF résumé file")
    parser.add_argument("--out", default="./profile.json", help="Output JSON file path")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")

    args = parser.parse_args()

    try:
        # Initialize analyzer
        analyzer = CVAnalyzer(api_key=args.api_key)

        # Analyze CV
        profile = analyzer.analyze(args.pdf_path, model=args.model)

        # Save to specified output file
        if args.out != "./profile.json":
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(profile.model_dump(), f, indent=2, default=str, ensure_ascii=False)
            print(f"Profile saved to {args.out}")
        else:
            print("Profile saved to profile.json")

        # Print summary
        print(f"\nCandidate: {profile.full_name}")
        print(f"Email: {profile.email}")
        print(f"Experience entries: {len(profile.experience)}")
        print(f"Education entries: {len(profile.education)}")
        print(f"Skills: {len(profile.skills)}")
        print(profile)

    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()