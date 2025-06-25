import json
import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np
from collections import Counter
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings('ignore')

# Try to import NLTK, but make it optional
try:
	import nltk
	from nltk.corpus import stopwords
	from nltk.tokenize import word_tokenize

	# Download required NLTK data
	for resource in ['punkt', 'punkt_tab', 'stopwords']:
		try:
			nltk.data.find(f'tokenizers/{resource}')
		except LookupError:
			try:
				nltk.download(resource, quiet=True)
			except:
				pass

	NLTK_AVAILABLE = True
except ImportError:
	NLTK_AVAILABLE = False
	print("NLTK not available, using simple tokenization")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MatchScore:
	"""Data class to hold matching scores and details"""
	overall_score: float
	semantic_similarity: float
	keyword_match_score: float
	tfidf_similarity: float
	matched_keywords: List[str]
	common_skills: List[str]
	match_details: Dict
	match_explanation: str


class GeneralJobMatcher:
	"""
	A general job matcher using text embeddings and multiple similarity metrics
	"""

	def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
		"""
		Initialize the matcher with a sentence transformer model

		Args:
			model_name: Name of the sentence transformer model
					   Options: 'all-MiniLM-L6-v2' (fastest, good quality)
							   'all-mpnet-base-v2' (best quality, slower)
							   'paraphrase-MiniLM-L3-v2' (very fast, lower quality)
		"""
		logger.info(f"Loading sentence transformer model: {model_name}")
		self.model = SentenceTransformer(model_name)

		# Initialize TF-IDF vectorizer
		self.tfidf = TfidfVectorizer(
			max_features=500,
			stop_words='english',
			ngram_range=(1, 2),
			min_df=1
		)

		# Get English stopwords with fallback
		try:
			self.stop_words = set(stopwords.words('english'))
		except:
			# Fallback to basic stopwords if NLTK fails
			self.stop_words = {
				'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was', 'were',
				'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
				'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
				'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
				'who', 'when', 'where', 'why', 'how', 'not', 'no', 'nor', 'but', 'or',
				'yet', 'so', 'for', 'and', 'in', 'of', 'to', 'from', 'with', 'by'
			}

		# Cache for embeddings to avoid recomputation
		self.embedding_cache = {}

	def clean_text(self, text: str) -> str:
		"""Clean and preprocess text"""
		# Convert to lowercase
		text = text.lower()

		# Remove special characters but keep spaces
		text = re.sub(r'[^a-zA-Z0-9\s+#]', ' ', text)

		# Remove extra whitespace
		text = ' '.join(text.split())

		return text

	def extract_keywords(self, text: str, top_n: int = 20) -> List[str]:
		"""Extract important keywords from text"""
		# Clean text
		cleaned = self.clean_text(text)

		# Tokenization with fallback
		if NLTK_AVAILABLE:
			try:
				tokens = word_tokenize(cleaned)
			except:
				tokens = cleaned.split()
		else:
			tokens = cleaned.split()

		# Remove stopwords and short words
		keywords = [
			token for token in tokens
			if token not in self.stop_words and len(token) > 2
		]

		# Count frequencies
		word_freq = Counter(keywords)

		# Get top keywords
		return [word for word, _ in word_freq.most_common(top_n)]

	def create_candidate_profile_text(self, candidate: Dict) -> str:
		"""Convert candidate profile to searchable text"""
		text_parts = []

		# Add summary
		if candidate.get('summary'):
			text_parts.append(candidate['summary'])

		# Add skills
		if candidate.get('skills'):
			text_parts.append(f"Skills: {', '.join(candidate['skills'])}")

		# Add experience
		for exp in candidate.get('experience', []):
			text_parts.append(f"{exp.get('position', '')} at {exp.get('company', '')}")
			if exp.get('description'):
				text_parts.append(exp['description'])

		# Add education
		for edu in candidate.get('education', []):
			text_parts.append(
				f"{edu.get('degree', '')} in {edu.get('field_of_study', '')} from {edu.get('institution', '')}"
			)

		# Add languages
		if candidate.get('languages'):
			text_parts.append(f"Languages: {', '.join(candidate['languages'])}")

		return ' '.join(text_parts)

	def create_job_text(self, job: Dict) -> str:
		"""Convert job vacancy to searchable text"""
		text_parts = []

		# Add job title
		if job.get('job_title'):
			text_parts.append(job['job_title'])

		# Add company
		if job.get('company_name'):
			text_parts.append(f"Company: {job['company_name']}")

		# Add job description
		if job.get('job_description'):
			text_parts.append(job['job_description'])
		else:
			# Fallback to short description if available
			if job.get('short_description'):
				text_parts.append(job['short_description'])

		# Add location
		if job.get('location'):
			text_parts.append(f"Location: {job['location']}")

		# Add other details
		if job.get('contract_type'):
			text_parts.append(f"Contract: {job['contract_type']}")
		if job.get('remote_type'):
			text_parts.append(f"Remote: {job['remote_type']}")

		return ' '.join(text_parts)

	def get_embedding(self, text: str) -> np.ndarray:
		"""Get embedding for text with caching"""
		# Check cache first
		text_hash = hash(text)
		if text_hash in self.embedding_cache:
			return self.embedding_cache[text_hash]

		# Generate embedding
		embedding = self.model.encode(text, convert_to_numpy=True)

		# Cache it
		self.embedding_cache[text_hash] = embedding

		return embedding

	def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
		"""Calculate semantic similarity using sentence transformers"""
		# Get embeddings
		embedding1 = self.get_embedding(text1)
		embedding2 = self.get_embedding(text2)

		# Calculate cosine similarity
		similarity = cosine_similarity(
			embedding1.reshape(1, -1),
			embedding2.reshape(1, -1)
		)[0][0]

		return float(similarity)

	def calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
		"""Calculate TF-IDF based similarity"""
		try:
			# Clean texts
			text1_clean = self.clean_text(text1)
			text2_clean = self.clean_text(text2)

			# Fit and transform
			tfidf_matrix = self.tfidf.fit_transform([text1_clean, text2_clean])

			# Calculate cosine similarity
			similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

			return float(similarity)
		except:
			return 0.0

	def calculate_keyword_overlap(self, candidate_text: str, job_text: str) -> Tuple[float, List[str]]:
		"""Calculate keyword overlap between texts"""
		# Extract keywords
		candidate_keywords = set(self.extract_keywords(candidate_text, top_n=30))
		job_keywords = set(self.extract_keywords(job_text, top_n=30))

		# Find common keywords
		common_keywords = list(candidate_keywords.intersection(job_keywords))

		# Calculate overlap score
		if not job_keywords:
			return 0.0, common_keywords

		overlap_score = len(common_keywords) / len(job_keywords)

		return min(1.0, overlap_score), common_keywords

	def extract_skills_from_texts(self, candidate_text: str, job_text: str) -> List[str]:
		"""Extract common technical skills and terms"""
		# Common skill patterns
		skill_patterns = [
			r'\b(?:python|java|javascript|c\+\+|c#|ruby|go|rust|php|swift|kotlin)\b',
			r'\b(?:react|angular|vue|django|flask|spring|node\.?js|express)\b',
			r'\b(?:aws|azure|gcp|cloud|docker|kubernetes|devops|ci/cd)\b',
			r'\b(?:machine learning|deep learning|data science|ai|ml|nlp|cv)\b',
			r'\b(?:sql|nosql|mongodb|postgresql|mysql|redis|elasticsearch)\b',
			r'\b(?:git|agile|scrum|jira|confluence)\b',
			r'\b(?:leadership|management|communication|teamwork|problem solving)\b'
		]

		candidate_lower = candidate_text.lower()
		job_lower = job_text.lower()

		common_skills = []
		for pattern in skill_patterns:
			# Check if skill appears in both texts
			candidate_matches = set(re.findall(pattern, candidate_lower))
			job_matches = set(re.findall(pattern, job_lower))
			common = candidate_matches.intersection(job_matches)
			common_skills.extend(list(common))

		return list(set(common_skills))  # Remove duplicates

	def generate_match_explanation(self, match_score: MatchScore) -> str:
		"""Generate human-readable explanation of the match"""
		score_percent = match_score.overall_score * 100

		if score_percent >= 80:
			level = "Excellent"
			desc = "This is a very strong match with high alignment across all factors."
		elif score_percent >= 65:
			level = "Good"
			desc = "This is a solid match with good alignment in key areas."
		elif score_percent >= 50:
			level = "Moderate"
			desc = "This is a reasonable match with some aligned areas."
		else:
			level = "Low"
			desc = "This match has limited alignment with your profile."

		explanation = f"{level} Match ({score_percent:.1f}%): {desc}"

		# Add specific insights
		if match_score.common_skills:
			explanation += f"\n• Key skills match: {', '.join(match_score.common_skills[:5])}"

		if match_score.semantic_similarity > 0.7:
			explanation += f"\n• Strong semantic alignment between your experience and job requirements"

		if len(match_score.matched_keywords) > 10:
			explanation += f"\n• High keyword overlap ({len(match_score.matched_keywords)} common terms)"

		return explanation

	def match_single_job(self, candidate_profile: Dict, job_vacancy: Dict) -> MatchScore:
		"""Match a single job with candidate profile using multiple similarity metrics"""
		# Create text representations
		candidate_text = self.create_candidate_profile_text(candidate_profile)
		job_text = self.create_job_text(job_vacancy)

		# Calculate similarities
		semantic_sim = self.calculate_semantic_similarity(candidate_text, job_text)
		tfidf_sim = self.calculate_tfidf_similarity(candidate_text, job_text)
		keyword_score, matched_keywords = self.calculate_keyword_overlap(candidate_text, job_text)

		# Extract common skills
		common_skills = self.extract_skills_from_texts(candidate_text, job_text)

		# Calculate weighted overall score
		# Semantic similarity is most important for general matching
		overall_score = (
				semantic_sim * 0.5 +  # 50% weight on semantic similarity
				tfidf_sim * 0.3 +  # 30% weight on TF-IDF
				keyword_score * 0.2  # 20% weight on keyword overlap
		)

		# Create match score
		match_score = MatchScore(
			overall_score=overall_score,
			semantic_similarity=semantic_sim,
			keyword_match_score=keyword_score,
			tfidf_similarity=tfidf_sim,
			matched_keywords=matched_keywords,
			common_skills=common_skills,
			match_details={
				'job_id': job_vacancy.get('job_id'),
				'job_title': job_vacancy.get('job_title'),
				'company': job_vacancy.get('company_name'),
				'location': job_vacancy.get('location'),
				'salary': job_vacancy.get('salary_range'),
				'url': job_vacancy.get('url'),
				'post_time': job_vacancy.get('post_time')
			},
			match_explanation=""
		)

		# Generate explanation
		match_score.match_explanation = self.generate_match_explanation(match_score)

		return match_score

	def match_jobs_batch(self, candidate_profile: Dict, job_vacancies: List[Dict],
	                     min_score: float = 0.5, top_n: Optional[int] = None) -> List[Dict]:
		"""
		Match multiple jobs with candidate profile using batch processing for efficiency

		Args:
			candidate_profile: Candidate's profile dictionary
			job_vacancies: List of job vacancy dictionaries
			min_score: Minimum score threshold (default 0.5)
			top_n: Return only top N matches (default: all matches above threshold)

		Returns:
			List of matched jobs with scores, sorted by overall score
		"""
		# Create candidate text once
		candidate_text = self.create_candidate_profile_text(candidate_profile)
		candidate_embedding = self.get_embedding(candidate_text)

		# Process all jobs
		matches = []

		# Create job texts
		job_texts = [self.create_job_text(job) for job in job_vacancies]

		# Batch encode all job texts for efficiency
		logger.info(f"Encoding {len(job_texts)} job descriptions...")
		job_embeddings = self.model.encode(job_texts, convert_to_numpy=True, show_progress_bar=True)

		# Calculate similarities
		logger.info("Calculating similarities...")
		for idx, (job, job_text, job_embedding) in enumerate(zip(job_vacancies, job_texts, job_embeddings)):
			try:
				# Semantic similarity
				semantic_sim = float(cosine_similarity(
					candidate_embedding.reshape(1, -1),
					job_embedding.reshape(1, -1)
				)[0][0])

				# Only calculate other metrics for promising matches
				if semantic_sim >= min_score * 0.7:  # Quick filter
					# TF-IDF similarity
					tfidf_sim = self.calculate_tfidf_similarity(candidate_text, job_text)

					# Keyword overlap
					keyword_score, matched_keywords = self.calculate_keyword_overlap(candidate_text, job_text)

					# Common skills
					common_skills = self.extract_skills_from_texts(candidate_text, job_text)

					# Overall score
					overall_score = (
							semantic_sim * 0.5 +
							tfidf_sim * 0.3 +
							keyword_score * 0.2
					)

					if overall_score >= min_score:
						match_score = MatchScore(
							overall_score=overall_score,
							semantic_similarity=semantic_sim,
							keyword_match_score=keyword_score,
							tfidf_similarity=tfidf_sim,
							matched_keywords=matched_keywords,
							common_skills=common_skills,
							match_details={
								'job_id': job.get('job_id'),
								'job_title': job.get('job_title'),
								'company': job.get('company_name'),
								'location': job.get('location'),
								'salary': job.get('salary_range'),
								'url': job.get('url'),
								'post_time': job.get('post_time')
							},
							match_explanation=""
						)

						match_score.match_explanation = self.generate_match_explanation(match_score)

						matches.append({
							'job_details': match_score.match_details,
							'overall_score': round(match_score.overall_score, 3),
							'semantic_similarity': round(match_score.semantic_similarity, 3),
							'keyword_match_score': round(match_score.keyword_match_score, 3),
							'tfidf_similarity': round(match_score.tfidf_similarity, 3),
							'matched_keywords': match_score.matched_keywords[:10],  # Top 10
							'common_skills': match_score.common_skills,
							'match_percentage': round(match_score.overall_score * 100, 1),
							'match_explanation': match_score.match_explanation
						})

			except Exception as e:
				logger.error(f"Error matching job {idx}: {e}")
				continue

		# Sort by overall score
		matches.sort(key=lambda x: x['overall_score'], reverse=True)

		# Return top N if specified
		if top_n:
			matches = matches[:top_n]

		logger.info(f"Found {len(matches)} matches above threshold {min_score}")

		return matches

	def generate_match_report(self, matches: List[Dict], candidate_name: str = "Candidate") -> str:
		"""Generate a comprehensive matching report"""
		if not matches:
			return "No suitable job matches found above the threshold."

		report = f"Job Matching Report for {candidate_name}\n{'=' * 60}\n\n"
		report += f"Total matches found: {len(matches)}\n"
		report += f"Average match score: {np.mean([m['overall_score'] for m in matches]):.2%}\n\n"

		# Score distribution
		excellent = sum(1 for m in matches if m['overall_score'] >= 0.8)
		good = sum(1 for m in matches if 0.65 <= m['overall_score'] < 0.8)
		moderate = sum(1 for m in matches if 0.5 <= m['overall_score'] < 0.65)

		report += f"Match Distribution:\n"
		report += f"- Excellent matches (80%+): {excellent}\n"
		report += f"- Good matches (65-79%): {good}\n"
		report += f"- Moderate matches (50-64%): {moderate}\n\n"

		# Top matches
		report += "Top 10 Matches:\n" + "-" * 60 + "\n"
		for i, match in enumerate(matches[:10], 1):
			job = match['job_details']
			report += f"\n{i}. {job['job_title']} at {job['company']}\n"
			report += f"   Overall Match: {match['match_percentage']}% "
			report += f"(Semantic: {match['semantic_similarity']:.2f}, "
			report += f"Keywords: {match['keyword_match_score']:.2f}, "
			report += f"TF-IDF: {match['tfidf_similarity']:.2f})\n"
			report += f"   Location: {job['location']} | Posted: {job.get('post_time', 'N/A')}\n"
			if match['common_skills']:
				report += f"   Common Skills: {', '.join(match['common_skills'][:5])}\n"
			report += f"   {match['match_explanation'].split(':')[1].split('.')[0]}\n"

		# Keyword analysis
		all_keywords = []
		for match in matches[:20]:  # Top 20 matches
			all_keywords.extend(match['matched_keywords'])

		if all_keywords:
			keyword_counts = Counter(all_keywords)
			report += f"\n\nMost Common Keywords in Top Matches:\n"
			for keyword, count in keyword_counts.most_common(15):
				report += f"- {keyword}: appears in {count} jobs\n"

		return report


# Example usage
if __name__ == "__main__":
	# Initialize matcher with fastest model
	matcher = GeneralJobMatcher(model_name='all-MiniLM-L6-v2')

	# Example candidate profile
	candidate_profile = {
		"full_name": "John Doe",
		"email": "john@example.com",
		"summary": "Experienced software engineer with expertise in web development and cloud technologies",
		"skills": ["Python", "JavaScript", "AWS", "Docker", "React", "Node.js"],
		"experience": [
			{
				"company": "Tech Corp",
				"position": "Senior Software Engineer",
				"description": "Developed scalable web applications using React and Node.js"
			}
		],
		"education": [
			{
				"degree": "Bachelor's",
				"field_of_study": "Computer Science",
				"institution": "Tech University"
			}
		]
	}

	# Example jobs
	sample_jobs = [
		{
			"job_id": "1",
			"job_title": "Full Stack Developer",
			"company_name": "Web Solutions Inc",
			"location": "Remote",
			"job_description": "Looking for a full stack developer with React and Node.js experience. AWS knowledge is a plus.",
			"salary_range": "£60,000 - £80,000"
		},
		{
			"job_id": "2",
			"job_title": "HR Manager",
			"company_name": "Business Corp",
			"location": "London",
			"job_description": "Seeking an HR manager with recruitment and employee relations experience.",
			"salary_range": "£40,000 - £50,000"
		}
	]

	# Match jobs
	matches = matcher.match_jobs_batch(
		candidate_profile,
		sample_jobs,
		min_score=0.3,
		top_n=10
	)

	# Generate report
	report = matcher.generate_match_report(matches, candidate_profile.get('full_name', 'Candidate'))
	print(report)