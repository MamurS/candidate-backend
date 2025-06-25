import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode, urlparse, parse_qs
from datetime import datetime, timedelta
import json
from enum import Enum
from typing import List, Dict, Optional, Tuple
import time
import re
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import asyncio
import aiohttp
from aiohttp import ClientSession, ClientTimeout

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define Enums to match your database schema
class SalaryRange(Enum):
    UNDER_20K = "UNDER_20K"
    BETWEEN_20K_30K = "20K-30K"
    BETWEEN_30K_40K = "30K-40K"
    BETWEEN_40K_50K = "40K-50K"
    BETWEEN_50K_70K = "50K-70K"
    BETWEEN_70K_100K = "70K-100K"
    OVER_100K = "OVER_100K"
    NEGOTIABLE = "NEGOTIABLE"


class JobType(Enum):
    FULL_TIME = "FULL_TIME"
    PART_TIME = "PART_TIME"
    CONTRACT = "CONTRACT"
    TEMPORARY = "TEMPORARY"
    APPRENTICESHIP = "APPRENTICESHIP"
    VOLUNTEER = "VOLUNTEER"


class RemotePreference(Enum):
    REMOTE_ONLY = "REMOTE_ONLY"
    HYBRID = "HYBRID"
    ONSITE_ONLY = "ONSITE_ONLY"
    FLEXIBLE = "FLEXIBLE"


class ExperienceLevel(Enum):
    ENTRY_LEVEL = "ENTRY_LEVEL"
    MID_LEVEL = "MID_LEVEL"
    SENIOR_LEVEL = "SENIOR_LEVEL"
    EXECUTIVE = "EXECUTIVE"


class RelocationPreference(Enum):
    YES = "YES"
    NO = "NO"
    MAYBE = "MAYBE"


class NoticePeriod(Enum):
    IMMEDIATE = "IMMEDIATE"
    ONE_WEEK = "ONE_WEEK"
    TWO_WEEKS = "TWO_WEEKS"
    ONE_MONTH = "ONE_MONTH"
    TWO_MONTHS = "TWO_MONTHS"
    THREE_MONTHS = "THREE_MONTHS"
    MORE_THAN_THREE = "MORE_THAN_THREE"


@dataclass
class JobVacancy:
    """Data class to hold job vacancy information"""
    job_title: str
    company_name: str
    location: str
    salary_range: str
    post_time: str
    job_description: str
    url: str
    job_id: Optional[str] = None
    remote_type: Optional[str] = None
    contract_type: Optional[str] = None
    hours_type: Optional[str] = None


class FindAJobGrabber:
    """
	A class to grab vacancies from the Find a Job website (findajob.dwp.gov.uk)
	"""

    def __init__(self):
        self.base_url = "https://findajob.dwp.gov.uk"
        self.search_url = f"{self.base_url}/search"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-GB,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _map_salary_to_params(self, salary_range: SalaryRange) -> Dict[str, int]:
        """Map SalaryRange enum to Find a Job salary parameters"""
        salary_mapping = {
            SalaryRange.UNDER_20K: {"st": 20000, "sf": 0},
            SalaryRange.BETWEEN_20K_30K: {"st": 30000, "sf": 20000},
            SalaryRange.BETWEEN_30K_40K: {"st": 40000, "sf": 30000},
            SalaryRange.BETWEEN_40K_50K: {"st": 50000, "sf": 40000},
            SalaryRange.BETWEEN_50K_70K: {"st": 70000, "sf": 50000},
            SalaryRange.BETWEEN_70K_100K: {"st": 100000, "sf": 70000},
            SalaryRange.OVER_100K: {"st": 999999, "sf": 100000},
            SalaryRange.NEGOTIABLE: {"st": None, "sf": None}
        }
        return salary_mapping.get(salary_range, {"st": None, "sf": None})

    def _map_job_type(self, job_type: JobType) -> str:
        """Map JobType enum to Find a Job contract type"""
        job_type_mapping = {
            JobType.FULL_TIME: "full_time",
            JobType.PART_TIME: "part_time",
            JobType.CONTRACT: "contract",
            JobType.TEMPORARY: "temporary",
            JobType.APPRENTICESHIP: "apprenticeship",
            JobType.VOLUNTEER: "volunteer"
        }
        return job_type_mapping.get(job_type, "")

    def _map_remote_preference(self, remote_pref: RemotePreference) -> str:
        """Map RemotePreference enum to Find a Job remote parameter"""
        remote_mapping = {
            RemotePreference.REMOTE_ONLY: "fully_remote",
            RemotePreference.HYBRID: "hybrid",
            RemotePreference.ONSITE_ONLY: "on_site_and_in_the_field",
            RemotePreference.FLEXIBLE: ""
        }
        return remote_mapping.get(remote_pref, "")

    def _build_search_params(self, job_preferences: Dict, days_posted: Optional[int] = None) -> Dict:
        """Build search parameters from job preferences - using simple search"""
        params = {}

        # Simple search - just use the search query
        if job_preferences.get('desired_role'):
            params['q'] = job_preferences['desired_role']

        # Optionally add location (default to UK if not specified)
        params['w'] = 'UK'  # Simple location parameter

        # Add freshness filter if specified
        if days_posted is not None:
            if days_posted <= 1:
                params['f'] = '7'  # Today
            elif days_posted <= 3:
                params['f'] = '3'  # Last 3 days
            elif days_posted <= 7:
                params['f'] = '1'  # Last 7 days
            elif days_posted <= 14:
                params['f'] = '2'  # Last 14 days
        # If more than 14 days, don't add filter (show all)

        return params

    def _parse_job_listing_from_search(self, job_element) -> Optional[Dict]:
        """Parse individual job listing from search results page"""
        try:
            job_data = {}

            # Extract job title and URL from the h3 > a element
            title_link = job_element.find('h3', class_='govuk-heading-s').find('a', class_='govuk-link')
            if title_link:
                job_data['job_title'] = title_link.text.strip()
                href = title_link.get('href')
                if href:
                    job_data['url'] = f"{self.base_url}{href}" if not href.startswith('http') else href
                    # Extract job ID from URL
                    job_id_match = re.search(r'/details/(\d+)', href)
                    if job_id_match:
                        job_data['job_id'] = job_id_match.group(1)

            # Parse job details from the ul.search-result-details
            details_list = job_element.find('ul', class_='search-result-details')
            if details_list:
                list_items = details_list.find_all('li')
                for li in list_items:
                    # Skip tags (they have govuk-tag class)
                    if 'govuk-tag' in li.get('class', []):
                        # Parse contract/hours type from tags
                        tag_text = li.text.strip()
                        if tag_text.lower() in ['permanent', 'temporary', 'contract', 'apprenticeship']:
                            job_data['contract_type'] = tag_text
                        elif tag_text.lower() in ['full time', 'part time']:
                            job_data['hours_type'] = tag_text
                        elif tag_text.lower() in ['on-site only', 'hybrid remote', 'fully remote']:
                            job_data['remote_type'] = tag_text
                        continue

                    # Get text content
                    text = li.get_text(strip=True)

                    # First item is usually the date
                    if not job_data.get('post_time') and re.match(r'\d+\s+\w+\s+\d{4}', text):
                        job_data['post_time'] = text

                    # Parse company and location (contains <strong> tag for company)
                    elif li.find('strong') and ' - ' in text:
                        strong_text = li.find('strong').text.strip()
                        job_data['company_name'] = strong_text
                        # Extract location after the dash
                        location_span = li.find('span')
                        if location_span:
                            job_data['location'] = location_span.text.strip()

                    # Parse salary (contains £ symbol)
                    elif '£' in text or 'per' in text.lower():
                        job_data['salary_range'] = text

            # Parse short description
            desc_p = job_element.find('p', class_='search-result-description')
            if desc_p:
                job_data['short_description'] = desc_p.text.strip()

            # Set defaults for missing fields
            if 'salary_range' not in job_data:
                job_data['salary_range'] = "Not specified"
            if 'company_name' not in job_data:
                job_data['company_name'] = "Not specified"
            if 'location' not in job_data:
                job_data['location'] = "Not specified"

            return job_data

        except Exception as e:
            logger.error(f"Error parsing job listing: {e}")
            return None

    def _parse_post_date(self, date_text: str) -> str:
        """Parse the post date from text"""
        try:
            # If it's already in a standard format, return as is
            if re.match(r'\d+\s+\w+\s+\d{4}', date_text):
                return date_text

            # Handle relative dates
            if "today" in date_text.lower():
                return datetime.now().strftime("%d %B %Y")
            elif "yesterday" in date_text.lower():
                return (datetime.now() - timedelta(days=1)).strftime("%d %B %Y")
            elif "days ago" in date_text:
                days = int(re.search(r'(\d+)\s*days?\s*ago', date_text).group(1))
                return (datetime.now() - timedelta(days=days)).strftime("%d %B %Y")
            else:
                return date_text
        except:
            return date_text

    async def _fetch_job_description_async(self, session: ClientSession, job_data: Dict) -> Dict:
        """Asynchronously fetch full job description for a single job"""
        try:
            job_url = job_data['url']
            async with session.get(job_url) as response:
                response.raise_for_status()
                html = await response.text()

                soup = BeautifulSoup(html, 'html.parser')

                # Find job description section
                desc_section = soup.find('section', {'id': 'job-description'})
                if desc_section:
                    for script in desc_section(["script", "style"]):
                        script.decompose()
                    full_description = desc_section.get_text(separator='\n', strip=True)
                else:
                    # Alternative: look for description in main content
                    main_content = soup.find('main')
                    if main_content:
                        for heading in main_content.find_all(['h2', 'h3']):
                            if 'description' in heading.text.lower() or 'summary' in heading.text.lower():
                                desc_container = heading.find_next_sibling()
                                if desc_container:
                                    full_description = desc_container.get_text(separator='\n', strip=True)
                                    break
                        else:
                            full_description = "Full description not available"
                    else:
                        full_description = "Full description not available"

                # Create JobVacancy object
                vacancy = JobVacancy(
                    job_title=job_data.get('job_title', 'Not specified'),
                    company_name=job_data.get('company_name', 'Not specified'),
                    location=job_data.get('location', 'Not specified'),
                    salary_range=job_data.get('salary_range', 'Not specified'),
                    post_time=job_data.get('post_time', 'Not specified'),
                    job_description=full_description,
                    url=job_data['url'],
                    job_id=job_data.get('job_id'),
                    contract_type=job_data.get('contract_type'),
                    hours_type=job_data.get('hours_type'),
                    remote_type=job_data.get('remote_type')
                )

                return vacancy.__dict__

        except Exception as e:
            logger.error(f"Error fetching job description for {job_data.get('job_title', 'Unknown')}: {e}")
            # Return job with error description
            vacancy = JobVacancy(
                job_title=job_data.get('job_title', 'Not specified'),
                company_name=job_data.get('company_name', 'Not specified'),
                location=job_data.get('location', 'Not specified'),
                salary_range=job_data.get('salary_range', 'Not specified'),
                post_time=job_data.get('post_time', 'Not specified'),
                job_description=f"Error fetching description: {str(e)}",
                url=job_data.get('url', ''),
                job_id=job_data.get('job_id'),
                contract_type=job_data.get('contract_type'),
                hours_type=job_data.get('hours_type'),
                remote_type=job_data.get('remote_type')
            )
            return vacancy.__dict__

    async def _fetch_all_job_descriptions_async(self, job_listings: List[Dict], max_concurrent: int = 10) -> List[Dict]:
        """Fetch all job descriptions asynchronously with rate limiting"""
        vacancies = []

        # Configure timeout
        timeout = ClientTimeout(total=30, connect=10, sock_read=10)

        # Configure connection pool
        connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)

        async with ClientSession(headers=self.headers, timeout=timeout, connector=connector) as session:
            # Process jobs in batches to control concurrency
            for i in range(0, len(job_listings), max_concurrent):
                batch = job_listings[i:i + max_concurrent]

                # Create tasks for this batch
                tasks = [self._fetch_job_description_async(session, job_data) for job_data in batch]

                # Wait for all tasks in this batch to complete
                results = await asyncio.gather(*tasks)
                vacancies.extend(results)

                # Log progress
                logger.info(f"Processed {min(i + max_concurrent, len(job_listings))}/{len(job_listings)} jobs")

                # Small delay between batches to be respectful
                if i + max_concurrent < len(job_listings):
                    await asyncio.sleep(0.5)

        return vacancies

    def grab_vacancies_async(self, job_preferences: Dict, max_results: int = 200, days_posted: Optional[int] = None,
                             max_concurrent: int = 10) -> List[Dict]:
        """
		Main method to grab vacancies asynchronously - much faster for large numbers of jobs

		This method now properly handles running in an existing event loop (like FastAPI)
		"""
        # First, get all job listings from search pages
        all_job_listings = []
        page = 1
        per_page = 50  # Maximum results per page

        try:
            while len(all_job_listings) < max_results:
                # Build search parameters
                params = self._build_search_params(job_preferences, days_posted)
                params['pp'] = per_page  # Results per page
                params['p'] = page  # Page number

                # Build full URL for logging
                full_url = f"{self.search_url}?{urlencode(params)}"
                logger.info(f"Fetching page {page}: {full_url}")
                if page == 1:
                    print(f"\nRequest URL: {full_url}\n")

                # Make search request
                response = self.session.get(self.search_url, params=params)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')

                # Find all job listings on this page
                job_listings = soup.find_all('div', class_='search-result')

                if not job_listings:
                    logger.info(f"No more job listings found on page {page}")
                    break

                logger.info(f"Found {len(job_listings)} job listings on page {page}")

                # Parse each job listing to get basic info
                for job_div in job_listings:
                    if len(all_job_listings) >= max_results:
                        break

                    job_data = self._parse_job_listing_from_search(job_div)
                    if job_data and job_data.get('url'):
                        all_job_listings.append(job_data)

                # Check if we have enough results or if there are no more pages
                if len(job_listings) < per_page:
                    logger.info("Reached last page of results")
                    break

                page += 1
                time.sleep(0.5)  # Be respectful between page requests

            logger.info(f"Collected {len(all_job_listings)} job listings from search results")

            # Now fetch all job descriptions asynchronously
            if all_job_listings:
                print(f"\nFetching detailed descriptions for {len(all_job_listings)} jobs asynchronously...")

                # Check if we're already in an event loop (e.g., FastAPI)
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an existing loop, we need to handle this differently
                    # We'll use threading to run the async function
                    import concurrent.futures
                    import threading

                    def run_async_in_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(
                                self._fetch_all_job_descriptions_async(all_job_listings, max_concurrent)
                            )
                        finally:
                            new_loop.close()

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_async_in_thread)
                        vacancies = future.result()
                except RuntimeError:
                    # No event loop is running, create a new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        vacancies = loop.run_until_complete(
                            self._fetch_all_job_descriptions_async(all_job_listings, max_concurrent)
                        )
                    finally:
                        loop.close()

                logger.info(f"Successfully grabbed {len(vacancies)} job vacancies with full descriptions")
                print(f"\nCompleted! Retrieved {len(vacancies)} job vacancies.")
                return vacancies
            else:
                logger.warning("No job listings found")
                return []

        except requests.RequestException as e:
            logger.error(f"Error making request: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def grab_vacancies(self, job_preferences: Dict, max_results: int = 50, days_posted: Optional[int] = None) -> List[
        Dict]:
        """
		Synchronous method to grab vacancies (slower but simpler)
		Good for small numbers of jobs or when async is not needed
		"""
        all_vacancies = []
        page = 1
        per_page = 50

        try:
            while len(all_vacancies) < max_results:
                # Build search parameters
                params = self._build_search_params(job_preferences, days_posted)
                params['pp'] = per_page
                params['p'] = page

                # Make request
                response = self.session.get(self.search_url, params=params)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')
                job_listings = soup.find_all('div', class_='search-result')

                if not job_listings:
                    break

                # Process each job
                for job_div in job_listings:
                    if len(all_vacancies) >= max_results:
                        break

                    job_data = self._parse_job_listing_from_search(job_div)
                    if job_data and job_data.get('url'):
                        # Fetch full description synchronously
                        try:
                            desc_response = self.session.get(job_data['url'])
                            desc_response.raise_for_status()
                            desc_soup = BeautifulSoup(desc_response.content, 'html.parser')

                            # Extract description
                            desc_section = desc_soup.find('section', {'id': 'job-description'})
                            if desc_section:
                                full_description = desc_section.get_text(separator='\n', strip=True)
                            else:
                                full_description = "Description not available"

                            # Create vacancy
                            vacancy = JobVacancy(
                                job_title=job_data.get('job_title', 'Not specified'),
                                company_name=job_data.get('company_name', 'Not specified'),
                                location=job_data.get('location', 'Not specified'),
                                salary_range=job_data.get('salary_range', 'Not specified'),
                                post_time=job_data.get('post_time', 'Not specified'),
                                job_description=full_description,
                                url=job_data['url'],
                                job_id=job_data.get('job_id'),
                                contract_type=job_data.get('contract_type'),
                                hours_type=job_data.get('hours_type'),
                                remote_type=job_data.get('remote_type')
                            )

                            all_vacancies.append(vacancy.__dict__)
                            time.sleep(0.5)  # Be respectful

                        except Exception as e:
                            logger.error(f"Error fetching job description: {e}")

                if len(job_listings) < per_page:
                    break

                page += 1

            return all_vacancies

        except Exception as e:
            logger.error(f"Error in synchronous grab: {e}")
            return []


# Example usage
if __name__ == "__main__":
    # Create grabber instance
    grabber = FindAJobGrabber()

    # Example job preferences for HR Recruiter
    job_preferences = {
        "expected_salary": "30K-40K",  # Use the exact enum value
        "job_type": "FULL_TIME",
        "desired_role": "Python Developer",
        "industries": "Human Resources,Recruitment",
        "willing_to_relocate": "NO",
        "remote_work_preference": "HYBRID",
        "years_of_experience": "MID_LEVEL",
        "notice_period": "ONE_MONTH"
    }

    # Method 1: Synchronous (slower, good for small numbers)
    print("Example 1: Synchronous method (5 jobs)")
    vacancies = grabber.grab_vacancies(job_preferences, max_results=5, days_posted=7)
    print(f"Found {len(vacancies)} jobs synchronously")

    # Method 2: Asynchronous (much faster for large numbers)
    # print("\nExample 2: Asynchronous method (20 jobs)")
    # vacancies_async = grabber.grab_vacancies_async(
    # 	job_preferences,
    # 	max_results=20,  # Get 20 jobs
    # 	days_posted=7,  # Posted in last 7 days
    # 	max_concurrent=10  # 10 concurrent requests
    # )

    # Print first few results
    for i, vacancy in enumerate(vacancies[:3]):
        print("\n" + "=" * 50)
        print(f"Job {i + 1}:")
        print(f"Title: {vacancy.get('job_title', 'N/A')}")
        print(f"Company: {vacancy.get('company_name', 'N/A')}")
        print(f"Location: {vacancy.get('location', 'N/A')}")
        print(f"Salary: {vacancy.get('salary_range', 'N/A')}")
        print(f"Posted: {vacancy.get('post_time', 'N/A')}")

    print(f"\nTotal jobs retrieved: {len(vacancies)}")