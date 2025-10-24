#
# By Ian Drumm, The University of Salford, UK.
# Enhanced version with better search handling and time binning
#
import praw
import json
import re
import os
from dotenv import load_dotenv
from typing import List, Dict, Set, Literal, Optional
from datetime import datetime, timedelta
from tqdm import tqdm
import time

class RedditScraper:
    def __init__(self):
        load_dotenv()
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT")
        )
        self.seen_comment_ids: Set[str] = set()  # Track duplicates

    def _split_or_query(self, query: str) -> List[str]:
        """
        Automatically split OR queries into individual searches for better results
        
        "climate change OR global warming" -> ["climate change", "global warming"]
        """
        # Remove quotes if present
        cleaned = query.replace('"', '')
        
        # Split by OR (case insensitive)
        queries = re.split(r'\s+OR\s+', cleaned, flags=re.IGNORECASE)
        
        # Clean up whitespace
        queries = [q.strip() for q in queries if q.strip()]
        
        # If only one query, return as-is; if many ORs, split them up
        if len(queries) == 1:
            return [query]  # Keep original if no ORs
        else:
            #print(f"  Detected {len(queries)} OR terms - splitting into separate searches for better results")
            return queries

    def _determine_time_filter(self, start_date: datetime = None) -> str:
        """
        Automatically determine the best time_filter for Reddit search
        """
        if not start_date:
            return "all"
        
        days_ago = (datetime.utcnow() - start_date).days
        
        if days_ago <= 7:
            return "week"
        elif days_ago <= 30:
            return "month"
        elif days_ago <= 365:
            return "year"
        else:
            return "all"

    def scrape_subreddit(
        self,
        subreddit_name: str,
        query: str,
        limit: int = 50,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[Dict]:
        """
        Scrape a single subreddit - now with automatic query splitting
        """
        subreddit = self.reddit.subreddit(subreddit_name)
        results = []
        
        # Determine best time filter
        time_filter = self._determine_time_filter(start_date)
        
        # Split query if it contains OR operators
        sub_queries = self._split_or_query(query)
        
        # Adjust limit per sub-query if we split
        limit_per_query = limit if len(sub_queries) == 1 else max(50, limit // len(sub_queries))
        
        # Do not print search status here, as scrape_multiple now handles overall progress
        
        for sub_query in sub_queries:
            # We skip the specific sub-query print here to keep binning progress cleaner
            
            try:
                # Use 'new' sorting for better date-range coverage
                submissions = list(subreddit.search(
                    sub_query,
                    sort="new",  # Changed from relevance
                    limit=limit_per_query,
                    time_filter=time_filter
                ))
            except Exception as e:
                # print(f"  Error searching: {e}")
                continue
            
            # Filter submissions by date first
            filtered_submissions = []
            for submission in submissions:
                submission_created_dt = datetime.utcfromtimestamp(submission.created_utc)
                
                if start_date and submission_created_dt < start_date:
                    continue
                if end_date and submission_created_dt > end_date:
                    continue
                    
                filtered_submissions.append((submission, submission_created_dt))
            
            if not filtered_submissions:
                continue
            
            # Now process with accurate progress
            desc = f"Fetching r/{subreddit_name}" if len(sub_queries) == 1 else f"  {sub_query[:30]}"
            for submission, submission_created_dt in tqdm(filtered_submissions, 
                                                        desc=desc,
                                                        leave=False,
                                                        unit=" posts"):
                try:
                    submission.comments.replace_more(limit=0)
                    
                    for comment in submission.comments.list():
                        if not hasattr(comment, 'body'):
                            continue
                        
                        # Skip duplicates (important with multiple queries)
                        if comment.id in self.seen_comment_ids:
                            continue

                        comment_created_dt = datetime.utcfromtimestamp(comment.created_utc)

                        if start_date and comment_created_dt < start_date:
                            continue
                        if end_date and comment_created_dt > end_date:
                            continue

                        self.seen_comment_ids.add(comment.id)

                        results.append({
                            "post": submission.title + "\n" + submission.selftext,
                            "comment": comment.body,
                            "post_metadata": {
                                "subreddit": subreddit_name,
                                "title": submission.title,
                                "created_utc": submission_created_dt.isoformat(),
                                "search_query": sub_query,
                                "post_score": submission.score,
                                "post_url": f"https://reddit.com{submission.permalink}"
                            },
                            "comment_metadata": {
                                "id": comment.id,
                                "created_utc": comment_created_dt.isoformat(),
                                "score": comment.score
                            }
                        })
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.05)
                    
                except Exception as e:
                    # print(f"  Error processing submission: {e}")
                    continue
        
        return results

    def _generate_time_bins(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        bin_type: Literal['day', 'week', 'month']
    ) -> List[Dict[str, Optional[datetime]]]:
        """
        Generates a list of {'start_date': ..., 'end_date': ...} dictionaries 
        for iterative scraping.
        """
        bins = []
        current_start = start_date
        
        while current_start < end_date:
            next_start = current_start # Initialize next_start
            
            if bin_type == 'day':
                # The end of the bin is the start of the next day
                next_start = current_start + timedelta(days=1)
            elif bin_type == 'week':
                # The end of the bin is the start of the next week (7 days)
                next_start = current_start + timedelta(weeks=1)
            elif bin_type == 'month':
                # Calculate the start of the next month
                year = current_start.year
                month = current_start.month % 12 + 1
                if month == 1:
                    year += 1
                try:
                    next_start = current_start.replace(year=year, month=month, day=1)
                except ValueError:
                    # Handle cases where the start day is invalid for the next month (e.g., Feb 30th)
                    # We'll default to the first of the month after
                    if month == 12:
                        year += 1
                        month = 1
                    else:
                        month += 1
                    next_start = datetime(year, month, 1, tzinfo=current_start.tzinfo)

            # Ensure the bin end doesn't exceed the overall end_date
            bin_end = min(next_start, end_date)
            
            bins.append({'start_date': current_start, 'end_date': bin_end})
            current_start = next_start
            
        return bins

    def scrape_multiple(
        self,
        subreddits: List[str],
        query: str,
        limit: int = 50,
        start_date: datetime = None,
        end_date: datetime = None,
        bin_by: Optional[Literal['day', 'week', 'month']] = None
    ) -> List[Dict]:
        """
        Scrape multiple subreddits with automatic query optimization and time binning.

        :param subreddits: List of subreddit names.
        :param query: The search query.
        :param limit: Max number of submissions to retrieve PER sub-query PER bin.
        :param start_date: Start of the overall date range.
        :param end_date: End of the overall date range.
        :param bin_by: 'day', 'week', or 'month' to split the scrape into bins.
        :return: List of collected comments.
        """
        all_data = []
        
        # --- Time Binning Logic ---
        if bin_by and start_date and end_date and start_date < end_date:
            time_bins = self._generate_time_bins(start_date, end_date, bin_by)
            binning_active = True
            print(f"Scraping split into {len(time_bins)} {bin_by} bins.")
        else:
            time_bins = [{'start_date': start_date, 'end_date': end_date}]
            binning_active = False
            
        # Format date range for display
        date_range_str = "No date filter"
        if start_date or end_date:
            start_str = start_date.strftime("%Y-%m-%d") if start_date else "Beginning"
            end_str = end_date.strftime("%Y-%m-%d") if end_date else "Present"
            date_range_str = f"{start_str} to {end_str}"
        
        print(f"Starting scrape with overall date range: {date_range_str}")
        
        # Check if query has many OR terms
        or_count = query.upper().count(' OR ')
        if or_count > 3:
            print(f"Note: Query has {or_count + 1} OR terms - will split into separate searches")
            print(f"This improves results but takes longer. Limit of {limit} applies per OR term PER bin.\n")

        # Iterate through subreddits and time bins
        for sub in subreddits:
            print(f"\n--- Processing r/{sub} ---")
            
            # Use tqdm to track progress through the time bins
            bin_iterator = tqdm(time_bins, desc=f"r/{sub} Bins", unit="bin")
            
            for bin_num, time_bin in enumerate(bin_iterator):
                bin_start = time_bin['start_date']
                bin_end = time_bin['end_date']
                
                # Update progress bar description
                bin_desc = f"{bin_start.strftime('%Y-%m-%d')} to {bin_end.strftime('%Y-%m-%d')}"
                bin_iterator.set_description(f"r/{sub} Bin ({bin_desc})")

                # The core scraping call for this specific subreddit and time bin
                data = self.scrape_subreddit(sub, query, limit, bin_start, bin_end)
                all_data.extend(data)
                
                # Print status per bin (optional)
                if binning_active:
                    bin_iterator.set_postfix_str(f"Found {len(data)} comments in this bin")
                
                # Delay between bins (helps with rate limits, especially for daily bins)
                time.sleep(0.5)

            print(f"  Total unique comments collected from r/{sub}: {sum(1 for d in all_data if d['post_metadata']['subreddit'] == sub)}")
            
            # Delay between subreddits
            time.sleep(1)
        
        print(f"\nTotal unique comments collected across all subreddits: {len(all_data)}")
        return all_data

def format_search_query(query: str) -> str:
    """
    Automatically format search queries to use proper Reddit search syntax.
    Converts terms like: AI regulation OR AI governance OR AI policy
    To: "AI regulation" OR "AI governance" OR "AI policy"
    """
    # If the query already has quotes around phrases, return as-is
    if '"' in query:
        return query
    
    # Split by OR/AND operators while preserving them
    parts = re.split(r'\s+(OR|AND|NOT)\s+', query, flags=re.IGNORECASE)
    
    formatted_parts = []
    for part in parts:
        part = part.strip()
        if part.upper() in ['OR', 'AND', 'NOT']:
            # Keep operators as-is
            formatted_parts.append(part.upper())
        elif part:
            # Wrap phrases in quotes if they contain spaces or are multi-word
            if ' ' in part or len(part.split()) > 1:
                formatted_parts.append(f'"{part}"')
            else:
                # Single words don't need quotes
                formatted_parts.append(part)
    
    formatted_query = ' '.join(formatted_parts)
    return formatted_query