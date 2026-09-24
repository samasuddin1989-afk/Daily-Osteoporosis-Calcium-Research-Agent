import os
import json
import random
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# 1. Configuration & Environment Setup
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL")

if not all([GEMINI_API_KEY, SENDER_EMAIL, GMAIL_APP_PASSWORD, RECEIVER_EMAIL]):
    raise ValueError("Missing required environment variables! Check GitHub Secrets.")

HISTORY_FILE = "sent_history.json"

# Initialize official Google GenAI Client
client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# 2. History & De-duplication Engine
# ---------------------------------------------------------------------------
def load_sent_history():
    """Loads previously sent article IDs/DOIs from history."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception as e:
            print(f"Warning: Could not parse history file: {e}")
            return set()
    return set()

def save_sent_history(history_set):
    """Saves updated sent IDs back to disk."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(list(history_set), f, indent=2)

# ---------------------------------------------------------------------------
# 3. Multi-Source Ingestion Engines (PubMed, Europe PMC, Crossref)
# ---------------------------------------------------------------------------

SEARCH_QUERIES = [
    "calcium supplementation osteoporosis management",
    "coral calcium OR eggshell calcium OR synthetic calcium",
    "bisphosphonates osteoporosis calcium therapy",
    "calcium carbonate vs citrate bone density",
    "calcium bioavailability bone mineral density",
    "postmenopausal osteoporosis calcium vitamin d",
    "osteoporosis fracture risk calcium intake"
]

def fetch_from_pubmed(sent_history, candidate_pool):
    """Fetches unique research papers from PubMed."""
    print("Searching PubMed...")
    for query in SEARCH_QUERIES:
        for offset in range(0, 150, 25): # Step through older articles if top results are sent
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": 25,
                "retstart": offset,
                "sort": "pub_date"
            }
            try:
                res = requests.get(search_url, params=params, timeout=12).json()
                id_list = res.get('esearchresult', {}).get('idlist', [])
                
                new_ids = [i for i in id_list if i not in sent_history and i not in candidate_pool]
                if not new_ids:
                    continue

                summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                summary_params = {
                    "db": "pubmed",
                    "id": ",".join(new_ids),
                    "retmode": "json"
                }
                details_res = requests.get(summary_url, params=summary_params, timeout=12).json()
                result_dict = details_res.get('result', {})

                for pmid in new_ids:
                    if pmid in result_dict:
                        meta = result_dict[pmid]
                        title = meta.get('title', 'No Title')
                        if title and len(title) > 10:
                            candidate_pool[pmid] = {
                                "id": pmid,
                                "source": "PubMed",
                                "title": title,
                                "date": meta.get('pubdate', 'N/A'),
                                "authors": ", ".join([a.get('name', '') for a in meta.get('authors', [])[:3]]),
                                "journal": meta.get('source', 'Scientific Journal'),
                                "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                            }
                if len(candidate_pool) >= 15:
                    return
            except Exception as e:
                print(f"PubMed fetch notice: {e}")

def fetch_from_europe_pmc(sent_history, candidate_pool):
    """Fetches unique research papers from Europe PMC."""
    print("Searching Europe PMC...")
    for query in SEARCH_QUERIES:
        search_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        params = {
            "query": f"{query} SORT_DATE:y",
            "format": "json",
            "pageSize": 25
        }
        try:
            res = requests.get(search_url, params=params, timeout=12).json()
            results = res.get('resultList', {}).get('result', [])
            for item in results:
                art_id = item.get('id') or item.get('doi')
                if not art_id or art_id in sent_history or art_id in candidate_pool:
                    continue
                
                title = item.get('title', '')
                if title and len(title) > 10:
                    candidate_pool[art_id] = {
                        "id": art_id,
                        "source": "Europe PMC",
                        "title": title,
                        "date": item.get('firstPublicationDate', 'N/A'),
                        "authors": item.get('authorString', 'N/A')[:60],
                        "journal": item.get('journalTitle', 'Europe PMC Journal'),
                        "link": f"https://europepmc.org/article/MED/{art_id}" if art_id.isdigit() else f"https://doi.org/{art_id}"
                    }
            if len(candidate_pool) >= 20:
                return
        except Exception as e:
            print(f"Europe PMC fetch notice: {e}")

def fetch_from_crossref(sent_history, candidate_pool):
    """Fetches unique journal articles from Crossref REST API."""
    print("Searching Crossref API...")
    headers = {"User-Agent": "MedicalDigestAgent/1.0 (mailto:admin@example.com)"}
    for query in SEARCH_QUERIES:
        search_url = "https://api.crossref.org/works"
        params = {
            "query": query,
            "filter": "type:journal-article",
            "sort": "published",
            "order": "desc",
            "rows": 25
        }
        try:
            res = requests.get(search_url, params=params, headers=headers, timeout=12).json()
            items = res.get('message', {}).get('items', [])
            for item in items:
                doi = item.get('DOI')
                if not doi or doi in sent_history or doi in candidate_pool:
                    continue
                
                titles = item.get('title', [])
                title = titles[0] if titles else ''
                if title and len(title) > 10:
                    pub_date = "N/A"
                    if 'published' in item and 'date-parts' in item['published']:
                        parts = item['published']['date-parts'][0]
                        pub_date = "-".join(map(str, parts))
                    
                    candidate_pool[doi] = {
                        "id": doi,
                        "source": "Crossref",
                        "title": title,
                        "date": pub_date,
                        "authors": "Various Authors",
                        "journal": item.get('container-title', ['Scientific Journal'])[0] if item.get('container-title') else 'Scientific Journal',
                        "link": f"https://doi.org/{doi}"
                    }
            if len(candidate_pool) >= 25:
                return
        except Exception as e:
            print(f"Crossref fetch notice: {e}")

# ---------------------------------------------------------------------------
# 4. Collection Master Pipeline (Ensures exactly 10 articles)
# ---------------------------------------------------------------------------
def collect_exact_10_articles(sent_history):
    candidate_pool = {}
    
    # Ingest from all 3 sources
    fetch_from_pubmed(sent_history, candidate_pool)
    fetch_from_europe_pmc(sent_history, candidate_pool)
    fetch_from_crossref(sent_history, candidate_pool)
    
    selected_articles = list(candidate_pool.values())
    
    # Shuffle slightly so sources mix well, then slice exact 10
    random.shuffle(selected_articles)
    final_10 = selected_articles[:10]
    
    if len(final_10) < 10:
        raise ValueError(f"Could not gather 10 articles. Only found {len(final_10)}. Pipeline halted.")
        
    return final_10

# ---------------------------------------------------------------------------
# 5. Digest Generation via Gemini 3.6 Flash
# ---------------------------------------------------------------------------
def generate_digest_html(articles):
    """Generates clean HTML digest using Google GenAI SDK."""
    articles_payload = json.dumps(articles, indent=2)

    prompt = f"""
    You are a clinical research AI assistant specializing in endocrinology, rheumatology, and bone health.
    
    Review these 10 articles retrieved from multiple scientific databases (PubMed, Europe PMC, Crossref):
    {articles_payload}
    
    Create a clean, beautiful HTML email body:
    1. A short executive overview (2-3 sentences) on current findings across calcium forms (coral, eggshell, synthetic), supplementation efficacy, and osteoporosis care.
    2. An ordered HTML list (`<ol>`) containing ALL 10 articles. For each item include:
       - The title hyperlinked (`<a href="...">`) to its publication URL.
       - A line showing [Source Database], Journal, Authors, and Date in muted grey text.
       - A 2-sentence clinical takeaway highlighting key findings and therapeutic relevance.

    Requirements:
    - Output ONLY valid HTML starting with `<div>` and ending with `</div>`.
    - Do NOT wrap response in markdown code blocks like ```html.
    """

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=3000
        )
    )

    clean_html = response.text.replace("```html", "").replace("```", "").strip()
    return clean_html

# ---------------------------------------------------------------------------
# 6. Email Dispatch Engine
# ---------------------------------------------------------------------------
def send_email(html_content):
    """Sends HTML email digest to listed recipients."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Daily Medical Digest: Calcium & Osteoporosis Research (10 Articles)"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    msg.attach(MIMEText(html_content, "html"))

    recipients = [email.strip() for email in RECEIVER_EMAIL.split(",") if email.strip()]

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER_EMAIL, GMAIL_APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, recipients, msg.as_string())

# ---------------------------------------------------------------------------
# 7. Main Execution Pipeline
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading history...")
    sent_history = load_sent_history()

    print("Gathering 10 unique articles from PubMed, Europe PMC, and Crossref...")
    articles = collect_exact_10_articles(sent_history)

    print("Generating AI digest with gemini-3.6-flash...")
    html_digest = generate_digest_html(articles)

    print("Sending email...")
    send_email(html_digest)

    # Record all 10 new IDs into history
    new_ids = [a['id'] for a in articles]
    sent_history.update(new_ids)
    save_sent_history(sent_history)
    print(f"Success! 10 articles sent and history updated with {len(new_ids)} new entries.")
