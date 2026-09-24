import os
import json
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# 1. Configuration & Credentials
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL")

if not all([GEMINI_API_KEY, SENDER_EMAIL, GMAIL_APP_PASSWORD, RECEIVER_EMAIL]):
    raise ValueError("Missing required environment variables! Check GitHub Secrets.")

HISTORY_FILE = "sent_history.json"

# Initialize Google GenAI Client
client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# 2. History & De-duplication Engine
# ---------------------------------------------------------------------------
def load_sent_history():
    """Loads previously sent article PMIDs to prevent duplicate alerts."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception as e:
            print(f"Warning: Could not parse history file: {e}")
            return set()
    return set()

def save_sent_history(history_set):
    """Saves updated history back to disk."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(list(history_set), f, indent=2)

# ---------------------------------------------------------------------------
# 3. Comprehensive Multi-Query PubMed Ingestion
# ---------------------------------------------------------------------------
def fetch_unique_pubmed_articles(sent_history, target_count=10):
    """
    Searches PubMed across multiple medical queries, filters out duplicates,
    and returns top candidate articles.
    """
    search_queries = [
        "calcium supplementation osteoporosis management",
        "coral calcium OR eggshell calcium OR synthetic calcium",
        "bisphosphonates osteoporosis calcium therapy",
        "calcium carbonate vs citrate bone density"
    ]
    
    candidates = {}
    
    for query in search_queries:
        if len(candidates) >= target_count * 3:
            break
            
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": 20,
            "sort": "pub_date"
        }
        try:
            res = requests.get(search_url, params=params, timeout=15).json()
            id_list = res.get('esearchresult', {}).get('idlist', [])
            
            for pmid in id_list:
                if pmid not in sent_history and pmid not in candidates:
                    candidates[pmid] = query
        except Exception as e:
            print(f"Failed query '{query}': {e}")

    if not candidates:
        return [], []

    # Fetch detailed metadata for collected PMIDs
    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    summary_params = {
        "db": "pubmed",
        "id": ",".join(list(candidates.keys())[:30]),
        "retmode": "json"
    }
    
    details_res = requests.get(summary_url, params=summary_params, timeout=15).json()
    result_dict = details_res.get('result', {})

    selected_articles = []
    new_pmids = []

    for pmid in candidates.keys():
        if pmid in result_dict:
            meta = result_dict[pmid]
            title = meta.get('title', 'No Title')
            pub_date = meta.get('pubdate', 'N/A')
            authors = ", ".join([a.get('name', '') for a in meta.get('authors', [])[:3]])
            journal = meta.get('source', 'Scientific Journal')
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            selected_articles.append({
                "pmid": pmid,
                "title": title,
                "date": pub_date,
                "authors": authors,
                "journal": journal,
                "link": link
            })
            new_pmids.append(pmid)

        if len(selected_articles) == target_count:
            break

    return selected_articles, new_pmids

# ---------------------------------------------------------------------------
# 4. Summarization via Gemini 3.6 Flash
# ---------------------------------------------------------------------------
def generate_digest_html(articles):
    """Generates a clean HTML research digest using Google GenAI SDK."""
    articles_payload = json.dumps(articles, indent=2)

    prompt = f"""
    You are a clinical research AI assistant specializing in endocrinology, rheumatology, and bone health.
    
    Review these 10 distinct articles/journals:
    {articles_payload}
    
    Create a clean, beautiful, HTML-formatted email body containing:
    1. A brief executive intro (2-3 sentences) summarizing recent clinical discussions on calcium forms (e.g., eggshell, coral, synthetic), supplementation efficacy, and osteoporosis management (including bisphosphonates/medications).
    2. An ordered HTML list (`<ol>`) of all 10 items. For each item include:
       - The title hyperlinked (`<a href="...">`) to its PubMed page.
       - A secondary line showing Journal Name, Authors, and Publication Date in muted grey text.
       - A 2-sentence clinical takeaway covering key findings, supplement type, or therapeutic relevance.

    Requirements:
    - Output ONLY valid raw HTML (start with `<div>` or `<html>`).
    - Do NOT include markdown code blocks or ```html markers.
    """

    # Model endpoint updated to gemini-3.6-flash
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=2500
        )
    )

    clean_html = response.text.replace("```html", "").replace("```", "").strip()
    return clean_html

# ---------------------------------------------------------------------------
# 5. Email Delivery Function
# ---------------------------------------------------------------------------
def send_email(html_content):
    """Delivers HTML digest to recipient(s) via Gmail SMTP."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = " Daily Medical Digest: Calcium Supplements & Osteoporosis Management"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    msg.attach(MIMEText(html_content, "html"))

    recipients = [email.strip() for email in RECEIVER_EMAIL.split(",") if email.strip()]

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER_EMAIL, GMAIL_APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, recipients, msg.as_string())

# ---------------------------------------------------------------------------
# 6. Main Execution Pipeline
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading history...")
    sent_history = load_sent_history()

    print("Fetching unique articles from PubMed...")
    articles, new_pmids = fetch_unique_pubmed_articles(sent_history, target_count=10)

    if not articles:
        print("No new unique articles found.")
    else:
        print(f"Retrieved {len(articles)} new articles. Generating AI digest with gemini-3.6-flash...")
        html_digest = generate_digest_html(articles)

        print("Sending email...")
        send_email(html_digest)

        # Update and save local history file
        sent_history.update(new_pmids)
        save_sent_history(sent_history)
        print("Execution complete & history updated!")
