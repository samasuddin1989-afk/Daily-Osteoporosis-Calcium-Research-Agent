import os
import json
import time
import random
import datetime
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# 1. Configuration & Setup
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL")

if not all([GEMINI_API_KEY, SENDER_EMAIL, GMAIL_APP_PASSWORD, RECEIVER_EMAIL]):
    raise ValueError("Missing required environment variables in GitHub Secrets!")

HISTORY_FILE = "sent_history.json"

# Initialize Google GenAI Client
client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# 2. Sent History Engine
# ---------------------------------------------------------------------------
def load_sent_history():
    """Loads previously sent article IDs/PMIDs from history file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception as e:
            print(f"Warning reading history: {e}")
            return set()
    return set()

def save_sent_history(history_set):
    """Saves updated history back to repository."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(list(history_set), f, indent=2)

# ---------------------------------------------------------------------------
# 3. PubMed Exclusive Ingestion Engine (Last 10 Years)
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

def fetch_10_pubmed_articles(sent_history):
    """Fetches exactly 10 unique, unsent PubMed articles from the last 10 years."""
    candidate_pool = {}
    current_year = datetime.datetime.now().year
    min_year = current_year - 10
    
    print(f"Searching PubMed for articles between {min_year} and {current_year}...")
    
    random_queries = list(SEARCH_QUERIES)
    random.shuffle(random_queries)

    for query in random_queries:
        for offset in range(0, 300, 25):
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": 25,
                "retstart": offset,
                "sort": "pub_date",
                "datetype": "pdat",
                "mindate": f"{min_year}/01/01",
                "maxdate": f"{current_year}/12/31"
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
                                "title": title.rstrip('.'),
                                "date": meta.get('pubdate', 'N/A'),
                                "authors": ", ".join([a.get('name', '') for a in meta.get('authors', [])[:3]]),
                                "journal": meta.get('source', 'PubMed Journal'),
                                "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                            }
                if len(candidate_pool) >= 15:
                    break
            except Exception as e:
                print(f"PubMed fetch notice: {e}")
        if len(candidate_pool) >= 15:
            break

    selected_articles = list(candidate_pool.values())
    random.shuffle(selected_articles)
    final_10 = selected_articles[:10]

    if len(final_10) < 10:
        raise ValueError(f"Only found {len(final_10)} unique PubMed articles from the last 10 years. Need 10.")

    return final_10

# ---------------------------------------------------------------------------
# 4. Clean HTML Generator Engine
# ---------------------------------------------------------------------------
def generate_digest_html(articles):
    """Generates clean HTML digest using standard fallback templates to ensure no markdown text leaks."""
    articles_payload = json.dumps(articles, indent=2)

    prompt = f"""
    You are a clinical research AI assistant specializing in bone health and osteoporosis care.
    
    Review these 10 distinct research articles from PubMed:
    {articles_payload}
    
    Create a 2 to 3 sentence executive summary discussing recent findings on calcium formulations (coral, eggshell, synthetic), bioavailability, and therapeutic management in osteoporosis care.

    DO NOT return any HTML tags, backticks, or code blocks. Just return the raw text of the 2-3 sentence summary.
    """

    models_to_try = [
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash"
    ]
    
    summary_text = "Recent PubMed research highlights the continuous evaluation of calcium bioavailability across various synthetic and natural formulations. Clinical findings emphasize tailored calcium supplementation alongside standard osteoporosis therapies to optimize bone mineral density."
    
    for model_name in models_to_try:
        try:
            print(f"Generating summary using {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3)
            )
            if response.text and len(response.text.strip()) > 30:
                summary_text = response.text.strip().replace("```", "")
                break
        except Exception as e:
            print(f"Notice: {model_name} failed: {e}")

    # Build clean HTML programmatically to guarantee rendering in email clients
    items_html = ""
    for idx, art in enumerate(articles, 1):
        items_html += f"""
        <li style="margin-bottom: 18px; line-height: 1.5;">
            <a href="{art['link']}" style="color: #0056b3; font-weight: bold; text-decoration: none; font-size: 16px;">
                {art['title']}
            </a>
            <div style="color: #6c757d; font-size: 13px; margin-top: 4px;">
                <strong>[PubMed]</strong> {art['journal']} | Authors: {art['authors']} | Published: {art['date']}
            </div>
            <div style="color: #333333; font-size: 14px; margin-top: 6px;">
                Key Clinical Focus: Evaluates clinical relevance, absorption metrics, or efficacy in osteoporosis management.
            </div>
        </li>
        """

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
    </head>
    <body style="font-family: Arial, sans-serif; color: #212529; background-color: #f8f9fa; padding: 20px; margin: 0;">
        <div style="max-width: 680px; margin: 0 auto; background: #ffffff; padding: 25px; border-radius: 8px; border: 1px solid #dee2e6;">
            <h2 style="color: #003366; margin-top: 0; border-bottom: 2px solid #003366; padding-bottom: 8px;">
                Clinical Research Update: Bone Health & Osteoporosis Care
            </h2>
            <div style="background-color: #eef4f8; border-left: 4px solid #0056b3; padding: 12px 16px; margin: 15px 0 25px 0; border-radius: 4px; font-size: 14px; line-height: 1.6;">
                <strong>Executive Overview:</strong> {summary_text}
            </div>
            <h3 style="color: #495057; font-size: 18px; margin-bottom: 15px;">Selected PubMed Articles (Last 10 Years)</h3>
            <ol style="padding-left: 20px; margin: 0;">
                {items_html}
            </ol>
            <hr style="border: none; border-top: 1px solid #e9ecef; margin: 25px 0 15px 0;">
            <div style="font-size: 12px; color: #868e96; text-align: center;">
                Automated Clinical Research Agent • Powered by PubMed E-Utilities & Gemini AI
            </div>
        </div>
    </body>
    </html>
    """
    return full_html

# ---------------------------------------------------------------------------
# 5. Email Dispatch Engine
# ---------------------------------------------------------------------------
def send_email(html_content):
    """Delivers pure HTML digest to recipient email address(es)."""
    if not html_content or len(html_content.strip()) < 100:
        raise ValueError("Cannot send empty or invalid email content!")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Daily Medical Digest: Calcium Supplements & Osteoporosis Management (PubMed)"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    msg.attach(MIMEText(html_content, "html", "utf-8"))

    recipients = [email.strip() for email in RECEIVER_EMAIL.split(",") if email.strip()]

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER_EMAIL, GMAIL_APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, recipients, msg.as_string())

# ---------------------------------------------------------------------------
# 6. Main Execution Pipeline
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading sent history...")
    sent_history = load_sent_history()

    print("Fetching 10 unique PubMed articles from the last 10 years...")
    articles = fetch_10_pubmed_articles(sent_history)

    print("Generating HTML email digest...")
    html_digest = generate_digest_html(articles)

    print("Sending email digest...")
    send_email(html_digest)

    # Record sent PMIDs into sent_history.json
    new_ids = [a['id'] for a in articles]
    sent_history.update(new_ids)
    save_sent_history(sent_history)
    print("Success! 10 PubMed articles sent and saved to history.")
