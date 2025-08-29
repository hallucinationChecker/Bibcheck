import re
import json
import sys
import time
import unicodedata
from PyPDF2 import PdfReader
from habanero import Crossref
import feedparser
import requests
import string

# ANSI Colors
GREEN = "\033[92m"  # matched titles
BLUE = "\033[94m"   # title+author match
RED = "\033[91m"    # no match
RESET = "\033[0m"

cr = Crossref()

# ---------------------------
# Venue alias normalization
# ---------------------------
VENUE_ALIASES = {
    # General ML / Web / Data Mining
    "European Conference on Computer Systems": "EuroSys",
    "International Conference on Machine Learning": "ICML",
    "World Wide Web Conference": "WWW",
    "International World Wide Web Conference": "WWW",
    "ACM SIGKDD International Conference on Knowledge Discovery and Data Mining": "KDD",
    "Advances in Neural Information Processing Systems": "NeurIPS",
    "Conference on Neural Information Processing Systems": "NeurIPS",
    "International Conference on Learning Representations": "ICLR",

    # HPC & Systems
    "International Conference for High Performance Computing, Networking, Storage and Analysis": "SC",
    "International Conference on High Performance Extreme Computing": "HPEC",
    "International Symposium on High-Performance Parallel and Distributed Computing": "HPDC",
    "IEEE International Parallel and Distributed Processing Symposium": "IPDPS",
    "ACM/IEEE International Conference on Supercomputing": "ICS",
    "ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming": "PPoPP",
    "European Conference on Parallel Processing": "Euro-Par",
    "International Conference on Cluster Computing": "CLUSTER",
    "International Conference on High Performance Computing and Communications": "HPCC",
    "International Symposium on Computer Architecture": "ISCA",
    "International Conference on Architectural Support for Programming Languages and Operating Systems": "ASPLOS",
    "International Symposium on Operating Systems Design and Implementation": "OSDI",
    "USENIX Symposium on Networked Systems Design and Implementation": "NSDI",
    "International Conference on Performance Engineering": "ICPE",
    "International Conference on Computer Communications": "INFOCOM"
}


# ---------------------------
# PDF and Text Cleanup
# ---------------------------
def fix_pdf_artifacts1(text: str) -> str:
    """Clean common PDF text extraction issues."""

    # Fix word broken across lines with hyphen: keep the hyphen
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1-\2", text)

    # Hyphen + space + lowercase: remove space
    text = re.sub(r"(\w)-\s+([a-z])", r"\1-\2", text)

    # Hyphen + space + uppercase: keep hyphen
    text = re.sub(r"(\w)-\s+([A-Z])", r"\1-\2", text)

    # Normal newline (no hyphen): replace with space
    text = re.sub(r"(\w)\n(\w)", r"\1 \2", text)

    return text


def fix_pdf_artifacts2(text: str) -> str:
    """
    Clean common PDF text extraction issues:
    - Broken hyphenation across lines: 'At-\ntention' -> 'Attention'
    - Mid-word split: 'At- tention' -> 'Attention'
    - Hyphen + space + lowercase: 'Revis- iting' -> 'Revisiting'
    - Hyphen + space + uppercase: 'Large- Scale' -> 'Large-Scale'
    - Missing space between lowercase + Uppercase: 'GraphDataset' -> 'Graph Dataset'
    - Lost spaces at newlines: 'Graph\nDataset' -> 'Graph Dataset'
    """

    # 1. Remove hyphen + (optional space) + newline + (optional space)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # 2. Remove hyphen + space + lowercase (broken word)
    text = re.sub(r"(\w)-\s+([a-z])", r"\1\2", text)

    # 3. Convert hyphen + space + Uppercase to proper hyphen
    text = re.sub(r"(\w)-\s+([A-Z])", r"\1-\2", text)

    # 4. Fix generic mid-word breaks (e.g., "At- tention" -> "Attention")
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)

    # 5. Insert space at newline if no punctuation
    text = re.sub(r"(\w)\n(\w)", r"\1 \2", text)

    return text

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    #return fix_pdf_artifacts(text)
    return text

def find_bibliography_section(text):
    m = re.search(r"(References|Bibliography)", text, re.IGNORECASE)
    return text[m.start():] if m else None

def parse_acm(bib_text):
    entries = re.split(r"\[\d+\]", bib_text)
    parsed = []
    for i, entry in enumerate(entries, 1):
        clean = entry.strip()
        if len(clean) > 5 and clean.lower() not in {"references", "bibliography"}:
            parsed.append({"id": i, "raw": " ".join(clean.split())})
    return parsed

def parse_ieee(bib_text):
    pattern = r"\[(\d+)\]\s*(.+?)(?=\[\d+\]|\Z)"
    matches = re.findall(pattern, bib_text, re.DOTALL)
    parsed = []
    for num, entry in matches:
        clean = " ".join(entry.split()).strip()
        if len(clean) > 5 and clean.lower() not in {"references", "bibliography"}:
            parsed.append({"id": num, "raw": clean})
    return parsed

# ---------------------------
# Helpers
# ---------------------------
def normalize_text(s):
    if not s:
        return ""
    return unicodedata.normalize("NFKD", str(s)).lower().strip()


def extract_fields(entry):
    authors = []
    title = ""

    # IEEE style (quoted title)
    m = re.search(r"^(.*?)['“\"](.+?)['”\"]", entry)
    if m:
        author_block = m.group(1)
        title = m.group(2)
    else:
        # Book / fallback
        # Split off the year at the end
        parts = re.split(r",?\s*(\d{4})[.,]?\s*$", entry)
        if len(parts) >= 2:
            pre_year = parts[0]   # everything before the year
            # Split into author block vs title at the first comma
            if "," in pre_year:
                author_block, title_part = pre_year.split(",", 1)
                title = title_part.strip()
            else:
                author_block, title = pre_year, ""
        else:
            author_block, title = entry, ""

    # --- Authors ---
    raw_authors = [a.strip() for a in author_block.split(",") if a.strip()]
    for a in raw_authors:
        surname = a.split()[-1].strip(string.punctuation)
        if surname.istitle() and len(surname) > 1:
            authors.append(surname)

    # --- Clean title ---
    #title = re.sub(r"(Springer|Pearson|Elsevier|Wiley|SIAM|PMLR|ACM|IEEE).*", "", title, flags=re.I)
    # Cut off publisher phrases after a period
    title = re.split(
        r"\.\s*(?:In\b|CoRR\b|arXiv\b|arxiv preprint|corr abs/|doi:|https?://|Springer|CRC Press|Wiley|Pearson|Elsevier)",
        title,
        flags=re.I,
        maxsplit=1
    )[0]
    
    title = re.sub(r"\s+", " ", title).strip(" .,;:")

    return title, authors, None



# ---------------------------
# External Checks
# ---------------------------
def check_arxiv_link(entry, force_id=None):
    """
    Check if a direct arXiv link or arXiv:ID is valid,
    and verify the title against the citation.
    """
    if force_id:
        arxiv_id = force_id
    else:
        m = re.search(
            r"(?:arXiv\s*:\s*|https?://arxiv\.org/abs/)"
            r"([0-9]{4}\.[0-9]{4,5}(?:v\d+)?|[a-z\-]+/\d{7}(?:v\d+)?)",
            entry,
            re.I,
        )
        if not m:
            return {"found": False}
        arxiv_id = m.group(1).replace(" ", "")

    url = f"https://arxiv.org/abs/{arxiv_id}"

    try:
        api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        feed = feedparser.parse(api_url)
        if not feed.entries:
            return {"found": False}

        arxiv_entry = feed.entries[0]
        arxiv_title = arxiv_entry.get("title", "").strip()

        # Extract citation title from the *real entry*, not the fake one
        cited_title, _, _ = extract_fields(entry)

        norm_cited_title = normalize_text(cited_title)
        norm_arxiv_title = normalize_text(arxiv_title)

        # Fuzzy token overlap is safer than substring
        cited_tokens = set(re.sub(r"[^a-z0-9 ]", " ", norm_cited_title).split())
        arxiv_tokens = set(re.sub(r"[^a-z0-9 ]", " ", norm_arxiv_title).split())
        overlap = len(cited_tokens & arxiv_tokens)
        required = max(3, len(cited_tokens) // 2)

        if overlap >= required:
            return {
                "found": True,
                "method": "arxiv-link",
                "arxiv_id": arxiv_id,
                "doi": url,
                "title": arxiv_title,
                "venue": "arXiv",
                "url": url,
                "authors": [a.name for a in arxiv_entry.get("authors", [])],
            }

    except Exception:
        pass

    return {"found": False}







def check_doi_link(entry):
    """Check if a direct DOI in the entry is valid via Crossref or ArXiv."""
    m = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", entry)
    if not m:
        return {"found": False}

    doi = m.group(0).lower()

    if doi.startswith("10.48550/arxiv."):
        arxiv_id = doi.split("arxiv.")[-1]
        # Call check_arxiv_link with the *original entry*, but pass arxiv_id explicitly
        return check_arxiv_link(entry, force_id=arxiv_id)

    # Regular Crossref lookup
    try:
        result = cr.works(ids=doi)
        return {
            "found": True,
            "method": "doi-link",
            "doi": doi,
            "title": result["message"].get("title", [""])[0],
            "venue": result["message"].get("container-title", [""])[0]
                      if result["message"].get("container-title") else ""
        }
    except Exception:
        pass

    return {"found": False}






# ---------------------------
# Crossref + fallback
# ---------------------------
def author_overlap(authors, crossref_authors):
    """Loose surname overlap check (at least one surname matches)."""
    for a in authors:
        a = a.lower()
        for ca in crossref_authors:
            ca = ca.lower()
            if a == ca:
                return True
            # fuzzy match: first 4 chars overlap
            if a.startswith(ca[:4]) or ca.startswith(a[:4]):
                return True
    return False


def check_with_crossref(entry):
    """Main validation pipeline with arXiv/DOI priority."""

    # Step 1: direct arXiv link check
    arxiv_direct = check_arxiv_link(entry)
    if arxiv_direct["found"]:
        return arxiv_direct

    # Step 2: direct DOI check
    doi_direct = check_doi_link(entry)
    if doi_direct["found"]:
        return doi_direct

    # Step 3: fallback to title-based Crossref lookup
    title, authors, venue = extract_fields(entry)
    try:
        result = cr.works(query_title=title, limit=3)
        items = result.get("message", {}).get("items", [])
        for item in items:
            crossref_title = item.get("title", [""])[0]
            norm_in_title = normalize_text(title)
            norm_cr_title = normalize_text(crossref_title)

            crossref_authors = [a.get("family", "").lower() for a in item.get("author", [])]

            title_matches = (norm_in_title in norm_cr_title or norm_cr_title in norm_in_title)
            authors_match = (not authors or author_overlap(authors, crossref_authors))

            if title_matches and authors_match:
                return {
                    "found": True,
                    "method": "title+author",
                    "doi": item.get("DOI"),
                    "title": crossref_title,
                    "venue": item.get("container-title", [""])[0] if item.get("container-title") else ""
                }
    except Exception:
        pass

    # Step 4: try OpenAlex
    openalex_result = check_with_openalex(title)
    if openalex_result["found"]:
        return openalex_result

    if ":" in title:
        search_title = title.split(":", 1)[0].strip()
        openalex_result = check_with_openalex(search_title)
        if openalex_result["found"]:
            return openalex_result


    # Step 5: try OpenReview
    openreview_result = check_with_openreview(entry)
    if openreview_result["found"]:
        return openreview_result

    # Step 6: try arXiv search
    arxiv_result = check_with_arxiv(title)
    if arxiv_result["found"]:
        return arxiv_result

    if ":" in title:
        search_title = title.split(":", 1)[1].strip()
        arxiv_result = check_with_arxiv(search_title)
        if arxiv_result["found"]:
            return arxiv_result

    # Step 7: Google Books fallback
    googlebooks_result = check_with_googlebooks(title, authors)
    if googlebooks_result["found"]:
        return googlebooks_result


    return {"found": False}


def clean_title_for_openalex(title: str) -> str:
    # Match ACM (. In …) or IEEE (,” in … or , Journal … etc.)
    title = re.split(
        r'(?:\.\s+In\b|,\s*(?:"|“)?\s*(In|Proceedings|Advances|Journal|Conference|Workshop)\b)',
        title,
        maxsplit=1,
        flags=re.I
    )[0]
    return title.strip(" .,\n\t\"“”")


def check_with_openalex(title):
    import urllib.parse
    cleaned = clean_title_for_openalex(title)

    search_q = urllib.parse.quote(cleaned)
    url = f"https://api.openalex.org/works?filter=title.search:{search_q}"
    r = requests.get(url, timeout=10)

    if r.status_code == 200:
        data = r.json()
        for item in data.get("results", []):
            oa_title = normalize_text(item.get("title", ""))
            if oa_title == normalize_text(cleaned):
                return {
                    "found": True,
                    "method": "openalex",
                    "doi": item.get("doi"),
                    "title": item.get("title"),
                    "venue": item.get("host_venue", {}).get("display_name"),
                    "url": item.get("id"),
                    "authors": [a["author"]["display_name"] for a in item.get("authorships", [])]
                }
    return {"found": False}

def check_with_openreview(entry):
    m = re.search(r"(https?://openreview\.net/forum\?id=\s*\S+)", entry)
    if m:
        url = m.group(1).replace(" ", "")
        try:
            r = requests.head(url, allow_redirects=True, timeout=5)
            if r.status_code == 200:
                return {
                    "found": True,
                    "method": "openreview",
                    "doi": None,
                    "title": None,
                    "venue": "OpenReview",
                    "url": url
                }
        except Exception:
            pass
    return {"found": False}


import requests
import re

def check_with_googlebooks(title, authors):
    """
    Try to find a match on Google Books.
    - Handles cases where article/chapter titles show up in subtitle or textSnippet.
    """
    try:
        query = f"intitle:{title}"
        url = f"https://www.googleapis.com/books/v1/volumes?q={query}"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return {"found": False}

        data = r.json()
        for item in data.get("items", []):
            info = item.get("volumeInfo", {})
            g_title = info.get("title") or ""
            g_subtitle = info.get("subtitle") or ""
            g_snippet = item.get("searchInfo", {}).get("textSnippet", "")

            # Choose the best candidate title
            if g_subtitle and g_subtitle.lower() not in g_title.lower():
                full_title = f"{g_title}: {g_subtitle}"
            elif g_title:
                full_title = g_title
            elif g_snippet:
                full_title = g_snippet
            else:
                continue

            g_authors = info.get("authors", [])
            publisher = info.get("publisher", "")
            published_date = info.get("publishedDate", "")
            url = item.get("selfLink")

            # Normalize and check overlap
            norm_input = set(re.sub(r"[^a-z0-9 ]", " ", title.lower()).split())
            norm_found = set(re.sub(r"[^a-z0-9 ]", " ", full_title.lower()).split())
            overlap = len(norm_input & norm_found)
            required = max(3, len(norm_input) // 2)

            if overlap >= required:
                return {
                    "found": True,
                    "method": "google-books",
                    "doi": None,
                    "title": full_title,
                    "venue": publisher,
                    "url": url,
                    "authors": g_authors,
                    "published": published_date,
                }

    except Exception as e:
        print("Google Books error:", e)

    return {"found": False}





def clean_for_arxiv(title: str) -> str:
    """Make arXiv title query safer (less aggressive trimming)."""
    # Remove common "In Proceedings of ..." but keep the full title otherwise
    title = re.split(r"\.\s*In\s*[A-Z]", title, maxsplit=1)[0]    
    title = re.sub(r"\bIn\s+Proceedings.*", "", title, maxsplit=1, flags=re.I)
    title = re.sub(r"\bIn\s+\d{4}.*", "", title, maxsplit=1, flags=re.I)
    return title.strip(" .,")

def check_with_arxiv(title):
    try:
        clean_title = clean_for_arxiv(title)

        if not clean_title:
            return {"found": False}

        clean_title = clean_title.replace("-", " ")        

        # Build query: first 10 words (better recall than just 8)
        query = "+AND+".join(w.lower() for w in clean_title.split()[:8])
        
        url = f"http://export.arxiv.org/api/query?search_query=ti:{query}&max_results=10"
        feed = feedparser.parse(url)

        for entry in feed.entries:
            arxiv_title = entry.get("title", "").lower()
            cited_title = clean_title.lower()

            # Word overlap (lenient but meaningful)
            overlap = len(set(cited_title.split()) & set(arxiv_title.split()))
            required = max(3, len(cited_title.split()) // 3)  # ~1/3 words must match

            if overlap >= required:
                return {
                    "found": True,
                    "method": "arxiv",
                    "doi": entry.get("id"),
                    "title": entry.get("title"),
                    "venue": "arXiv",
                    "url": entry.get("id"),
                    "authors": [a.name for a in entry.get("authors", [])]
                }
    except Exception:
        pass
    return {"found": False}


# ---------------------------
# Main Logic
# ---------------------------
def parse_and_validate(pdf_path, style="ieee"):
    text = extract_text_from_pdf(pdf_path)
    bib_text = find_bibliography_section(text)
    if not bib_text:
        print("No bibliography found.")
        return []
    entries = parse_ieee(bib_text) if style.lower() == "ieee" else parse_acm(bib_text)
    results = []
    for entry in entries:
        entry1 = entry.copy()
        entry1["raw"] = fix_pdf_artifacts1(entry["raw"])

        # First attempt
        result = {"input": entry1["raw"], "crossref": check_with_crossref(entry1["raw"])}
        results.append(result)
        crossref = result["crossref"]

        if not crossref.get("found"):
            # Retry with more aggressive PDF fixes
            entry1["raw"] = fix_pdf_artifacts2(entry["raw"])
            result = {"input": entry1["raw"], "crossref": check_with_crossref(entry1["raw"])}
            results.append(result)
            crossref = result["crossref"]

        # Print matched titles in green, everything else in blue
        if crossref.get("found"):
            input_title, _, _ = extract_fields(entry1["raw"])
            found_title = crossref.get("title", "")

            print(f"\n{GREEN}INPUT TITLE: {input_title}{RESET}")
            print(f"{GREEN}FOUND TITLE: {found_title}{RESET}\n")

            color = BLUE  # all other JSON output is blue
        else:
            color = RED

        # Print full JSON in blue or red
        print(color + json.dumps(result, indent=2) + RESET, flush=True)

    return results


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python parse_bib.py <pdf_path> <ieee|acm>")
        sys.exit(1)
    pdf_path, style = sys.argv[1], sys.argv[2]
    parsed = parse_and_validate(pdf_path, style)

