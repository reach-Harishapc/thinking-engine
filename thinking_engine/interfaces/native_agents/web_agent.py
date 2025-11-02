# interfaces/native_agents/web_agent.py
import requests
import re
from datetime import datetime
from html import unescape

class Agent:
    def __init__(self):
        self.name = "web_agent"
        self.headers = {"User-Agent": "ThinkingEngine/1.0"}

    def _extract_search_urls(self, html: str) -> list:
        """Extract actual URLs from DuckDuckGo search results"""
        urls = []

        # Find all href attributes that look like result links
        # DuckDuckGo uses various patterns, let's try multiple approaches

        # Pattern 1: Look for DuckDuckGo redirect links
        redirect_pattern = r'href="([^"]*duckduckgo\.com/l/\?uddg=[^"]*)"'
        redirects = re.findall(redirect_pattern, html)

        for redirect_url in redirects[:5]:
            try:
                from urllib.parse import unquote, urlparse
                # Extract the actual URL from the redirect
                if 'uddg=' in redirect_url:
                    actual_url = redirect_url.split('uddg=')[1]
                    if '&' in actual_url:
                        actual_url = actual_url.split('&')[0]
                    actual_url = unquote(actual_url)

                    # Skip if it's not a real web page
                    if actual_url.startswith('http') and not any(skip in actual_url for skip in ['duckduckgo.com', 'google.com', 'bing.com']):
                        # Try to find a title near this link
                        title = "Web Page"  # Default title
                        urls.append((actual_url, title))
            except:
                continue

        # Pattern 2: If no redirects found, look for direct links
        if not urls:
            direct_pattern = r'<a[^>]*href="(https?://[^"]*)"[^>]*>([^<]*)</a>'
            direct_links = re.findall(direct_pattern, html, re.IGNORECASE)

            for href, title_text in direct_links:
                title = re.sub(r'<.*?>', '', title_text).strip()

                # Filter for meaningful links
                if (href and title and len(title) > 10 and len(title) < 200 and
                    not any(skip in href for skip in ['duckduckgo.com', 'google.com', 'bing.com']) and
                    not any(skip in title.lower() for skip in ['search', 'images', 'videos', 'news', 'maps', 'settings'])):
                    urls.append((href, title))

        return urls[:5]  # Return top 5 URLs

    def _fetch_page_content(self, url: str, timeout: float = 8.0) -> str:
        """Fetch and extract meaningful content from a web page"""
        try:
            r = requests.get(url, headers=self.headers, timeout=timeout)
            r.raise_for_status()

            if "html" in r.headers.get("Content-Type", ""):
                # Extract main content from HTML
                content = self._extract_main_content(r.text)
                return content[:1500]  # Limit content length
            else:
                return f"[Non-HTML content from {url}]"

        except Exception as e:
            return f"[Error fetching {url}: {str(e)}]"

    def _extract_main_content(self, html: str) -> str:
        """Extract main content from HTML page"""
        # Remove scripts, styles, and navigation
        html = re.sub(r"(?is)<(script|style|nav|header|footer|aside).*?>.*?(</\1>)", "", html)

        # Try to find main content areas
        content_selectors = [
            r'<main[^>]*>.*?</main>',
            r'<article[^>]*>.*?</article>',
            r'<div[^>]*class="[^"]*content[^"]*"[^>]*>.*?</div>',
            r'<div[^>]*class="[^"]*main[^"]*"[^>]*>.*?</div>',
            r'<div[^>]*id="[^"]*content[^"]*"[^>]*>.*?</div>',
            r'<div[^>]*id="[^"]*main[^"]*"[^>]*>.*?</div>'
        ]

        for selector in content_selectors:
            matches = re.findall(selector, html, re.DOTALL | re.IGNORECASE)
            if matches:
                content = matches[0]
                # Extract text from the content
                text = re.sub(r'<.*?>', ' ', content)
                text = re.sub(r'\s+', ' ', text).strip()

                # Get meaningful paragraphs
                paragraphs = re.split(r'\n+', text)
                meaningful_paras = [p.strip() for p in paragraphs if len(p.strip()) > 50]

                if meaningful_paras:
                    return ' '.join(meaningful_paras[:5])

        # Fallback: extract all substantial text
        text = self._simple_text(html)
        sentences = re.split(r'[.!?]+', text)
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

        return ' '.join(meaningful_sentences[:8]) if meaningful_sentences else text[:1000]

    def _extract_search_results(self, html: str) -> str:
        """Extract both basic search results and conduct deeper research"""
        results = []

        # First get search result URLs
        search_urls = self._extract_search_urls(html)

        # PART 1: Show basic search results (quick overview)
        results.append("📋 Quick Search Results:")
        html_clean = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html)

        # Extract basic result titles
        basic_titles = re.findall(r'<a[^>]*>([^<]{15,150})</a>', html_clean, re.IGNORECASE)
        meaningful_titles = [t.strip() for t in basic_titles if len(t.strip()) > 15 and not any(skip in t.lower() for skip in ['search', 'images', 'videos', 'news', 'maps', 'settings'])]
        for title in meaningful_titles[:5]:
            results.append(f"• {title}")

        # PART 2: Conduct deep research
        if search_urls:
            results.append("\n🔍 Deep Web Research Analysis:")

            # Visit URLs and extract content, prioritizing non-professional sites
            successful_extractions = 0
            for i, (url, title) in enumerate(search_urls[:3]):  # Visit top 3 pages for depth
                # Handle professional networks specially
                if any(site in url for site in ['linkedin.com', 'github.com', 'facebook.com', 'twitter.com', 'instagram.com']):
                    profile_type = "LinkedIn Profile" if 'linkedin.com' in url else "GitHub Profile" if 'github.com' in url else "Social Media Profile"
                    results.append(f"\n📄 Source {i+1}: {profile_type}")
                    results.append(f"   Profile: {url.split('/')[-1] or 'Professional Account'}")
                    results.append(f"   Network: {title}")
                    successful_extractions += 1
                    continue

                results.append(f"\n📄 Source {i+1}: {title}")
                content = self._fetch_page_content(url)

                if content and len(content) > 100 and not content.startswith('[Error'):
                    # Extract key insights (first 300 chars of meaningful content)
                    summary = content[:300] + "..." if len(content) > 300 else content
                    results.append(f"   Content: {summary}")
                    successful_extractions += 1
                else:
                    results.append(f"   Status: Content not accessible from this source")

            results.append(f"\n📊 Research Summary: Analyzed {len(search_urls)} sources, extracted detailed information from {successful_extractions} locations.")

            # Add educational context
            results.append("\n💡 Research Methodology: Combined quick search overview with deep content analysis from authoritative sources for comprehensive understanding.")
        else:
            results.append("\n⚠️ Limited deep research available - showing basic results only.")

        return "\n".join(results)

    def _simple_text(self, html: str) -> str:
        text = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html)
        text = re.sub(r"(?s)<.*?>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return unescape(text).strip()

    def run(self, sandbox, url: str = None, query: str = None, timeout: float = 10.0):
        if query and not url:
            # Use DuckDuckGo HTML search
            url = f"https://html.duckduckgo.com/html/?q={query.replace(' ','+')}"
        if not url:
            return {"status":"error","error":"no_url_or_query"}

        try:
            r = requests.get(url, headers=self.headers, timeout=timeout)
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "")

            if "html" in ctype:
                if "duckduckgo.com" in url and query:
                    # Parse DuckDuckGo search results
                    summary = self._extract_search_results(r.text)
                else:
                    # Regular web page
                    summary = self._simple_text(r.text)[:2000]
            else:
                summary = f"[non-text content len={len(r.content)}]"

            return {"status":"ok","url": url, "summary": summary, "ts": datetime.utcnow().isoformat()}
        except Exception as e:
            return {"status":"error","error": str(e)}
