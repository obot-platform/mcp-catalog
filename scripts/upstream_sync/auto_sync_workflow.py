#!/usr/bin/env python3
"""
MCP Registry Sync Workflow

This script synchronizes MCP servers from the official registry with our catalog,
creating GitHub issues for new servers that meet our criteria.
"""

import os
import time
import json
import base64
import re
import math
import datetime as dt
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse, urlunparse
from difflib import SequenceMatcher

import requests
import yaml
from packaging.version import Version, InvalidVersion
from openai import OpenAI

# =============================================================================
# CONFIGURATION
# =============================================================================

# Registry and repository configuration
REGISTRY_URL = os.getenv("REGISTRY_URL", "https://registry.modelcontextprotocol.io/v0/servers")
TARGET_REPO_FULL = os.getenv("GITHUB_REPOSITORY", "obot-platform/mcp-catalog")
CATALOG_OWNER = os.getenv("CATALOG_OWNER", "obot-platform")
CATALOG_REPO = os.getenv("CATALOG_REPO", "mcp-catalog")

# Authentication tokens
ISSUE_TOKEN = os.environ["GITHUB_TOKEN"]
CATALOG_TOKEN = os.environ["GITHUB_TOKEN"]

# Runtime configuration
ISSUE_LABELS = [s.strip() for s in os.getenv("ISSUE_LABELS", "VerifiedMCPServer").split(",") if s.strip()]
STAR_MIN = int(os.getenv("STAR_MIN", "500"))
RECENT_DAYS = int(os.getenv("RECENT_DAYS", "30"))

# State tracking
SELECTED_SERVERS_FILE = os.path.join(
    os.path.dirname(__file__) if __file__ else ".",
    "selected_server.json"
)

# Parse repository info
OWNER, REPO = TARGET_REPO_FULL.split("/", 1)

# Initialize HTTP sessions
S = requests.Session()
S.headers.update({"User-Agent": "mcp-registry-sync/1.0"})

GH = requests.Session()
GH.headers.update({
    "User-Agent": "mcp-catalog-selector/1.1", 
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"
})

# Initialize OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _auth_headers(token: str) -> Dict[str, str]:
    """Generate authorization headers for GitHub API."""
    return {
        "Authorization": f"Bearer {token}", 
        "Accept": "application/vnd.github+json"
    } if token else {"Accept": "application/vnd.github+json"}

def first_str(*vals) -> Optional[str]:
    """Return the first non-empty string from the arguments."""
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def _norm(s: str | None) -> str:
    """Normalize string for comparison by removing special chars and common suffixes."""
    if not s: 
        return ""
    s = s.lower()
    s = re.sub(r"[\W_]+", "", s)  # keep alnum only
    s = re.sub(r"(inc|corp|labs|llc|ltd|hq)$", "", s)  # drop common tails
    s = re.sub(r"(ai|app)$", "", s)  # mild heuristic
    return s

def _sim(a: str, b: str) -> float:
    """Calculate similarity ratio between two normalized strings."""
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()

def days_since(iso: str | None) -> float:
    """Calculate days since an ISO timestamp."""
    if not iso: 
        return math.inf
    t = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return (dt.datetime.now(dt.timezone.utc) - t).days

def normalize_url(url):
    """Normalize URL for comparison."""
    parsed = urlparse(url)
    netloc = parsed.hostname.lower()
    if parsed.port and not (
        (parsed.scheme == "http" and parsed.port == 80) or
        (parsed.scheme == "https" and parsed.port == 443)
    ):
        netloc += f":{parsed.port}"
    path = parsed.path or "/"
    return urlunparse((parsed.scheme.lower(), netloc, path.rstrip("/"), "", "", ""))

# =============================================================================
# REGISTRY AND VERSION HANDLING
# =============================================================================

def _server_key(e: dict) -> str:
    """Generate a unique key for a server entry."""
    meta = (e.get("_meta") or {}).get("io.modelcontextprotocol.registry/official") or {}
    if meta.get("serverId"):
        return f"id:{meta['serverId']}"
    if e.get("name"):
        return f"name:{e['name'].lower()}"
    repo = (e.get("repository") or {}).get("url", "")
    try:
        u = urlparse(repo)
        if u.netloc == "github.com":
            parts = [p for p in u.path.strip("/").split("/") if p]
            if len(parts) >= 2:
                return f"repo:{parts[0].lower()}/{parts[1].removesuffix('.git').lower()}"
    except Exception:
        pass
    return f"obj:{id(e)}"

def _parse_ver_str(v: Optional[str]) -> Optional[Version]:
    """Parse version string into Version object."""
    if not isinstance(v, str) or not v:
        return None
    try:
        return Version(v.lstrip("vV"))
    except InvalidVersion:
        return None

def fetch_registry_servers() -> list[dict]:
    """Fetch all servers with pagination; keep only the highest version per server."""
    best: dict[str, Tuple[Optional[Version], dict]] = {}  # key -> (semver, entry)
    params, attempts = {"limit": 100}, 0

    while True:
        r = S.get(REGISTRY_URL, params=params, timeout=30)
        if r.status_code >= 500 and attempts < 3:
            attempts += 1
            time.sleep(2 ** attempts)
            continue
        r.raise_for_status()
        data = r.json()
        items = data.get("servers") or data.get("items") or data.get("data") or data
        if not isinstance(items, list):
            raise RuntimeError(f"Unexpected registry payload keys: {list(data.keys())}")

        for entry in items:
            e = entry.get("server")
            if not e:
                continue
            e["active"] = entry.get("_meta", {}).get("io.modelcontextprotocol.registry/official", {}).get("status", "").lower()
            k = _server_key(e)
            new_v = _parse_ver_str(e.get("version"))
            old = best.get(k)
            if not old:
                best[k] = (new_v, e)
            else:
                old_v, _ = old
                # choose if new_v is strictly newer; otherwise keep existing
                if new_v is not None and (old_v is None or new_v > old_v):
                    best[k] = (new_v, e)

        metadata = data.get("metadata")
        nxt = metadata.get("nextCursor") or metadata.get("next_cursor") or (metadata.get("cursor") or {}).get("next")
        if not nxt:
            break
        params["cursor"] = nxt

    return [v[1] for v in best.values()]

# =============================================================================
# GITHUB API FUNCTIONS
# =============================================================================

def github_api(url: str, token: str = "", params=None) -> dict:
    """Make authenticated GitHub API request."""
    r = S.get(url, headers=_auth_headers(token), params=params or {}, timeout=30)
    if r.status_code == 403 and "rate limit" in r.text.lower():
        reset = r.headers.get("x-ratelimit-reset")
        raise RuntimeError(f"GitHub API rate-limited. Try later. reset={reset}")
    r.raise_for_status()
    return r.json()

def parse_repo_url(url: str):
    """Parse GitHub repository URL into owner/repo tuple."""
    try:
        u = urlparse(url)
        if u.netloc.lower() != "github.com": 
            return (None, None)
        parts = [p for p in u.path.strip("/").split("/") if p]
        if len(parts) < 2: 
            return (None, None)
        return (parts[0], parts[1].removesuffix(".git"))
    except Exception:
        return (None, None)

def get_default_branch(owner: str, repo: str, token: str = "") -> str:
    """Get the default branch for a repository."""
    data = github_api(f"https://api.github.com/repos/{owner}/{repo}", token)
    return data.get("default_branch", "main")

def repo_info(owner: str, repo: str):
    """Get repository information using the authenticated session."""
    r = GH.get(f"https://api.github.com/repos/{owner}/{repo}", timeout=20)
    r.raise_for_status()
    d = r.json()
    return {
        "owner_login": (d.get("owner") or {}).get("login", ""),
        "owner_type": (d.get("owner") or {}).get("type", ""),
        "stars": d.get("stargazers_count", 0),
        "pushed_at": d.get("pushed_at"),
        "is_fork": bool(d.get("fork")),
        "is_archived": bool(d.get("archived")),
    }

def org_verified(owner_login: str) -> bool:
    """Check if a GitHub organization is verified."""
    r = GH.get(f"https://api.github.com/orgs/{owner_login}", timeout=20)
    if r.status_code == 404:
        return False
    r.raise_for_status()
    return bool(r.json().get("is_verified", False))

# =============================================================================
# CATALOG MANAGEMENT
# =============================================================================

def list_yaml_paths(owner: str, repo: str, token: str = "") -> List[str]:
    """List *.yml/*.yaml files in the root directory only."""
    branch = get_default_branch(owner, repo, token)
    tree = github_api(f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}", token)
    paths = []
    for node in tree.get("tree", []):
        if (node.get("type") == "blob" and 
            node.get("path", "").lower().endswith((".yaml", ".yml")) and
            "/" not in node.get("path", "")):  # Only root directory files
            paths.append(node["path"])
    return paths

def read_file_text(owner: str, repo: str, path: str, token: str = "") -> str:
    """Read file content from GitHub repository."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    data = github_api(url, token)
    if isinstance(data, dict) and data.get("encoding") == "base64":
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    # fallback if api returns redirect/download_url
    download_url = data.get("download_url")
    if download_url:
        r = S.get(download_url, timeout=30)
        r.raise_for_status()
        return r.text
    raise RuntimeError(f"Unsupported content response shape for {path}")

def load_y_ids_from_catalog(owner: str, repo: str, token: str = "") -> list[dict]:
    """Load catalog entries from YAML files."""
    res = []
    for path in list_yaml_paths(owner, repo, token):
        try:
            txt = read_file_text(owner, repo, path, token)
            doc = yaml.safe_load(txt)
        except Exception:
            continue
        docs = doc if isinstance(doc, list) else [doc]
        for d in docs:
            short_desc = d.get("description", "").split("## Features")[0].strip()
            r = {
                "name": d.get("name"), 
                "repoURL": d.get("repoURL"), 
                "runtime": d.get("runtime"), 
                "short_desc": short_desc
            }
            if r["runtime"] == "remote":
                r["remoteConfig"] = d.get("remoteConfig")
            res.append(r)
    return res

# =============================================================================
# ISSUE MANAGEMENT
# =============================================================================

def create_issue_for_server(server: dict) -> tuple[str, int, str]:
    """
    Create a GitHub issue for an MCP server using its metadata.
    
    Returns:
        Tuple of (issue_url, issue_number, node_id)
    """
    name = server.get('name', 'Unknown')
    title = f"[MCP Catalog] New MCP server candidate: {name}"
    
    # Extract metadata
    description = server.get('description', '')
    version = server.get('version', '')
    repo_info = server.get('repository', {})
    repo_url = repo_info.get('url', '') if repo_info else ''
    
    # Extract packages info
    packages = server.get('packages', [])
    pkg_info = []
    if packages:
        for pkg in packages:
            if isinstance(pkg, dict):
                pkg_id = pkg.get('identifier', '')
                pkg_version = pkg.get('version', '')
                registry_type = pkg.get('registryType', '')
                if pkg_id:
                    pkg_info.append(f"  - {pkg_id} (v{pkg_version}, {registry_type})")
    
    # Extract remotes info
    remotes = server.get('remotes', [])
    remote_info = []
    if remotes:
        for remote in remotes:
            if isinstance(remote, dict):
                remote_type = remote.get('type', '')
                remote_url = remote.get('url', '')
                if remote_url:
                    remote_info.append(f"  - {remote_type}: {remote_url}")
    
    # Build the issue body
    body_parts = [
        "Automatically Discovered via MCP Registry.", 
        ""
    ]
    
    if name:
        body_parts.append(f"**Name:** {name}")
    if "kind" in server:
        body_parts.append(f"**Kind:** {server['kind']}")
    if description:
        body_parts.append(f"**Description:** {description}")
    if version:
        body_parts.append(f"**Version:** {version}")
    if repo_url:
        body_parts.append(f"**Repository:** {repo_url}")
    
    if pkg_info:
        body_parts.extend(["", "**Packages:**"] + pkg_info)
    
    if remote_info:
        body_parts.extend(["", "**Remote Endpoints:**"] + remote_info)
    
    if server.get('kind'):
        body_parts.extend(["", f"**Server Type:** {server['kind']}"])
    
    body_parts.extend([
        "", "---", "", 
        "If we want to catalog this server, please add/update its YAML in `obot-platform/mcp-catalog` and link the PR here."
    ])
    
    body = "\n".join(body_parts)
    
    print("→", title)
    
    if not ISSUE_TOKEN:
        print(f"   [DRY RUN] Would create issue with body:\n{body[:200]}...")
        return "https://github.com/example/repo/issues/12345", 12345, "mock_node_id"  # Mock for dry run
    
    headers = {
        'Authorization': f'token {ISSUE_TOKEN}',
        'Accept': 'application/vnd.github+json',
    }
    
    r = requests.post(
        f"https://api.github.com/repos/{CATALOG_OWNER}/{CATALOG_REPO}/issues",
        headers=headers,
        json={"title": title, "body": body, "labels": ISSUE_LABELS or [], "type": "Feature"},
        timeout=30
    )
    r.raise_for_status()
    
    issue_data = r.json()
    issue_url = issue_data.get('html_url', 'unknown URL')
    issue_number = issue_data.get('number', 0)
    node_id = issue_data.get('node_id', '')
    
    print(f"   ✓ Created issue #{issue_number}: {issue_url}")
    return issue_url, issue_number, node_id

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def load_selected_servers() -> dict:
    """Load the selected servers data from the JSON file."""
    if not os.path.exists(SELECTED_SERVERS_FILE):
        print(f"No selected servers file found at {SELECTED_SERVERS_FILE}, starting fresh")
        return {}
    
    try:
        with open(SELECTED_SERVERS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Loaded {len(data)} previously processed servers from {SELECTED_SERVERS_FILE}")
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"Warning: Could not load selected servers file: {e}")
        return {}

def save_selected_servers(selected_servers: dict):
    """Save the selected servers data to the JSON file."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(SELECTED_SERVERS_FILE), exist_ok=True)
        
        # Write with pretty formatting
        with open(SELECTED_SERVERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(selected_servers, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(selected_servers)} selected servers to {SELECTED_SERVERS_FILE}")
    except Exception as e:
        print(f"Error saving selected servers file: {e}")

def add_server_to_state(server: dict, issue_url: str, existing_servers: dict) -> dict:
    """Add a newly processed server to the state dict."""
    server_name = server.get('name', '')
    if not server_name:
        return existing_servers
    
    server_record = {
        'name': server_name,
        'description': server.get('description', ''),
        'version': server.get('version', ''),
        'repository_url': server.get('repository', {}).get('url', ''),
        'server_type': server.get('kind', ''),
        'issue_url': issue_url,
        'processed_at': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    }
    
    existing_servers[server_name] = server_record
    return existing_servers

# =============================================================================
# AI CATEGORIZATION CACHE
# =============================================================================

CACHE_FILE_PATH = os.path.join(
    os.path.dirname(__file__), 
    "ai_categorization_cache.json"
)

def load_ai_cache() -> dict:
    """
    Load AI categorization cache from JSON file.
    
    Returns:
        Dict mapping server identifiers to categorization results:
        {
            "server_name": {
                "ai_decision": "official" | "community" | "uncertain",
                "ai_confidence": 0.0-1.0,
                "ai_reason": "reason text",
                "cached_at": "ISO timestamp",
                "repository_url": "https://..."
            }
        }
    """
    if not os.path.exists(CACHE_FILE_PATH):
        print(f"No AI cache found at {CACHE_FILE_PATH}, starting fresh")
        return {}
    
    try:
        with open(CACHE_FILE_PATH, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        print(f"✓ Loaded AI categorization cache with {len(cache)} entries")
        return cache
    except Exception as e:
        print(f"Warning: Could not load AI cache: {e}")
        return {}

def save_ai_cache(cache: dict):
    """
    Save AI categorization cache to JSON file.
    
    Args:
        cache: Dict mapping server identifiers to categorization results
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(CACHE_FILE_PATH), exist_ok=True)
        
        # Write with pretty formatting
        with open(CACHE_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved AI categorization cache with {len(cache)} entries to {CACHE_FILE_PATH}")
    except Exception as e:
        print(f"Error saving AI cache: {e}")

def get_cache_key(server: dict) -> str:
    """
    Generate a stable cache key for a server.
    Uses repository URL if available, otherwise server name.
    """
    repo_url = server.get('repository', {}).get('url', '')
    repo_name = server.get("name", "")
    if repo_url:
        return f"{repo_name}:{normalize_url(repo_url)}"
    return server.get('name', f"unknown_{id(server)}")

# =============================================================================
# AI-POWERED CLASSIFICATION
# =============================================================================

AI_OFFICIAL_JUDGE_PROMPT = """You are a reviewer deciding whether a GitHub MCP server is "official" or "community".

Definition:
- "Official": Published by the organization that owns/operates the underlying product/service the server integrates with (e.g., Google→Gmail, Teamwork→teamwork.com).
- "Community": Any third-party implementation for someone else's product/service.

Inputs you may receive:
- server_name, github_org, repository URL, remotes (URLs), description,
- repo metadata (owner type Organization/User, verified flag, fork/archived),
- packages/identifiers (e.g., npm scope, docker repo) if available.

Heuristics (no static allowlists; infer from evidence):
Strong signals of Official:
- Organization name matches the product/brand (normalized) found in server_name or repo (e.g., chrome-devtools ⇄ ChromeDevTools; mcpcap ⇄ mcpcap).
- Remote endpoints are on the organization's domain and reference the product (e.g., mcp.ai.teamwork.com, mcp.blockscout.com, strata.klavis.ai).
- Verified GitHub Organization; repository is not a fork and not archived.
- Package namespace owned by the org (e.g., npm scope @brave/*, docker image under the org).
- if `Test` is in the server_name, then it might be a test server and should be community. Judge this by your best judgement.

Generic services require stricter proof:
- For generic/ubiquitous services {"gmail","postgres","postgresql","mysql","slack","github","jira","confluence","notion","airtable","sheets","docs"}:
  Only mark Official if the publisher is the actual brand owner (e.g., Google for Gmail, Slack for Slack).
  Otherwise classify as Community, even if the org's domain appears in remotes.

Weak/noisy signals (use with caution):
- Reversed-domain in server name and domain similarity alone are insufficient for generic services.

Negative signals (reduce confidence):
- Repo is a fork or archived; owner is a personal user (not an org);
- Remotes hosted on generic third‑party platforms without clear brand ownership evidence.

Output JSON only:
{ "decision": "official" | "community" | "uncertain", "confidence": 0.0..1.0, "reason": "<one concise sentence>" }
Keep reasons short and evidence‑based (mention org, service, and the key signal).
"""

def gpt5_judge_service_ownership(server):
    """Use AI to determine if a server is official or community."""
    response = client.responses.create(
        model="gpt-4.1",
        input=AI_OFFICIAL_JUDGE_PROMPT + "\n\nInput:\n" + json.dumps(server)
    )
    return response

def name_reversed_domain(name: str | None) -> str | None:
    """Extract vendor token from reversed domain name."""
    if not name or "/" not in name: 
        return None
    prefix = name.split("/", 1)[0]  # e.g., com.google
    labels = prefix.split(".")
    if len(labels) < 2: 
        return None
    sld = labels[-1]  # 'google' or 'containers'
    return _norm(sld)

def remote_domains(remotes: list | None) -> set[str]:
    """Extract normalized domain names from remote URLs."""
    out = set()
    for r in remotes or []:
        try:
            host = urlparse(r.get("url", "")).netloc.lower()
            if host:
                parts = host.split(".")
                if len(parts) >= 2:
                    out.add(_norm(parts[-2]))  # SLD: vercel, atlassian, google
        except Exception:
            pass
    return out

def is_popular_community(repo_meta: dict) -> bool:
    """Check if a community server meets popularity criteria."""
    return (
        repo_meta["stars"] >= STAR_MIN and
        days_since(repo_meta["pushed_at"]) <= RECENT_DAYS and
        not repo_meta["is_archived"]
    )

# =============================================================================
# FILTERING AND CLASSIFICATION
# =============================================================================

def filter_group_x_ai(servers: List[dict], catalog_entries: List[dict], existing_servers: List[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
    """Filter MCP servers using AI judge for official/community classification."""
    filtered_servers = []
    non_active = []
    likely_remote = []
    
    # Load AI categorization cache
    ai_cache = load_ai_cache()
    cache_hits = 0
    cache_misses = 0
    
    print(f"Starting AI-enhanced filtering with {len(servers)} servers...")
    url_sets = set(normalize_url(y.get("repoURL")) for y in catalog_entries)
    one_long_name_str = " ".join([y.get("name") for y in catalog_entries]).lower()
    
    for i, server in enumerate(servers):
        # Basic filter: must be active
        if str(server.get("active", "")).lower() != "active":
            non_active.append(server)
            continue

        full_name = server.get("name")
        if full_name in existing_servers:
            print(f"Skipping server {full_name} because it already exists in the state index.")
            continue
            
        name = full_name.split("/")[-1].lower()
        if name != "mcp" and name in one_long_name_str:
            print(f"[SKIP] Skipping server {full_name} because it is a duplicate of an existing server in the catalog.")
            continue
        
        url = server.get("repository", {}).get("url", "")
        if url:
            if normalize_url(url) in url_sets:
                print(f"Skipping server {server.get('name')} {url} because it already exists in the catalog, URL duplicate.")
                continue
                
        owner, repo = parse_repo_url(url)
        if not owner:
            if "remotes" in server:
                server["kind"] = "remote"
                likely_remote.append(server)
            continue

        try:
            m = repo_info(owner, repo)
        except Exception as err:
            print(f"Error fetching repo info for {owner}/{repo}: {err}")
            continue

        # Check AI categorization cache first
        cache_key = get_cache_key(server)
        cached_result = ai_cache.get(cache_key)
        
        if cached_result:
            # Use cached AI decision
            ai_result = {
                "decision": cached_result["ai_decision"],
                "confidence": cached_result["ai_confidence"],
                "reason": cached_result["ai_reason"]
            }
            cache_hits += 1
            print(f"  💾 Using cached AI decision for {owner}/{repo}")
        else:
            # Call AI judge to determine official vs community
            ai_response = gpt5_judge_service_ownership(server)
            ai_result = json.loads(ai_response.output[0].content[0].text)
            
            # Save to cache
            ai_cache[cache_key] = {
                "ai_decision": ai_result["decision"],
                "ai_confidence": ai_result["confidence"],
                "ai_reason": ai_result["reason"],
                "cached_at": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                "repository_url": url,
                "server_name": server.get('name', '')
            }
            cache_misses += 1
        
        server["_ai_decision"] = ai_result["decision"]
        server["_ai_confidence"] = ai_result["confidence"]
        server["_ai_reason"] = ai_result["reason"]

        # Determine final classification
        if ai_result["decision"] == "official":
            server["kind"] = "official"
            print(f"  ✅ Official: {owner}/{repo} (AI confidence: {ai_result['confidence']:.2f})")
            print(f"    🤖 AI: {ai_result['reason']}")
            filtered_servers.append(server)
            
        elif ai_result["decision"] == "community":
            # Apply community filtering criteria (stars + recent activity)
            community_ok = is_popular_community(m)
            
            if community_ok:
                server["kind"] = "community"
                print(f"  ✓ Community: {owner}/{repo} (AI confidence: {ai_result['confidence']:.2f}, stars: {m['stars']}, recent: {days_since(m['pushed_at']):.0f}d)")
                print(f"    🤖 AI: {ai_result['reason']}")
                filtered_servers.append(server)
            else:
                print(f"  ✗ Community (filtered): {owner}/{repo} - insufficient stars ({m['stars']}) or not recent ({days_since(m['pushed_at']):.0f}d)")
                print(f"    🤖 AI: {ai_result['reason']}")
        
        else:  # uncertain
            print(f"  ❓ Uncertain: {owner}/{repo} (AI confidence: {ai_result['confidence']:.2f})")
            print(f"    🤖 AI: {ai_result['reason']}")

    official_count = len([s for s in filtered_servers if s.get("kind") == "official"])
    community_count = len([s for s in filtered_servers if s.get("kind") == "community"])
    
    # Save updated AI cache
    if cache_misses > 0:
        save_ai_cache(ai_cache)
    
    print(f"\nFiltered down to {len(filtered_servers)} servers:")
    print(f"  - Official: {official_count}")
    print(f"  - Community: {community_count}")
    print(f"\nAI Cache Statistics:")
    print(f"  - Cache hits: {cache_hits}")
    print(f"  - Cache misses (new AI calls): {cache_misses}")
    print(f"  - Total cached entries: {len(ai_cache)}")
    
    return filtered_servers, non_active, likely_remote
_project_id_cache = {}

def get_project_id(project_number: int, owner: str, token: str = None) -> str:
    """
    Get the GitHub project ID from project number (cached).
    
    Args:
        project_number: The project number (e.g., 2)
        owner: GitHub username or organization name
        token: GitHub personal access token
    
    Returns:
        Project ID string, or None if not found
    """
    cache_key = f"{owner}:{project_number}"
    
    # Return cached value if available
    if cache_key in _project_id_cache:
        return _project_id_cache[cache_key]
    
    if token is None:
        token = os.getenv('GITHUB_TOKEN')
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
    }
    
    query = """
    query($owner: String!, $number: Int!) {
      organization(login: $owner) {
        projectV2(number: $number) {
          id
        }
      }
      user(login: $owner) {
        projectV2(number: $number) {
          id
        }
      }
    }
    """
    
    response = requests.post(
        'https://api.github.com/graphql',
        json={'query': query, 'variables': {'owner': owner, 'number': project_number}},
        headers=headers
    )
    
    if response.status_code != 200:
        print(f"Error getting project: {response.status_code}")
        return None
    
    data = response.json()
    
    # Extract project ID
    project_id = None
    if data.get('data', {}).get('organization', {}).get('projectV2'):
        project_id = data['data']['organization']['projectV2']['id']
    elif data.get('data', {}).get('user', {}).get('projectV2'):
        project_id = data['data']['user']['projectV2']['id']
    
    if project_id:
        _project_id_cache[cache_key] = project_id
        print(f"✓ Found project ID: {project_id}")
    
    return project_id

def add_issue_to_project(issue_node_id: str, issue_number: int, project_id: str, token: str = None) -> bool:
    """
    Add an existing issue to a GitHub project.
    
    Args:
        issue_node_id: Issue's GraphQL node ID (from create response)
        issue_number: Issue number (for logging only)
        project_id: Project ID (from get_project_id - cached)
        token: GitHub personal access token
    
    Returns:
        True if successful, False otherwise
    """
    
    if token is None:
        token = os.getenv('GITHUB_TOKEN')
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
    }
    
    if not issue_node_id:
        print(f"   ✗ Missing node ID for issue #{issue_number}")
        return False
    
    # Add issue to project
    add_item_mutation = """
    mutation($projectId: ID!, $contentId: ID!) {
      addProjectV2ItemById(input: {projectId: $projectId, contentId: $contentId}) {
        item {
          id
        }
      }
    }
    """
    
    response = requests.post(
        'https://api.github.com/graphql',
        json={'query': add_item_mutation, 'variables': {
            'projectId': project_id,
            'contentId': issue_node_id
        }},
        headers=headers
    )
    
    if response.status_code != 200:
        print(f"Error adding item to project: {response.status_code}")
        return False
    
    data = response.json()
    
    if 'errors' in data:
        print("Error adding item to project:")
        for error in data['errors']:
            print(f"  - {error.get('message', error)}")
        return False
    
    print(f"   ✓ Added issue #{issue_number} to project")
    return True

def get_issue_node_id(issue_number: int, owner: str, repo: str, token: str) -> Optional[str]:
    """
    Get the GraphQL node ID for an issue.
    
    Args:
        issue_number: Issue number
        owner: Repository owner
        repo: Repository name
        token: GitHub personal access token
    
    Returns:
        Node ID string or None if not found
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()["node_id"]
    return None

def add_sub_issue_graphql(parent_node_id: str, child_node_id: str,
                          token: str, parent_issue_number: int = None, child_issue_number: int = None) -> bool:
    """
    Add a sub-issue using GitHub's GraphQL API.
    
    Args:
        parent_node_id: Parent issue's GraphQL node ID (cached)
        child_node_id: Child issue's GraphQL node ID (from create response)
        token: GitHub personal access token
        parent_issue_number: Parent issue number (for logging only)
        child_issue_number: Child issue number (for logging only)
    
    Returns:
        True if successful
    """
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    if not child_node_id:
        if child_issue_number:
            print(f"   ✗ Missing node ID for issue #{child_issue_number}")
        else:
            print(f"   ✗ Missing child node ID")
        return False
    
    # Step 2: Use GraphQL mutation to add sub-issue
    query = """
    mutation AddSubIssue($parentIssueId: ID!, $subIssueId: ID!) {
      addSubIssue(input: {issueId: $parentIssueId, subIssueId: $subIssueId}) {
        issue {
          id
          title
        }
        subIssue {
          id
          title
          url
        }
      }
    }
    """
    
    variables = {
        'parentIssueId': parent_node_id,
        'subIssueId': child_node_id
    }
    
    response = requests.post(
        "https://api.github.com/graphql",
        headers=headers,
        json={"query": query, "variables": variables}
    )
    
    result = response.json()
    
    if 'errors' in result:
        print(f"   ✗ GraphQL errors adding sub-issue:")
        for error in result['errors']:
            print(f"     - {error.get('message')}")
        return False
    
    if 'data' in result and result['data'].get('addSubIssue'):
        if parent_issue_number:
            print(f"   ✓ Added as sub-issue of #{parent_issue_number}")
        else:
            print(f"   ✓ Added as sub-issue")
        return True
    
    print(f"   ✗ Unexpected response: {result}")
    return False

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("=== MCP Server Catalog Selection Workflow ===\n")
    PROJECT_NUMBER = 2  # fixed Obot AI project
    INDEX_ISSUE_NUMBER = 143  # Index issue for all MCP servers
    
    project_id = get_project_id(PROJECT_NUMBER, CATALOG_OWNER, ISSUE_TOKEN)
    if not project_id:
        print("Error: Could not get project ID")
        exit(1)
    
    # Fetch parent node ID once (cached for all sub-issue operations)
    print(f"Fetching node ID for index issue #{INDEX_ISSUE_NUMBER}...")
    parent_node_id = get_issue_node_id(INDEX_ISSUE_NUMBER, CATALOG_OWNER, CATALOG_REPO, ISSUE_TOKEN)
    if not parent_node_id:
        print(f"Warning: Could not get node ID for index issue #{INDEX_ISSUE_NUMBER}")
        print("Sub-issues will not be created, but other operations will continue.")
    else:
        print(f"✓ Cached parent node ID for issue #{INDEX_ISSUE_NUMBER}")
    # Step 1: Fetch servers from registry
    print("Fetching servers from MCP registry...")
    servers = fetch_registry_servers()
    print(f"Found {len(servers)} servers from registry")

    # Step 2: Load existing catalog IDs to avoid duplicates
    print("\nLoading existing catalog IDs...")
    catalog_entries = load_y_ids_from_catalog(CATALOG_OWNER, CATALOG_REPO, CATALOG_TOKEN)
    print(f"Found {len(catalog_entries)} existing catalog entries")

    # Step 3: Load existing state
    print(f"\nLoading existing selected servers state...")
    existing_servers = load_selected_servers()
    print(f"Found {len(existing_servers)} previously processed servers")

    # Step 4: Filter and classify servers
    print("\nFiltering and classifying servers...")
    filtered_servers, non_active, likely_remote = filter_group_x_ai(servers, catalog_entries, existing_servers)

    # Step 5: Create issues for new servers
    print(f"\nProcessing {len(filtered_servers)} servers...")
    new_issues_created = 0

    for server in filtered_servers:
        issue_url, issue_number, issue_node_id = create_issue_for_server(server)
        add_issue_to_project(issue_node_id, issue_number, project_id, ISSUE_TOKEN)
        
        # Add as sub-issue if parent node ID was successfully fetched
        if parent_node_id and issue_node_id:
            add_sub_issue_graphql(parent_node_id, issue_node_id, ISSUE_TOKEN, INDEX_ISSUE_NUMBER, issue_number)
        
        existing_servers = add_server_to_state(server, issue_url, existing_servers)
        new_issues_created += 1


    # Step 6: Save updated state
    save_selected_servers(existing_servers)

    # Step 7: Summary
    print(f"\n=== Summary ===")
    print(f"New issues created: {new_issues_created}")
    print(f"Total tracked servers: {len(existing_servers)}")
    print(f"Non-active servers: {len(non_active)}")
    print(f"Remote servers: {len(likely_remote)}")

if __name__ == "__main__":
    main()