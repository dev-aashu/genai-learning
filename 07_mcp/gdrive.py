"""
Google Drive helpers used by the MCP server.
Read-only access. Login happens once via gdrive_auth.py, which saves secrets/token.json.
"""
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

ROOT = Path(__file__).resolve().parents[1]
CREDENTIALS_FILE = ROOT / "secrets" / "credentials.json"
TOKEN_FILE = ROOT / "secrets" / "token.json"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
MAX_CHARS = 5000


def get_drive_service():
    """Load saved login (token.json), refresh it if expired, and return a Drive API client."""
    if not TOKEN_FILE.exists():
        raise RuntimeError("Not logged in to Google Drive. Run: python 07_mcp/gdrive_auth.py")
    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            TOKEN_FILE.write_text(creds.to_json())
        else:
            raise RuntimeError("Google login expired. Run: python 07_mcp/gdrive_auth.py")
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def search_files(query: str, max_results: int = 10) -> str:
    service = get_drive_service()
    safe = query.replace("\\", "\\\\").replace("'", "\\'")
    result = service.files().list(
        q=f"name contains '{safe}' and trashed = false",
        pageSize=max_results,
        fields="files(id, name, mimeType, modifiedTime)",
        orderBy="modifiedTime desc",
    ).execute()
    files = result.get("files", [])
    if not files:
        return f"No files found matching '{query}'."
    lines = [f"- {f['name']} (id: {f['id']}, type: {f['mimeType']}, modified: {f['modifiedTime'][:10]})"
             for f in files]
    return "\n".join(lines)


def read_file(file_id: str) -> str:
    service = get_drive_service()
    meta = service.files().get(fileId=file_id, fields="name, mimeType").execute()
    mime = meta["mimeType"]

    if mime == "application/vnd.google-apps.document":          # Google Doc
        data = service.files().export(fileId=file_id, mimeType="text/plain").execute()
    elif mime == "application/vnd.google-apps.spreadsheet":     # Google Sheet (first sheet)
        data = service.files().export(fileId=file_id, mimeType="text/csv").execute()
    elif mime.startswith("text/"):                               # .txt, .md, .csv files
        data = service.files().get_media(fileId=file_id).execute()
    else:
        return f"'{meta['name']}' is a {mime} file. Only Google Docs, Sheets and text files are supported."

    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n...[truncated]"
    return f"File: {meta['name']}\n\n{text}"