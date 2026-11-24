"""
Run this ONCE to log in to Google Drive. It opens the browser and saves secrets/token.json.
(The MCP server can't do this itself: it talks over stdin/stdout, so it must not print or open a browser.)
"""
from google_auth_oauthlib.flow import InstalledAppFlow

from gdrive import CREDENTIALS_FILE, SCOPES, TOKEN_FILE, search_files

flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
creds = flow.run_local_server(port=0)
TOKEN_FILE.write_text(creds.to_json())
print(f"✅ Logged in. Token saved to {TOKEN_FILE}\n")
print("Test search (latest files with 'a' in the name):")
print(search_files("a", max_results=5))