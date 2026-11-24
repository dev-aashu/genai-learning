# 11: Google Drive tools via MCP (local)

Added two read-only Google Drive tools to the MCP server (07_mcp/server.py):

- `search_drive_files(query)`: find files by name
- `read_drive_file(file_id)`: read Google Docs / Sheets / text files

No change to the agent or the app: they discovered the new tools automatically.

## Setup

1. Google Cloud: enable Drive API, OAuth consent (External, test user), Desktop client
2. Save the client JSON as `secrets/credentials.json` (gitignored)
3. Log in once: `cd 07_mcp && python gdrive_auth.py` (saves `secrets/token.json`)

## Security

- Scope: `drive.readonly`, so the agent can only read files
- `secrets/` is in .gitignore; credentials and token are never committed
