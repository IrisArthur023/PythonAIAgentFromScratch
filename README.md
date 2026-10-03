# Field Notes research assistant

1. Copy `sample.env` to `.env` and add your Anthropic API key.
2. Double-click `run.bat` to create a private Python environment, install both backend and Next.js dependencies on first run, start the app, and open it automatically.

The ChatGPT-style Next.js frontend runs at `http://localhost:3000` and forwards API requests to the Python backend on port 8000.

To work on the frontend directly, run `corepack pnpm dev` from `frontend/` while `python main.py` is running in the project root.

## Build modes

- **Research** searches the web and Wikipedia, then saves completed briefs to local memory.
- **Plan** turns an app idea into ordered milestones, dependencies, and acceptance criteria.
- **Code** drafts focused, file-specific implementation guidance without modifying files.
- **Debug** analyzes error messages and relevant code, then suggests the smallest repair.

Plan, Code, and Debug can include a bounded, read-only snapshot of the backend and Next.js
frontend. Secrets and `.env` files are never included.

## Research memory

Completed briefs are stored locally in Chroma at `chroma_db/`. When you research a
new topic, the app retrieves up to three related briefs and provides them to the
agent as reference material. Delete that folder to clear the saved research memory.
