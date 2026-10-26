# Lazy Notes

Your notes, summarized while you relax.

Lazy Notes is a small web app for the Hacktoberfest “Slow and Steady” challenge. Paste lecture notes, a meeting transcript, or a `.txt` / `.md` file. A model running **on your computer** through [Ollama](https://ollama.com) returns:

- a two-line recap
- a short, medium, or detailed summary
- key takeaways
- action items, with the owner and deadline marked **unspecified** when the notes do not state them

The app does not call OpenAI or any other hosted AI API, and it does not ask for an API key. If Ollama is not running, you can still open the app and load the sample meeting. A summary is created only when the local model answers. Nothing is filled in with a canned summary.

Notes stay in the browser tab until you refresh. This app does not write them to a database.

## Privacy

When you click **Generate summary**, the text in the notes box is sent only to the Ollama address in the sidebar. The default is `http://127.0.0.1:11434` on this computer.

That request is plain HTTP, which is how Ollama works on localhost. Lazy Notes does not add its own encryption, analytics, or an account system. Anyone who can use that Ollama server can receive the notes. If you change the address to another machine, the notes go there, and the app asks you to confirm first. Redirects are not followed.

## What you need

- Python 3.10 or newer
- Ollama, for actual summaries
- A model such as `llama3.2` (the suggested default)

The first summary can take a minute while the model loads into memory. Later ones are faster. A GPU is optional.

## 1. Install the Python dependencies

macOS and Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Windows (Command Prompt) uses `.venv\Scripts\activate.bat` instead of the PowerShell line.

## 2. Install Ollama

Download the app from <https://ollama.com/download>.

- **macOS and Windows:** open the Ollama app. It starts the local server for you.
- **Linux:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

On Linux, `ollama serve` keeps running in that terminal. Use a second terminal for the next commands. Some installers also start a background service, in which case `ollama serve` is unnecessary.

## 3. Download the model

```bash
ollama pull llama3.2
```

A different installed model works too. Type its name in the sidebar, or set it before starting the app:

```bash
export OLLAMA_MODEL=llama3.2
export OLLAMA_HOST=http://127.0.0.1:11434
```

`OLLAMA_HOST` is the address **this app calls**. Use `127.0.0.1` even if the Ollama server listens on `0.0.0.0`. Windows PowerShell uses `$env:OLLAMA_MODEL = "llama3.2"` and `$env:OLLAMA_HOST = "http://127.0.0.1:11434"`.

## 4. Run Lazy Notes

From the project folder, with the virtual environment activated:

```bash
streamlit run app.py
```

Open <http://localhost:47231>.

The committed Streamlit config chooses port **47231** so it is less likely to collide with another app. If that port is taken:

```bash
streamlit run app.py --server.port 47232
```

`python app.py` only prints the command above. Streamlit has to run the file.

## Use the app

1. Paste notes, upload a `.txt` or `.md` file, or click **Load sample meeting notes**.
2. Choose Short, Medium, or Detailed.
3. Click **Generate summary**.
4. Read the four sections. Copy the export with the icon on the code box, or download `lazy-notes-summary.md` / `lazy-notes-summary.txt`.

The sample button only fills the text box. The same text is in [`examples/sample-meeting.md`](examples/sample-meeting.md) if you want to try the upload.

PDF, Word, and image files are refused. Notes longer than 36,000 characters are refused so a laptop-sized context window can see the whole document. Split a longer file and summarize it in parts.

## Where to look in the code

| File | What it does |
| --- | --- |
| `app.py` | The Streamlit page |
| `summarizer.py` | Prompts, the Ollama client, and the rule that drops unsupported owners, deadlines, and tasks |
| `examples/sample-meeting.md` | The demo meeting |
| `tests/test_summarizer.py` | Tests that do not need a model |

The system prompt is `SYSTEM_PROMPT` in `summarizer.py`.

## Tests

With the virtual environment activated:

```bash
python -m unittest tests.test_summarizer -v
```

These tests check parsing, the “do not invent names or dates” rules, file types, and the HTTP client against a tiny local stand-in. They do not download or run `llama3.2`.

## Troubleshooting

**“Could not reach Ollama”**  
Start the Ollama app, or run `ollama serve`, then click **Recheck connection**.

**“The model … is not installed”**

```bash
ollama pull llama3.2
```

**The model is slow or the request times out**  
Choose Short, or run `ollama run llama3.2` once in a terminal so the model is loaded. Then generate again.

**The reply was not valid JSON**  
No summary is shown. Click **Generate summary** again, or switch to Short. The raw reply, if any, stays in an expander and is not treated as the summary.

**Port already in use**  
Run `streamlit run app.py --server.port 47232`.

## Limits

- Summaries depend on the local model. A small model can miss a task or phrase one awkwardly. Owners and deadlines that are not written in the notes are forced to `unspecified`, and tasks that do not share words with the notes are dropped. That reduces invented names and dates. It cannot catch every paraphrase.
- The app does not read PDF or Word files.
- Authenticated proxies are not supported. The address cannot include a username or password.
