"""Lazy Notes — a local, beginner-friendly notes summarizer.

Run it with:
    streamlit run app.py

The page talks to Ollama on this computer. It does not call a hosted AI API.
Prompts, limits, and the HTTP client live in summarizer.py.
"""

from __future__ import annotations

import streamlit as st

from summarizer import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    MAX_NOTES_CHARS,
    SAMPLE_MEETING_NOTES,
    WARN_NOTES_CHARS,
    OllamaError,
    OllamaStatus,
    Summary,
    check_ollama,
    classify_endpoint,
    default_host,
    default_model,
    endpoint_is_loopback,
    format_markdown,
    format_plain,
    model_installed,
    normalize_host,
    prepare_notes,
    read_text_file,
    summarize,
    validate_model_name,
)

OLLAMA_DOWNLOAD_URL = "https://ollama.com/download"
STATUS_CACHE_SECONDS = 20

APP_CSS = """
<style>
  .block-container {
    padding-top: 1.4rem;
    padding-bottom: 3rem;
    max-width: 1140px;
  }
  .ln-hero {
    display: flex;
    gap: 0.85rem;
    align-items: flex-start;
    margin: 0 0 0.35rem;
  }
  .ln-mark {
    flex: 0 0 auto;
    margin-top: 0.15rem;
  }
  .ln-kicker {
    margin: 0 0 0.28rem;
    color: #2f6f5e;
    font-size: 0.76rem;
    font-weight: 650;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .ln-hero h1 {
    font-family: Georgia, "Iowan Old Style", "Palatino Linotype", Palatino, serif;
    font-size: 2.45rem;
    font-weight: 600;
    letter-spacing: -0.03em;
    line-height: 1;
    margin: 0 0 0.35rem;
    color: #1c2430;
  }
  .ln-tag {
    margin: 0;
    color: #52615c;
    font-size: 1.08rem;
  }
  div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #fffdf8;
    border-color: #e3dbcf !important;
    border-radius: 14px;
  }
  .stButton > button {
    border-radius: 10px;
  }
  div[data-testid="stTextArea"] textarea {
    line-height: 1.5;
  }
  footer { visibility: hidden; height: 0; }
  .stAppDeployButton { display: none; }
  @media (max-width: 640px) {
    .ln-hero h1 { font-size: 2rem; }
  }
</style>
"""

LENGTH_HELP = {
    "Short": "Two or three sentences.",
    "Medium": "A single paragraph.",
    "Detailed": "A fuller recap of each main point.",
}


def running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def init_state() -> None:
    host, host_warning = default_host()
    defaults = {
        "notes": "",
        "result": None,
        "error": None,
        "upload_error": None,
        "file_token": None,
        "loaded_name": "",
        "uploader_key": 0,
        "length": "Medium",
        "model": default_model(),
        "host": host,
        "host_warning": host_warning,
        "allow_remote": False,
        "status_cache": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main() -> None:
    st.set_page_config(
        page_title="Lazy Notes",
        page_icon="✎",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()
    st.markdown(APP_CSS, unsafe_allow_html=True)
    render_sidebar()
    render_header()
    status = current_status()
    render_status_banner(status)
    left, right = st.columns([1, 1], gap="large")
    with left:
        render_input(status)
    with right:
        render_output()


def render_header() -> None:
    st.markdown(
        """
        <div class="ln-hero">
          <div class="ln-mark" aria-hidden="true">
            <svg width="34" height="34" viewBox="0 0 34 34" fill="none">
              <rect x="4" y="3.5" width="18" height="24" rx="3" stroke="#2F6F5E" stroke-width="1.7"/>
              <path d="M8.5 10h9M8.5 15h9M8.5 20h6" stroke="#2F6F5E" stroke-width="1.7" stroke-linecap="round"/>
              <path d="M20 20.5l8 8" stroke="#B08968" stroke-width="1.8" stroke-linecap="round"/>
            </svg>
          </div>
          <div>
            <p class="ln-kicker">On your computer</p>
            <h1>Lazy Notes</h1>
            <p class="ln-tag">Your notes, summarized while you relax.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Paste a lecture, a meeting transcript, or another text document. "
        "A model running through Ollama on your machine writes the summary."
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Model")
        st.text_input(
            "Ollama model",
            key="model",
            help=f"Suggested default: {DEFAULT_MODEL}. Install it with ollama pull {DEFAULT_MODEL}.",
        )
        st.text_input(
            "Ollama address",
            key="host",
            help="The app sends notes only to this address. Leave the default to stay on this computer.",
        )
        if st.session_state.host_warning:
            st.warning(st.session_state.host_warning)
        _forget_stale_result("seen_model", st.session_state.model.strip())
        _render_host_choice()

        st.divider()
        st.header("Connection")
        if st.button("Recheck connection", use_container_width=True):
            st.session_state.status_cache = None
        status = current_status(force=False)
        if status.state == "reachable":
            if status.models:
                st.caption("Installed models: " + ", ".join(status.models))
            else:
                st.caption("Ollama is running and has no models installed yet.")
        elif status.state == "unreachable":
            st.caption("Not connected yet. The start command is on the main page.")
        elif status.message:
            st.caption(status.message)

        st.divider()
        st.header("Privacy")
        st.markdown(privacy_text(st.session_state.host))
        with st.expander("How to set up Ollama"):
            render_setup_steps()


def render_input(status: OllamaStatus) -> None:
    st.subheader("Your notes")
    load_col, clear_col = st.columns(2)
    if load_col.button(
        "Load sample meeting notes",
        use_container_width=True,
        help="Replaces the text in the box with a sample meeting. It does not generate a summary.",
    ):
        st.session_state.notes = SAMPLE_MEETING_NOTES.strip()
        st.session_state.loaded_name = "the sample meeting notes"
        st.session_state.upload_error = None
        st.session_state.error = None
        st.session_state.result = None
        _reset_uploader()
        st.rerun()
    if clear_col.button("Clear", use_container_width=True):
        st.session_state.notes = ""
        st.session_state.loaded_name = ""
        st.session_state.upload_error = None
        st.session_state.error = None
        st.session_state.result = None
        _reset_uploader()
        st.rerun()

    uploaded = st.file_uploader(
        "Or upload a .txt or .md file",
        key=f"upload_{st.session_state.uploader_key}",
        help="PDF, Word, and image files are not supported.",
    )
    _consume_upload(uploaded)
    if st.session_state.upload_error:
        st.error(st.session_state.upload_error)
    elif st.session_state.loaded_name:
        st.caption(
            f"Loaded {st.session_state.loaded_name}. You can edit the text before generating."
        )

    st.text_area(
        "Notes",
        key="notes",
        height=360,
        placeholder="Paste lecture notes, a transcript, or any text you do not want to read twice.",
    )
    _render_count(len(st.session_state.notes))
    st.radio(
        "Summary length",
        list(LENGTH_HELP),
        key="length",
        horizontal=True,
    )
    st.caption(LENGTH_HELP.get(st.session_state.length, ""))

    if st.button("Generate summary", type="primary", use_container_width=True):
        with st.spinner("Summarizing locally. The first run can take a minute while the model loads."):
            _generate()
        st.session_state.status_cache = None
        st.rerun()

    st.caption(
        "Generate sends the notes in the box only to the Ollama address in the sidebar. "
        "Nothing is saved on a server by this app."
    )
    if status.state != "reachable":
        st.caption("You can read and edit the sample without Ollama. A summary waits for the model.")


def render_output() -> None:
    error = st.session_state.error
    result = st.session_state.result
    if error is not None:
        _show_error(error)
    if result is None:
        if error is None:
            _render_empty()
        else:
            st.caption("No summary was created.")
        return
    _render_summary(result)


def render_status_banner(status: OllamaStatus) -> None:
    model = st.session_state.model.strip() or DEFAULT_MODEL
    if status.state == "reachable" and model_installed(status.models, model):
        st.caption(f"Local model ready · {model} at {status.host}")
        return
    if status.state == "reachable":
        st.warning(
            f"Ollama is running at {status.host}, but {model} is not installed. "
            "No summary will be invented in its place. Download the model, or type an installed model name in the sidebar."
        )
        st.code(f"ollama pull {model}", language="bash")
        if status.models:
            st.caption("Installed now: " + ", ".join(status.models))
        return
    if status.state == "needs_confirmation":
        st.warning(status.message)
        return
    if status.state == "invalid":
        st.error(status.message)
        return
    st.warning(status.message or "Ollama is not reachable.")
    st.code(f"ollama serve\nollama pull {model}", language="bash")


def render_setup_steps() -> None:
    st.markdown(
        f"1. Install Ollama from [{OLLAMA_DOWNLOAD_URL}]({OLLAMA_DOWNLOAD_URL}).\n"
        "2. Start it. On macOS and Windows, open the Ollama app. "
        "On Linux, run `ollama serve` if it is not already a system service.\n"
        f"3. Download the suggested model: `ollama pull {DEFAULT_MODEL}`.\n"
        "4. Click **Recheck connection**. You can type another installed model name above."
    )
    st.caption("Keep this app open in one terminal and Ollama in another if you start it by hand.")


def current_status(force: bool = False) -> OllamaStatus:
    import time

    host = st.session_state.host
    allow_remote = bool(st.session_state.allow_remote)
    cached = st.session_state.status_cache
    now = time.monotonic()
    if (
        not force
        and isinstance(cached, dict)
        and cached.get("host") == host
        and cached.get("allow_remote") == allow_remote
        and now - cached.get("at", 0) < STATUS_CACHE_SECONDS
    ):
        return cached["status"]
    status = check_ollama(host, allow_remote=allow_remote)
    st.session_state.status_cache = {
        "host": host,
        "allow_remote": allow_remote,
        "at": now,
        "status": status,
    }
    return status


def privacy_text(host: str) -> str:
    try:
        clean = normalize_host(host)
        kind = classify_endpoint(clean)
    except ValueError:
        return (
            "Enter a valid Ollama address. Notes are sent only to that address when you "
            "generate a summary, and only after it is accepted."
        )
    if kind == "loopback":
        where = "That address is on this computer."
    elif kind == "private":
        where = "That address is on a private network, so notes leave this computer when you generate."
    elif kind == "blocked":
        where = "That address is blocked, so notes will not be sent."
    else:
        where = "That address is not local, so notes leave this computer when you generate."
    scheme = clean.split("://", 1)[0]
    if scheme == "https":
        transport = "The request uses HTTPS."
    else:
        transport = (
            "The request uses plain HTTP, which is the usual way to reach Ollama on localhost. "
            "Lazy Notes does not add encryption of its own."
        )
    return (
        f"When you generate a summary, the notes in the text box are sent to {clean} "
        f"for inference. {where} {transport} "
        "They are not intentionally uploaded to OpenAI or any other hosted AI API, "
        "and this app does not add analytics. Anyone who can use that Ollama server can receive the request."
    )


def _generate() -> None:
    st.session_state.error = None
    try:
        notes = prepare_notes(st.session_state.notes)
        model = validate_model_name(st.session_state.model)
        host = normalize_host(st.session_state.host)
        allow_remote = bool(st.session_state.allow_remote) and not endpoint_is_loopback(host)
        _preflight(host, model, allow_remote)
        result = summarize(
            notes,
            st.session_state.length,
            model,
            host,
            allow_remote=allow_remote,
        )
    except ValueError as exc:
        st.session_state.error = OllamaError("config", str(exc))
    except OllamaError as exc:
        st.session_state.error = exc
    else:
        st.session_state.result = result
        st.session_state.error = None


def _preflight(host: str, model: str, allow_remote: bool) -> None:
    """Fail fast, before notes are sent, when Ollama or the model is missing."""
    status = check_ollama(host, allow_remote=allow_remote, timeout=2)
    if status.state == "reachable" and model_installed(status.models, model):
        return
    if status.state == "reachable":
        installed = ", ".join(status.models) if status.models else "none"
        raise OllamaError(
            "model_not_found",
            (
                f"The model {model} is not installed in Ollama "
                f"(installed: {installed}). No summary was invented in its place."
            ),
            hint_command=f"ollama pull {model}",
        )
    if status.state == "unreachable":
        raise OllamaError(
            "connection",
            status.message,
            hint_command=f"ollama serve\nollama pull {model}",
        )
    raise OllamaError("config", status.message or "Check the Ollama address in the sidebar.")


def _render_empty() -> None:
    with st.container(border=True):
        st.subheader("Summary")
        if st.session_state.error is None:
            st.markdown("Your summary will show up here after the local model answers.")
        st.markdown(
            "You will get four sections:\n"
            "- A two-line recap\n"
            "- A summary at the length you chose\n"
            "- Key takeaways\n"
            "- Action items, with **unspecified** wherever a name or deadline was not in the notes"
        )
        st.caption("Loading the sample only fills the text box. It does not pretend to be a summary.")


def _render_summary(result: Summary) -> None:
    with st.container(border=True):
        st.subheader("Two-minute recap")
        for line in result.recap.split("\n"):
            st.markdown(line)

    with st.container(border=True):
        st.subheader("Summary")
        st.caption(f"{result.length} · {result.model}")
        st.markdown(result.summary)

    with st.container(border=True):
        st.subheader("Key takeaways")
        if result.takeaways:
            for item in result.takeaways:
                st.markdown(f"- {item}")
            if result.takeaways_truncated:
                st.caption(f"Showing the first {len(result.takeaways)} takeaways the model returned.")
        else:
            st.markdown("The model did not list any key takeaways.")

    with st.container(border=True):
        st.subheader("Action items")
        if result.action_items:
            for index, item in enumerate(result.action_items, start=1):
                st.markdown(f"**{index}.** {item.task}")
                st.caption(f"Owner: {item.owner}  ·  Deadline: {item.deadline}")
            if result.actions_truncated:
                st.caption(f"Showing the first {len(result.action_items)} action items.")
            st.caption(
                "Owners and deadlines stay only when those words appear in the notes. "
                "Otherwise they are marked unspecified."
            )
        else:
            st.markdown(
                "No supported action items were found. Ideas and undecided topics are left off "
                "this list, and missing names or dates are not filled in."
            )

    markdown = format_markdown(result)
    plain = format_plain(result)
    download_md, download_txt = st.columns(2)
    download_md.download_button(
        "Download .md",
        data=markdown.encode("utf-8"),
        file_name="lazy-notes-summary.md",
        mime="text/markdown",
        use_container_width=True,
    )
    download_txt.download_button(
        "Download .txt",
        data=plain.encode("utf-8"),
        file_name="lazy-notes-summary.txt",
        mime="text/plain",
        use_container_width=True,
    )
    with st.expander("Copy the export"):
        st.caption("Use the copy icon on this box, or download the file.")
        st.code(markdown, language="markdown")


def _show_error(error: OllamaError) -> None:
    st.error(error.message)
    if error.hint_command:
        st.code(error.hint_command, language="bash")
    if error.detail:
        label = (
            "What the model returned (not used as your summary)"
            if error.code == "invalid_response"
            else "Technical detail"
        )
        with st.expander(label):
            st.code(error.detail)


def _consume_upload(uploaded) -> None:
    if uploaded is None:
        return
    token = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.file_token == token:
        return
    st.session_state.file_token = token
    try:
        text = read_text_file(uploaded.name, uploaded.getvalue())
    except ValueError as exc:
        st.session_state.upload_error = str(exc)
        return
    st.session_state.notes = text
    st.session_state.loaded_name = uploaded.name
    st.session_state.upload_error = None
    st.session_state.error = None
    st.session_state.result = None
    st.rerun()


def _reset_uploader() -> None:
    st.session_state.uploader_key += 1
    st.session_state.file_token = None


def _render_count(count: int) -> None:
    if count > MAX_NOTES_CHARS:
        st.warning(
            f"{count:,} characters. The limit is {MAX_NOTES_CHARS:,}. "
            "Shorten the notes or split them into parts."
        )
    elif count > WARN_NOTES_CHARS:
        st.caption(f"{count:,} characters. Long notes take longer and use more memory.")
    else:
        st.caption(f"{count:,} characters · up to {MAX_NOTES_CHARS:,}")


def _forget_stale_result(slot: str, value: str) -> None:
    previous = st.session_state.get(slot)
    if previous not in (None, value):
        st.session_state.error = None
    st.session_state[slot] = value


def _render_host_choice() -> None:
    """Explain where notes will go. Confirmation is reset when the address changes."""
    try:
        clean = normalize_host(st.session_state.host)
        kind = classify_endpoint(clean)
    except ValueError as exc:
        st.error(str(exc))
        return
    if st.session_state.get("confirmed_host") not in (None, clean):
        # A failure from the previous address should not sit next to the new one.
        st.session_state.error = None
        st.session_state.status_cache = None
    if st.session_state.get("confirmed_host") != clean and kind in {"private", "public"}:
        st.session_state.allow_remote = False
    st.session_state.confirmed_host = clean
    if kind == "loopback":
        st.caption("This address stays on this computer.")
        return
    if kind == "blocked":
        st.error(f"Lazy Notes will not contact {clean}. Use {DEFAULT_HOST}.")
        return
    st.checkbox(
        "Send notes to this non-local Ollama address",
        key="allow_remote",
        help="Leave this off unless you intend to send notes to another machine.",
    )
    if kind == "private":
        st.caption(f"{clean} is on a private network. Notes leave this computer if you generate.")
    else:
        st.caption(f"{clean} is not a local address. Notes leave this computer if you generate.")


if __name__ == "__main__":
    if running_in_streamlit():
        main()
    else:
        print("Start Lazy Notes with:\n  streamlit run app.py")
else:
    # `streamlit run` executes this file as __main__. The branch above handles
    # that. Importing the module (for example from a test) should not start the UI.
    pass
