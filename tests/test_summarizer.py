"""Tests for Lazy Notes that do not need a running model."""

from __future__ import annotations

import http.server
import json
import socket
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import summarizer
from summarizer import (
    DEFAULT_HOST,
    SAMPLE_MEETING_NOTES,
    OllamaError,
    check_ollama,
    choose_num_ctx,
    classify_endpoint,
    extract_json_object,
    format_markdown,
    format_plain,
    model_installed,
    normalize_host,
    normalize_summary,
    prepare_notes,
    read_text_file,
    summarize,
    user_prompt,
)


NOTES = SAMPLE_MEETING_NOTES.strip()


class HostRulesTest(unittest.TestCase):
    def test_normalize_host(self):
        self.assertEqual(normalize_host("127.0.0.1:11434"), "http://127.0.0.1:11434")
        self.assertEqual(normalize_host("http://127.0.0.1:11434/"), DEFAULT_HOST)
        self.assertEqual(normalize_host("http://127.0.0.1:11434/api"), DEFAULT_HOST)
        self.assertEqual(normalize_host("https://localhost:11434"), "https://localhost:11434")
        self.assertEqual(normalize_host("http://[::1]:11434"), "http://[::1]:11434")

    def test_normalize_rejects_bad_addresses(self):
        for raw in ("", "ftp://127.0.0.1", "http://user:pass@127.0.0.1:11434", "http:///missing"):
            with self.assertRaises(ValueError):
                normalize_host(raw)

    def test_classify_endpoint(self):
        self.assertEqual(classify_endpoint("http://127.0.0.1:11434"), "loopback")
        self.assertEqual(classify_endpoint("http://localhost:11434"), "loopback")
        self.assertEqual(classify_endpoint("http://[::1]:11434"), "loopback")
        self.assertEqual(classify_endpoint("http://192.168.1.20:11434"), "private")
        self.assertEqual(classify_endpoint("http://10.1.2.3:11434"), "private")
        self.assertEqual(classify_endpoint("http://8.8.8.8:11434"), "public")
        self.assertEqual(classify_endpoint("http://169.254.169.254:80"), "blocked")
        self.assertEqual(classify_endpoint("http://0.0.0.0:11434"), "blocked")

    def test_model_name_matching_does_not_confuse_prefixes(self):
        installed = ["llama3.2:latest", "mistral:latest"]
        self.assertTrue(model_installed(installed, "llama3.2"))
        self.assertTrue(model_installed(installed, "llama3.2:latest"))
        self.assertTrue(model_installed(installed, "mistral"))
        self.assertFalse(model_installed(installed, "llama3"))
        self.assertFalse(model_installed(installed, "llama3.2:1b"))
        self.assertFalse(model_installed([], "llama3.2"))


class NotesInputTest(unittest.TestCase):
    def test_prepare_notes(self):
        self.assertEqual(prepare_notes("  hello\r\nthere  "), "hello\nthere")
        with self.assertRaises(ValueError):
            prepare_notes("   \n")
        with self.assertRaises(ValueError):
            prepare_notes("a" * (summarizer.MAX_NOTES_CHARS + 1))

    def test_read_text_file(self):
        self.assertEqual(read_text_file("notes.md", b"# Title\n"), "# Title\n")
        self.assertEqual(read_text_file("NOTES.TXT", "café".encode("utf-8")), "café")
        self.assertEqual(read_text_file("bom.txt", "hi".encode("utf-8-sig")), "hi")
        with self.assertRaises(ValueError) as caught:
            read_text_file("slides.pdf", b"%PDF")
        self.assertIn(".txt", str(caught.exception))
        with self.assertRaises(ValueError):
            read_text_file("empty.txt", b"")
        with self.assertRaises(ValueError):
            read_text_file("binary.txt", b"a\x00b")
        with self.assertRaises(ValueError):
            read_text_file("latin.txt", "é".encode("latin-1"))

    def test_sample_file_matches_the_in_app_notes(self):
        path = ROOT / "examples" / "sample-meeting.md"
        self.assertEqual(path.read_text(encoding="utf-8").strip(), NOTES)
        self.assertIn("Luis Ortega", NOTES)
        self.assertIn("no deadline", NOTES.lower())

    def test_context_window_fits_the_character_cap(self):
        notes = "word " * (summarizer.MAX_NOTES_CHARS // 5)
        notes = notes[: summarizer.MAX_NOTES_CHARS]
        predict = int(summarizer.LENGTHS["Detailed"]["num_predict"])
        self.assertLessEqual(choose_num_ctx(notes, predict), summarizer.MAX_CONTEXT)
        self.assertEqual(choose_num_ctx("short note", 700), 4096)


class GroundingTest(unittest.TestCase):
    def _summary(self, payload):
        return normalize_summary(payload, NOTES, "llama3.2", "Medium")

    def test_keeps_supported_facts_and_drops_inventions(self):
        summary = self._summary(
            {
                "summary": "The reading-list bug comes first.",
                "takeaways": ["Saved lists disappear after refresh.", "   ", {"text": "Export work waits."}],
                "action_items": [
                    {
                        "task": "Patch the save handler",
                        "owner": "Luis Ortega",
                        "deadline": "Thursday, 19 March",
                    },
                    {
                        "task": "Review the empty-state layout",
                        "owner": "Amira Khan",
                        "deadline": "Friday, 2 April",
                    },
                    {
                        "task": "Do the spike",
                        "owner": "someone",
                        "deadline": "next week",
                    },
                    {
                        "task": "Order catering platters",
                        "owner": "Sarah Chen",
                        "deadline": "tomorrow",
                    },
                    {
                        "task": "Patch the save handler",
                        "owner": "Luis Ortega",
                        "deadline": "Thursday, 19 March",
                    },
                ],
                "recap": "Lists disappear after refresh.\nLuis will patch the save handler.\nExtra line.",
            }
        )
        self.assertEqual(
            [(item.task, item.owner, item.deadline) for item in summary.action_items],
            [
                ("Patch the save handler", "Luis Ortega", "Thursday, 19 March"),
                ("Review the empty-state layout", "Amira Khan", "unspecified"),
                ("Do the spike", "unspecified", "next week"),
            ],
        )
        self.assertEqual(summary.takeaways, ["Saved lists disappear after refresh.", "Export work waits."])
        self.assertEqual(summary.recap, "Lists disappear after refresh.\nLuis will patch the save handler.")

    def test_month_abbreviation_must_still_appear(self):
        summary = self._summary(
            {
                "summary": "A short summary.",
                "takeaways": [],
                "action_items": [
                    {"task": "Patch the save handler", "owner": "Luis", "deadline": "19 Mar"},
                    {"task": "Patch the save handler again", "owner": "Luis", "deadline": "19 March 2099"},
                ],
                "recap": ["The bug is in the save handler.", "Luis will patch it."],
            }
        )
        self.assertEqual(summary.action_items[0].deadline, "19 Mar")
        self.assertEqual(summary.action_items[1].deadline, "unspecified")
        self.assertEqual(summary.recap, "The bug is in the save handler.\nLuis will patch it.")

    def test_owner_must_be_a_name_from_the_notes(self):
        summary = self._summary(
            {
                "summary": "A short summary.",
                "takeaways": "Ship the fix.\nLeave export for later.",
                "action_items": {
                    "task": "Patch the save handler",
                    "owner": "Priya Shah and a stranger",
                    "deadline": "unspecified",
                },
                "recap": "One sentence only.",
            }
        )
        self.assertEqual(summary.action_items[0].owner, "unspecified")
        self.assertEqual(summary.takeaways, ["Ship the fix.", "Leave export for later."])
        self.assertEqual(summary.recap, "One sentence only.")

    def test_placeholder_recap_is_rejected(self):
        with self.assertRaises(OllamaError) as caught:
            self._summary(
                {
                    "summary": "The save handler needs a patch.",
                    "takeaways": ["The save handler has a bug."],
                    "action_items": [],
                    "recap": "First line.\nSecond line.",
                }
            )
        self.assertEqual(caught.exception.code, "invalid_response")

    def test_missing_summary_is_rejected(self):
        with self.assertRaises(OllamaError) as caught:
            self._summary({"summary": "  ", "recap": "Line one.\nLine two."})
        self.assertEqual(caught.exception.code, "invalid_response")

    def test_takeaway_cap_is_reported(self):
        summary = self._summary(
            {
                "summary": "A short summary.",
                "takeaways": [f"Point {index} about refresh" for index in range(18)],
                "action_items": [],
                "recap": "First line. Second line stays with it.",
            }
        )
        self.assertEqual(len(summary.takeaways), summarizer.MAX_TAKEAWAYS)
        self.assertTrue(summary.takeaways_truncated)
        self.assertEqual(summary.recap.split("\n")[0], "First line.")


class ParseAndExportTest(unittest.TestCase):
    def test_extract_json_from_fenced_and_prose(self):
        payload = extract_json_object(
            '```json\n{"summary":"Use {curly} braces.","takeaways":[],"action_items":[],"recap":"A.\\nB."}\n```'
        )
        self.assertEqual(payload["summary"], "Use {curly} braces.")
        prose = 'Sure. {"summary":"Hi","takeaways":[],"action_items":[],"recap":"A.\\nB."} thanks'
        self.assertEqual(extract_json_object(prose)["summary"], "Hi")
        with self.assertRaises(OllamaError):
            extract_json_object("no json here")

    def test_prompt_includes_the_notes_and_the_length(self):
        prompt = user_prompt("The aqueduct notes.", "Short")
        self.assertIn("The aqueduct notes.", prompt)
        self.assertIn("Summary length: Short", prompt)
        self.assertIn("Never invent", summarizer.SYSTEM_PROMPT)

    def test_exports_include_each_section(self):
        summary = normalize_summary(
            {
                "summary": "Ship the reading-list fix first.",
                "takeaways": [],
                "action_items": [
                    {"task": "Patch the save handler", "owner": "", "deadline": "Thursday, 19 March"}
                ],
                "recap": "The list bug is first.\nExport work waits.",
            },
            NOTES,
            "llama3.2",
            "Short",
        )
        for rendered in (format_markdown(summary), format_plain(summary)):
            self.assertIn("Two-minute recap", rendered)
            self.assertIn("Ship the reading-list fix first.", rendered)
            self.assertIn("Patch the save handler", rendered)
            self.assertIn("unspecified", rendered)
            self.assertIn("Thursday, 19 March", rendered)
            self.assertNotIn("Sarah", rendered)


class OllamaClientTest(unittest.TestCase):
    def test_summarize_posts_notes_only_to_the_configured_host(self):
        seen = {}

        def respond(method, path, raw):
            seen["method"] = method
            seen["path"] = path
            seen["body"] = json.loads(raw.decode("utf-8"))
            content = json.dumps(
                {
                    "summary": "The saved list disappears after refresh.",
                    "takeaways": ["Fix the save handler first."],
                    "action_items": [],
                    "recap": "Lists disappear after refresh.\nThe fix comes before export work.",
                }
            )
            return 200, json.dumps({"message": {"role": "assistant", "content": content}}), []

        with serve(respond) as host:
            summary = summarize(NOTES, "Medium", "llama3.2", host, timeout=5)
        self.assertEqual(seen["method"], "POST")
        self.assertEqual(seen["path"], "/api/chat")
        self.assertEqual(seen["body"]["model"], "llama3.2")
        self.assertFalse(seen["body"]["stream"])
        self.assertEqual(seen["body"]["format"], "json")
        self.assertIn(NOTES, seen["body"]["messages"][1]["content"])
        self.assertEqual(summary.summary, "The saved list disappears after refresh.")
        self.assertEqual(summary.model, "llama3.2")

    def test_model_not_found(self):
        def respond(method, path, raw):
            return 404, json.dumps({"error": "model 'llama3.2' not found"}), []

        with serve(respond) as host:
            with self.assertRaises(OllamaError) as caught:
                summarize("Notes about the save handler.", "Short", "llama3.2", host, timeout=5)
        self.assertEqual(caught.exception.code, "model_not_found")
        self.assertIn("ollama pull llama3.2", caught.exception.hint_command)

    def test_invalid_model_text_is_not_turned_into_a_summary(self):
        def respond(method, path, raw):
            return 200, json.dumps({"message": {"content": "Here is a casual reply."}}), []

        with serve(respond) as host:
            with self.assertRaises(OllamaError) as caught:
                summarize("Notes about the save handler.", "Short", "llama3.2", host, timeout=5)
        self.assertEqual(caught.exception.code, "invalid_response")
        self.assertIn("casual reply", caught.exception.detail)

    def test_redirect_is_not_followed(self):
        def respond(method, path, raw):
            return 302, "", [("Location", "http://example.com/collect")]

        with serve(respond) as host:
            with self.assertRaises(OllamaError) as caught:
                summarize("Notes about the save handler.", "Short", "llama3.2", host, timeout=5)
        self.assertEqual(caught.exception.code, "http")
        self.assertIn("redirect", caught.exception.message.lower())

    def test_timeout(self):
        def respond(method, path, raw):
            time.sleep(1.2)
            return 200, "{}", []

        with serve(respond) as host:
            started = time.monotonic()
            with self.assertRaises(OllamaError) as caught:
                summarize("Notes about the save handler.", "Short", "llama3.2", host, timeout=0.4)
        self.assertEqual(caught.exception.code, "timeout")
        self.assertLess(time.monotonic() - started, 1.1)

    def test_connection_refused(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        with self.assertRaises(OllamaError) as caught:
            summarize(
                "Notes about the save handler.",
                "Short",
                "llama3.2",
                f"http://127.0.0.1:{port}",
                timeout=2,
            )
        self.assertEqual(caught.exception.code, "connection")
        self.assertIn("ollama serve", caught.exception.hint_command)

    def test_refuses_non_local_hosts_without_sending(self):
        def fail_if_called(*args, **kwargs):
            raise AssertionError(f"unexpected request: {args} {kwargs}")

        with mock.patch("summarizer.http_request", side_effect=fail_if_called):
            for host in ("http://8.8.8.8:11434", "http://192.168.0.8:11434", "http://169.254.169.254"):
                with self.assertRaises(OllamaError) as caught:
                    summarize("Notes about the save handler.", "Short", "llama3.2", host)
                self.assertEqual(caught.exception.code, "config")
            status = check_ollama("https://example.com:11434", allow_remote=False)
        self.assertEqual(status.state, "needs_confirmation")

    def test_tags_request_carries_no_notes(self):
        seen = {}

        def respond(method, path, raw):
            seen["method"] = method
            seen["path"] = path
            seen["body"] = raw
            return 200, json.dumps({"models": [{"name": "llama3.2:latest"}]}), []

        with serve(respond) as host:
            status = check_ollama(host, timeout=5)
        self.assertEqual(status.state, "reachable")
        self.assertEqual(seen["method"], "GET")
        self.assertEqual(seen["path"], "/api/tags")
        self.assertEqual(seen["body"], b"")
        self.assertEqual(status.models, ["llama3.2:latest"])

    def test_source_does_not_name_a_hosted_ai_endpoint(self):
        for name in ("app.py", "summarizer.py", "requirements.txt"):
            text = (ROOT / name).read_text(encoding="utf-8").lower()
            for banned in (
                "api.openai.com",
                "anthropic.com",
                "generativelanguage.googleapis.com",
            ):
                self.assertNotIn(banned, text)


def serve(responder):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            self._reply()

        def do_POST(self):
            self._reply()

        def _reply(self):
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length else b""
            status, body, extra = responder(self.command, self.path, raw)
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for key, value in extra:
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

    class Quiet(http.server.ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = Quiet(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = f"http://127.0.0.1:{server.server_address[1]}"

    class _Running:
        def __enter__(self):
            return host

        def __exit__(self, exc_type, exc, tb):
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    return _Running()


if __name__ == "__main__":
    unittest.main()
