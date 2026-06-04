import pytest
from httpx import AsyncClient, ASGITransport
import importlib.util
import sys
import os

# Load the actual app from modules/api/script.py
spec = importlib.util.spec_from_file_location("script", os.path.join(os.path.dirname(__file__), "modules/api/script.py"))
module = importlib.util.load_from_spec(spec)
spec.loader.exec_module(module)
app = module.app


@pytest.mark.parametrize("filename", [
    "../../../etc/passwd",          # classic path traversal
    "....//....//etc/passwd",       # double-dot obfuscation
    "%2e%2e%2fetc%2fpasswd",        # URL-encoded traversal
    "audio.wav",                    # valid input (should succeed or fail gracefully)
])
@pytest.mark.asyncio
async def test_audio_transcription_path_traversal(filename):
    """Invariant: The transcription endpoint must reject or safely handle filenames
    containing path traversal sequences, never resolving outside the allowed root."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        fake_audio = b"RIFF\x00\x00\x00\x00WAVEfmt "
        files = {"file": (filename, fake_audio, "audio/wav")}
        response = await client.post("/v1/audio/transcriptions", files=files)

        # Traversal payloads must never result in a 200 that leaks file contents
        if filename != "audio.wav":
            assert response.status_code in (400, 401, 403, 404, 422, 500), (
                f"Path traversal payload '{filename}' was not rejected: "
                f"status={response.status_code}, body={response.text[:200]}"
            )
            # Ensure /etc/passwd contents are not reflected in the response
            assert "root:" not in response.text, (
                f"Possible path traversal: /etc/passwd content leaked for filename='{filename}'"
            )