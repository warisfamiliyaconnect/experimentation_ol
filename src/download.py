import argparse
import os
import sys
import time
import requests
from urllib3.exceptions import InsecureRequestWarning
from huggingface_hub import HfApi, hf_hub_download
from src.utils import setup_logger

# Ensure stdout uses UTF-8 on Windows to handle unicode log characters
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── SSL / Proxy workaround ────────────────────────────────────────────────────
# Corporate / institutional networks often intercept HTTPS with a self-signed
# certificate.  We disable verification and suppress the resulting warning.
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

original_request = requests.Session.request

def patched_request(self, *args, **kwargs):
    kwargs["verify"] = False
    # Lift read timeout so large file chunks don't time out mid-transfer
    t = kwargs.get("timeout", None)
    if t is None:
        kwargs["timeout"] = 600.0
    elif isinstance(t, (int, float)):
        kwargs["timeout"] = max(t, 600.0)
    elif isinstance(t, tuple) and len(t) == 2:
        kwargs["timeout"] = (
            max(t[0] if t[0] is not None else 0, 600.0),
            max(t[1] if t[1] is not None else 0, 600.0),
        )
    return original_request(self, *args, **kwargs)

requests.Session.request = patched_request
# ─────────────────────────────────────────────────────────────────────────────

logger = setup_logger("download")

SKIP_PATTERNS = {".msgpack", ".h5", ".ot"}


def _should_skip(filename: str) -> bool:
    return any(filename.endswith(ext) for ext in SKIP_PATTERNS)


def download_model(
    repo_id: str,
    output_dir: str,
    token: str = None,
    max_retries: int = 5,
    retry_delay: float = 10.0,
) -> str:
    """
    Downloads every file in a Hugging Face model repo **sequentially**, with
    per-file retry logic.  This avoids the IncompleteRead / ChunkedEncoding
    errors that occur when parallel downloads are interrupted by a proxy.

    Already-downloaded files are skipped automatically (idempotent).
    """
    logger.info(f"Initiating sequential download of '{repo_id}' -> '{output_dir}'")
    os.makedirs(output_dir, exist_ok=True)

    api = HfApi()
    logger.info("Fetching repository file listing …")
    repo_files = [
        f.rfilename
        for f in api.list_repo_tree(repo_id=repo_id, recursive=True, token=token)
        if hasattr(f, "rfilename") and not _should_skip(f.rfilename)
    ]
    logger.info(f"Found {len(repo_files)} files to download.")

    failed = []
    for idx, filename in enumerate(repo_files, 1):
        dest_path = os.path.join(output_dir, filename.replace("/", os.sep))

        # Skip files that are already fully present
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            logger.info(f"[{idx}/{len(repo_files)}] SKIP (exists): {filename}")
            continue

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    f"[{idx}/{len(repo_files)}] Downloading (attempt {attempt}): {filename}"
                )
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=output_dir,
                    token=token,
                )
                logger.info(f"[{idx}/{len(repo_files)}] OK: {filename}")
                break
            except Exception as exc:
                logger.warning(
                    f"[{idx}/{len(repo_files)}] Attempt {attempt} failed for '{filename}': {exc}"
                )
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay}s …")
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"[{idx}/{len(repo_files)}] FAILED after {max_retries} attempts: {filename}"
                    )
                    failed.append(filename)

    if failed:
        logger.error(f"Download completed with {len(failed)} failed file(s): {failed}")
    else:
        logger.info(f"All files successfully downloaded to: {output_dir}")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robust sequential OLMo checkpoint downloader with per-file retries"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="allenai/OLMo-2-0425-1B",
        help="Hugging Face Hub repository ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/OLMo-2-0425-1B",
        help="Local directory to store downloaded weights",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face access token (optional, for gated repos)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Number of retry attempts per file on network failure",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=10.0,
        help="Seconds to wait between retry attempts",
    )

    args = parser.parse_args()
    download_model(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        token=args.token,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )
