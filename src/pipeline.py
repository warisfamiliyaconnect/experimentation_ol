import json
import os
import argparse
from typing import Dict, Any, List

def process_telemetry(log_file: str, output_file: str, enforce_eos: bool = True) -> None:
    """Reads telemetry JSONL logs, filters for high-quality interactions,

    and writes them in ChatML format to output_file.
    """
    if not os.path.exists(log_file):
        print(f"Error: Log file '{log_file}' does not exist.")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    total_records = 0
    exported_records = 0
    discarded_stats = {
        "missing_text": 0,
        "error_status": 0,
        "oom": 0,
        "timeout": 0,
        "cancellation": 0,
        "non_eos_finish": 0
    }

    print(f"Processing logs from: {log_file}")
    with open(log_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            total_records += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipped invalid JSON line {total_records}: {e}")
                continue

            # 1. Quality Filters
            # Must have prompt and response text
            prompt = data.get("prompt")
            response = data.get("response")
            if not prompt or not response:
                discarded_stats["missing_text"] += 1
                continue

            # Must not be an error
            if data.get("status") == "error" or data.get("error") is not None:
                discarded_stats["error_status"] += 1
                continue

            # Check reliability flags
            if data.get("is_oom", False):
                discarded_stats["oom"] += 1
                continue
            if data.get("is_timeout", False):
                discarded_stats["timeout"] += 1
                continue
            if data.get("is_cancellation", False):
                discarded_stats["cancellation"] += 1
                continue

            # Finish reason check (ensure thought completed)
            finish_reason = data.get("finish_reason")
            allowed_finish_reasons = {"eos", "stop"}
            if enforce_eos and finish_reason not in allowed_finish_reasons:
                discarded_stats["non_eos_finish"] += 1
                continue

            # 2. Format to ChatML structure
            chat_record = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
            }

            f_out.write(json.dumps(chat_record, ensure_ascii=False) + "\n")
            exported_records += 1

    print(f"Processing complete!")
    print(f"Total telemetry logs parsed: {total_records}")
    print(f"Exported to SFT Dataset:    {exported_records}")
    print("Discarded logs breakdown:")
    for reason, count in discarded_stats.items():
        print(f"  - {reason.replace('_', ' ').capitalize()}: {count}")
    print(f"SFT Dataset saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telemetry to SFT dataset formatting pipeline.")
    parser.add_argument("--log-file", type=str, default="logs/telemetry.jsonl", help="Path to telemetry log file")
    parser.add_argument("--output-file", type=str, default="data/processed/sft_dataset.jsonl", help="Path to write clean dataset")
    parser.add_argument("--allow-truncated-finish", action="store_true", help="Allow generations that did not finish with EOS/Stop (e.g. hit length limit)")

    args = parser.parse_args()
    process_telemetry(args.log_file, args.output_file, enforce_eos=not args.allow_truncated_finish)
