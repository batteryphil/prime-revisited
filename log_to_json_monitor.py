import time
import re
import json
import os

log_file = "training_titan_mimo.log"
json_file = "stats_titan_mimo.json"
pattern = re.compile(r"\[PRIME\] Step (\d+) \| Loss: ([\d.]+) \| Flips: (\d+) \| TPS: ([\d.]+) \| V\+:([\d.]+) V-:([\d.]+)")

def run():
    print("[TELEMETRY] Starting live log-to-JSON parser...")
    history = []
    
    # Check if we should parse from the beginning or just tail it
    # We will parse from the beginning to build the history array
    if not os.path.exists(log_file):
        print(f"[WARN] {log_file} not found.")
        return

    with open(log_file, "r") as f:
        while True:
            line = f.readline()
            if not line:
                time.sleep(2)  # Wait for new logs
                continue
            
            match = pattern.search(line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                flips = int(match.group(3))
                tps = float(match.group(4))
                vote_pos = float(match.group(5))
                vote_neg = float(match.group(6))
                
                entry = {
                    "step": step,
                    "loss": loss,
                    "flips": flips,
                    "tps": tps,
                    "vote_pos": vote_pos,
                    "vote_neg": vote_neg,
                    "vote_neut": max(0.0, 1.0 - vote_pos - vote_neg)
                }
                
                # Merge heavy metrics if available
                if os.path.exists("heavy_telemetry.json"):
                    try:
                        with open("heavy_telemetry.json", "r") as hf:
                            heavy = json.load(hf)
                            # Only merge if the checkpoint step is somewhat recent (e.g. within 100 steps)
                            if abs(heavy.get("step", 0) - step) < 100:
                                entry.update(heavy)
                    except Exception:
                        pass
                
                history.append(entry)
                
                # The dashboard expects an array of the last N points to draw the chart
                if len(history) > 100:
                    history = history[-100:]
                    
                # Write atomically to avoid read collisions from the UI
                tmp_file = json_file + ".tmp"
                with open(tmp_file, "w") as out:
                    json.dump(history, out)
                os.replace(tmp_file, json_file)

if __name__ == "__main__":
    run()
