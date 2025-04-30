import os
import socket
import pickle
import json
import time
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Stream preprocessed batch_*.pkl files over a TCP socket as JSON lines"
    )
    parser.add_argument(
        "--host", default="localhost", help="Host address to bind (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=6100, help="Port number to bind (default: 6100)"
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Folder containing batch_*.pkl files to stream",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to wait between sending consecutive batches",
    )
    args = parser.parse_args()

    files = sorted(
        [f for f in os.listdir(args.folder) if f.startswith("batch_") and f.endswith(".pkl")]
    )
    if not files:
        print(f"No batch files found in {args.folder}. Exiting.")
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((args.host, args.port))
    sock.listen(1)
    print(f"[stream_server] Listening on {args.host}:{args.port}, waiting for client...")

    conn, addr = sock.accept()
    print(f"[stream_server] Client connected from {addr}, starting stream.")

    for fn in files:
        path = os.path.join(args.folder, fn)
        try:
            with open(path, "rb") as f:
                batch = pickle.load(f)
        except Exception as e:
            print(f"[stream_server] ERROR loading {fn}: {e}, skipping.")
            continue

        payload = json.dumps(batch) + "\n"
        try:
            conn.sendall(payload.encode("utf-8"))
            print(f"[stream_server] Sent {fn}")
        except (BrokenPipeError, ConnectionResetError):
            print("[stream_server] Client disconnected, stopping stream.")
            break

        time.sleep(args.sleep)

    conn.close()
    sock.close()
    print("[stream_server] Stream finished, server shutting down.")

if __name__ == "__main__":
    main()