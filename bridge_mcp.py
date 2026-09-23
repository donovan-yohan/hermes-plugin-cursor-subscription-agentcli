"""Inventory/parking MCP server for the Cursor agent. No host imports, no tool implementations.

The plugin starts this server as the only MCP server in a private workspace. It answers
`initialize` and `tools/list`, and parks every `tools/call`: the JSON-RPC request is held open
(no response) while the plugin — which watches the agent's stream-json output — ends the Hermes
response with finish_reason tool_calls. Hermes executes the tool with its own hooks and
approvals, then the plugin delivers results over the loopback socket named by
HERMES_CURSOR_BRIDGE_PORT; this server answers the parked requests in order and the agent
continues.
"""
import hashlib
import json
import os
import socket
import sys
import threading
from pathlib import Path


def main():
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    log_path = sys.argv[2] if len(sys.argv) > 2 else None

    def log(payload):
        if not log_path:
            return
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(json.dumps(payload) + '\n')

    # The port arrives as a literal argv (Cursor sanitizes MCP child env, so an env var
    # never reaches this process). argv[3] is authoritative; the env var is a fallback.
    raw_port = sys.argv[3] if len(sys.argv) > 3 else os.environ.get('HERMES_CURSOR_BRIDGE_PORT')
    port = int(raw_port)
    sock = socket.create_connection(('127.0.0.1', port))
    lock = threading.Lock()
    pending = []  # parked JSON-RPC rows, FIFO per tool name

    def answer(row, result):
        with lock:
            print(json.dumps({'jsonrpc': '2.0', 'id': row['id'], 'result': result}), flush=True)
        log({'answered': result is not None})

    def reader():
        for line in sock.makefile('r', encoding='utf-8'):
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if message.get('type') != 'deliver':
                continue
            for item in message.get('results') or []:
                text = item.get('result') or '(empty tool result)'
                content = [{'type': 'text', 'text': text}]
                if item.get('is_error'):
                    content = [{'type': 'text', 'text': 'ERROR: ' + text}]
                row = None
                for index, candidate in enumerate(pending):
                    if candidate['name'] == item.get('name'):
                        row = pending.pop(index)
                        break
                if row is None:
                    log({'stray_result': item.get('name')})
                    continue
                answer(row, {'content': content, 'isError': bool(item.get('is_error'))})

    threading.Thread(target=reader, daemon=True).start()
    log({'hello': True})
    try:
        for line in sys.stdin:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            method = row.get('method')
            if 'id' not in row:
                continue
            if method == 'initialize':
                answer(row, {'protocolVersion': '2024-11-05', 'capabilities': {'tools': {}},
                             'serverInfo': {'name': 'hermes', 'version': '1'}})
            elif method == 'tools/list':
                answer(row, {'tools': manifest})
            elif method == 'tools/call':
                params = row.get('params') or {}
                row = dict(row, name=params.get('name', ''))
                pending.append(row)
                log({'parked': params.get('name')})
            else:
                answer(row, {})
    finally:
        sock.close()


if __name__ == '__main__':
    main()
