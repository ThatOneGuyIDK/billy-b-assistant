#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEMD_DEST_DIR="/etc/systemd/system"

echo "🔧 Copying systemd service files..."
for service_file in "$SCRIPT_DIR"/*.service; do
    service_name=$(basename "$service_file")
    echo "  ↪ Installing $service_name..."
    sudo cp "$service_file" "$SYSTEMD_DEST_DIR/$service_name"
    sudo chmod 644 "$SYSTEMD_DEST_DIR/$service_name"
    sudo systemctl enable "$service_name"
    sudo systemctl restart "$service_name"
done

echo "⏳ Reloading systemd..."
sudo systemctl daemon-reexec
sudo systemctl daemon-reload

echo "✅ Billy service installed."
