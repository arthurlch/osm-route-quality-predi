#!/bin/bash
set -e

echo "📦 Installing dependencies..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

uv venv .venv

source .venv/bin/activate

uv pip install -r requirements.txt
uv pip install -e .

echo "✅ Setup complete!"