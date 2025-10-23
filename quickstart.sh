#!/bin/bash
# Quick start script for Music Map project

set -e

echo "=================================="
echo "Music Map Quick Start"
echo "=================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found!"
    echo "Creating .env from template..."
    cp .env.template .env
    echo ""
    echo "Please edit .env and add your Spotify credentials:"
    echo "  - SPOTIFY_CLIENT_ID"
    echo "  - SPOTIFY_CLIENT_SECRET"
    echo ""
    echo "Get credentials from: https://developer.spotify.com/dashboard"
    echo ""
    read -p "Press Enter after you've added your credentials..."
fi

# Test setup
echo ""
echo "Testing setup..."
python test_setup.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Setup test failed. Please fix the issues above."
    exit 1
fi

echo ""
echo "=================================="
echo "Setup test passed! 🎉"
echo "=================================="
echo ""
echo "Now you can run the pipeline:"
echo ""
echo "TESTING (recommended first):"
echo "  1. python setup/scrape_songs.py --subset 1000"
echo "  2. python setup/generate_embeddings.py"
echo "  3. python setup/train_umap.py"
echo "  4. python user/main.py --timeframe week"
echo ""
echo "PRODUCTION (full scale):"
echo "  1. python setup/scrape_songs.py"
echo "  2. python setup/generate_embeddings.py"
echo "  3. python setup/train_umap.py"
echo "  4. python user/main.py --timeframe month"
echo ""
echo "=================================="
