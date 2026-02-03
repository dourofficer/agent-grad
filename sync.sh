#!/bin/bash
set -e

# ==============================================================================
# Sync Script
# 
# Usage: ./sync.sh
# 
# Description:
# 1. Syncs the 'resources' directory to Hugging Face Hub (Push).
# 2. Syncs with GitHub:
#    - Auto-commits local changes.
#    - Pulls remote changes (rebase).
#    - Syncs environment with 'uv'.
#    - Pushes changes to remote.
# 
# Prerequisites:
# - uv must be installed.
# - HF_TOKEN and HF_REPO_ID should be set (env vars or .env file).
# ==============================================================================

# --- Configuration ---

# Directory of this script
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env file if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Loading configuration from .env..."
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Hugging Face Settings
export HF_REPO_ID="${HF_REPO_ID:-pvhoang14/self-healing-llm}"
export HF_TOKEN="${HF_TOKEN:-}"
HF_PATH_IN_REPO="${HF_PATH_IN_REPO:-resources}"
HF_REPO_TYPE="${HF_REPO_TYPE:-dataset}"
LOCAL_RESOURCES_DIR="$PROJECT_ROOT/resources"
PYTHON_UPLOAD_SCRIPT="$PROJECT_ROOT/src/utils/python_scripts/hf_upload.py"
PYTHON_DOWNLOAD_SCRIPT="$PROJECT_ROOT/src/utils/python_scripts/hf_download.py"

# --- Functions ---

check_uv() {
    if ! command -v uv &> /dev/null; then
        echo "Error: 'uv' is not installed or not in PATH."
        exit 1
    fi
}

check_hf_config() {
    if [ -z "$HF_REPO_ID" ]; then
        echo "Error: HF_REPO_ID is not set."
        exit 1
    fi
    if [ -z "$HF_TOKEN" ]; then
        echo "Error: HF_TOKEN is not set. Ensure it is exported or in a .env file."
        exit 1
    fi
}

sync_github() {
    echo "----------------------------------------"
    echo "Syncing GitHub (Code & Environment)..."
    echo "----------------------------------------"

    # 0. Ensure we are on the main branch
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    if [ "$CURRENT_BRANCH" = "HEAD" ]; then
        echo "⚠️  Detached HEAD detected. Switching to 'main'..."
        git checkout main || {
            echo "Error: Could not switch to main. Please resolve conflicts or stash changes."
            exit 1
        }
        CURRENT_BRANCH="main"
    fi

    echo "Current branch: $CURRENT_BRANCH"

    # 1. Check for uncommitted changes and commit them
    if [ -n "$(git status --porcelain)" ]; then
        echo "Detected uncommitted changes. Committing..."
        git add .
        git commit -m "Auto-sync: $(date '+%Y-%m-%d %H:%M:%S')"
    else
        echo "No local changes to commit."
    fi

    # 2. Pull remote changes (using rebase to keep history clean)
    echo "Pulling from origin/$CURRENT_BRANCH..."
    git pull origin "$CURRENT_BRANCH" --rebase

    # 3. Sync environment (dependencies/python version)
    echo "Syncing environment with uv..."
    uv sync

    # 4. Push changes
    echo "Pushing instructions to origin/$CURRENT_BRANCH..."
    git push origin "$CURRENT_BRANCH"

    echo "GitHub sync complete."
}

sync_resources() {
    echo ""
    echo "----------------------------------------"
    echo "Syncing resources to Hugging Face..."
    echo "----------------------------------------"

    if [ -d "$LOCAL_RESOURCES_DIR" ]; then
        check_hf_config
        echo "Uploading resources to $HF_REPO_ID..."
        mkdir -p "$LOCAL_RESOURCES_DIR"
        
        # 1. Download Changes (Pull)
        echo "Downloading updates from $HF_REPO_ID..."
        # We use PROJECT_ROOT as local_dir because hf_download.py handles the path_in_repo structure
        uv run python "$PYTHON_DOWNLOAD_SCRIPT" \
            --local_dir "$PROJECT_ROOT" \
            --repo_id "$HF_REPO_ID" \
            --path_in_repo "$HF_PATH_IN_REPO" \
            --token "$HF_TOKEN" \
            --repo_type "$HF_REPO_TYPE"

        # 2. Upload Changes (Push)
        echo "Uploading local changes to $HF_REPO_ID..."
        uv run python "$PYTHON_UPLOAD_SCRIPT" \
            --local_dir "$LOCAL_RESOURCES_DIR" \
            --repo_id "$HF_REPO_ID" \
            --path_in_repo "$HF_PATH_IN_REPO" \
            --token "$HF_TOKEN" \
            --repo_type "$HF_REPO_TYPE"
            
        echo "Resource sync complete."
    else
        echo "Warning: '$LOCAL_RESOURCES_DIR' not found. Skipping."
    fi
}

# --- Main Execution ---

echo "Starting Multi-Environment Sync..."
check_uv

# Order: Sync Resources -> Sync GitHub
# (Resources first as requested, though order often matters less unless code depends on new resources immediately)
sync_resources
sync_github

echo ""
echo "✅ All sync operations completed successfully."
