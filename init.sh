export GITHUB_TOKEN="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"
export REPO_NAME="dourofficer/agent-grad"

git config --local user.name "dourofficer"
git config --local user.email "dourofficer@gmail.com"
git remote set-url origin https://staticpunch:${GITHUB_TOKEN}@github.com/${REPO_NAME}.git
