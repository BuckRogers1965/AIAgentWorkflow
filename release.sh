#!/usr/bin/bash

# --- Check for the required release message ---
if [ -z "$1" ]; then
    echo "❌ Error: A release message is required."
    echo "Usage: ./release.sh \"Your descriptive release message here.\""
    exit 1
fi

# --- Capture the release message from the first command-line argument ---
RELEASE_MESSAGE="$1"

# --- Pre-flight Check: Ensure GitHub CLI is installed ---
if ! command -v gh &> /dev/null
then
    echo "❌ Error: GitHub CLI ('gh') is required. Install gh with sudo apt update; sudo apt install gh"
    echo "create classic token at https://github.com/settings/tokens"
    echo "make sure it has repo checked and read:org under admin"
    echo "after install run echo "github token you created" | gh auth login --with-token"
    exit 1
fi


VERSION="v$(date +'%Y.%m.%d.%H%M')"
echo "🚀 Preparing release: $VERSION"

ARCHIVE_NAME="agent_workflow_release-$VERSION.zip"
echo "📦 Creating archive: $ARCHIVE_NAME"


zip "releases/$ARCHIVE_NAME" \
    config.json dynamic_workflows_agents.py \
    requirements.txt \
    README.md \
    Quick_Start_Guide.md \
    editor_new_theme/*.py \
    editor_new_theme/themes.json \
    editor_new_theme/README.md \
    service_demos/flask_web_service.py \
    service_demos/README.md \
    unit_testing/*.py \
    unit_testing/README.md
    
# Check the exit code of the last command ($?). '0' means success.
if [ $? -ne 0 ]; then
    # The zip command failed.
    echo "❌ Error: Failed to create zip archive. Cleaning up..."
    # Clean up the potentially broken zip file.
    rm -f "releases/$ARCHIVE_NAME"
    # Exit with an error code to stop the script.
    exit 1
fi

echo "✅ Archive created successfully."

echo "✅ Adding new archive to project .."
git add "releases/$ARCHIVE_NAME"
git commit -m "feat: Add release archive for version $VERSION"
git push

# Create an annotated tag for the current commit.
git tag -a "$VERSION" -m "Release of $VERSION"
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to create git tag. Make sure your code is committed."
    exit 1
fi

# Push the new tag to the remote repository.
echo "⬆️ Pushing tag to remote repository..."
git push origin "$VERSION"
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to push tag to remote."
    exit 1
fi
echo "✅ Git tag pushed successfully."

# Create the GitHub release and upload the local zip file as an asset.
echo "🌐 Creating GitHub release and uploading 'releases/$ARCHIVE_NAME' as an asset..."
gh release create "$VERSION" "releases/$ARCHIVE_NAME" --title "Release $VERSION" --notes "$RELEASE_MESSAGE - Automatic package release for version $VERSION."
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to create GitHub release and upload asset."
    exit 1
fi

echo "🎉🚀🎉 Your release is fully published on GitHub!"
