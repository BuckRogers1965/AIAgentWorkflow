#!/usr/bin/bash

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

# --- Step 4: If Zip Succeeded, Tag the Release in Git ---
# This block only runs if the zip command was successful.
echo "📌 Tagging release in Git with: $VERSION"

# Create an annotated tag. This is better than a lightweight tag.
git tag -a "$VERSION" -m "Release of $VERSION"

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to create git tag. You may have uncommitted changes or the tag may already exist."
    echo "Aborting without pushing."
    exit 1
fi

echo "✅ Adding new archive to project .."
git add "releases/$ARCHIVE_NAME"
git commit -m "feat: Add release archive for version $VERSION"
git push

echo "⬆️ Pushing tag to remote repository..."
git push origin "$VERSION"

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to push tag to remote. Check your network connection and permissions."
    exit 1
fi

echo "✅ Git tag pushed successfully."

# --- Final Instructions ---
echo ""
echo "🎉 Release process complete!"
echo "--------------------------------"
echo "Next steps:"
echo "1. Go to your GitHub repository's 'Releases' page."
echo "2. Click 'Draft a new release'."
echo "3. Choose the tag '$VERSION' from the dropdown."
echo "4. Upload the file '$ARCHIVE_PATH' as the binary."
echo "5. Write your release notes and publish!"
