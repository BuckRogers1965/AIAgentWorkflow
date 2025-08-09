#!/bin/bash

# Ensure at least one argument (commit message) is provided
if [ -z "$1" ]; then
  echo "Usage: gitsave \"commit message\" [files or directories to add]"
  exit 1
fi

# Store the commit message and shift to get the file arguments
commit_message="$1"
shift

# Ensure at least two arguments (commit message) is provided
if [ -z "$1" ]; then
  echo "Usage: gitsave \"commit message\" [files or directories to add]"
  exit 1
fi


# Pull the latest changes from the repository
echo "Pulling latest changes..."
if ! git pull; then
  echo "Error: Failed to pull latest changes."
  exit 1
fi

# Add specified files or directories, or all uncommitted changes if none specified
if [ "$#" -eq 0 ]; then
  echo "No files specified. Adding all uncommitted changes."
  git add .
else
  echo "Adding specified files: $@"
  git add "$@"
fi

# Commit the changes with the provided message
echo "Committing changes with message: \"$commit_message\""
if ! git commit -m "$commit_message"; then
  echo "Error: Failed to commit changes."
  exit 1
fi

# Push the changes to the remote repository
echo "Pushing changes..."
if ! git push; then
  echo "Error: Failed to push changes."
  exit 1
fi

echo "Changes saved and pushed successfully!"

