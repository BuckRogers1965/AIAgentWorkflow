#!/usr/bin/bash


VERSION="v$(date +'%Y.%m.%d.%H%M')"
echo "🚀 Preparing release: $VERSION"

ARCHIVE_NAME="agent_workflow_release-$VERSION.zip"
echo "📦 Creating archive: $ARCHIVE_NAME"

zip -9 "releases/$ARCHIVE_NAME" \
    config.json dynamic_workflows_agents.py \
    requirements.txt \
    README.md \
    docs/*.md \
    editor/*.py \
    editor/themes.json \
    editor/README.md \
    service_demos/flask_web_service.py \
    service_demos/mcp_client_demo.py \
    service_demos/mcp_service.py \
    service_demos/README.md \
    unit_testing/*.py \
    unit_testing/README.md \
    utilities/config_diff.py
    
