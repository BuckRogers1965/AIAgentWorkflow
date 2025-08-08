!#/usr/bin/bash
zip -r agent_workflow_release.zip \
    config.json dynamic_workflows_agents.py \
    requirements.txt \
    Quick_Start_Guide.md \
    editor_new_theme/*.py \
    service_demos/flask_web_service.py \
    unit_testing/*.py 
    
