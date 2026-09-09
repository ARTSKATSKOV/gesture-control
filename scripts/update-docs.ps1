$ProjectPath = "C:\Users\awesa\Projects\gesture-control"

$Prompt = @"
Update the Obsidian documentation for the current project.

Inspect the latest Git commit and git diff HEAD~1..HEAD.

Follow the project-documenter skill.

IMPORTANT:
- Do not modify source code.
- Do not modify Git contents.
- Do not create commits.
- Write documentation only to the configured Obsidian Vault.
- Do not duplicate documentation for a commit already documented.
"@

& hermes `
    --in $ProjectPath `
    --skills project-documenter `
    --oneshot $Prompt

exit $LASTEXITCODE