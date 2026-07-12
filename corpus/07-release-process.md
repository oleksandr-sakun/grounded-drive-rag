# Release Process

## Deployment windows
Deploys to production are allowed Monday to Thursday, 09:00–16:00 CET.
No deploys on Friday. No deploys during a customer peak window (announced in #eng each quarter).

## Requirements before deploy
- All CI checks green
- At least one approving review
- A rollback plan written in the pull request description

## Rollback
Any engineer may roll back a deploy without asking permission. Rolling back is never the wrong call.
Post a note in #eng afterwards explaining what happened.

## Hotfixes
An S1 incident suspends the deployment window. Hotfixes may ship at any time, including Friday, with a single approving review.
