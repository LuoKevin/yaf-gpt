# Workflow Game: Feature Delivery Quests

This repository uses a developer workflow game to track implementation progress for major feature work.
The game is for engineering execution only, not end-user product behavior.

## Levels and XP

- Level 1: 0-149 XP
- Level 2: 150-349 XP
- Level 3: 350-649 XP
- Level 4: 650-999 XP
- Level 5: 1000+ XP

## Quest Rewards

- Quest 0: +100 XP
- Quest 1: +150 XP
- Quest 2: +200 XP
- Quest 3: +250 XP
- Quest 4: +300 XP
- Quest 5 (Boss Battle): +400 XP

Total campaign XP: 1400.

## Definition of Done Gates

Each quest is complete only when:

1. Code changes for the quest are merged locally and buildable.
2. The quest verification command finishes successfully.
3. A scorecard entry is recorded in `docs/workflow_scorecard_template.md` (or a copy of it).

## Quest Verification Commands

- Quest 0: `bash scripts/workflow_game_checks.sh quest0`
- Quest 1: `bash scripts/workflow_game_checks.sh quest1`
- Quest 2: `bash scripts/workflow_game_checks.sh quest2`
- Quest 3: `bash scripts/workflow_game_checks.sh quest3`
- Quest 4: `bash scripts/workflow_game_checks.sh quest4`
- Quest 5: `bash scripts/workflow_game_checks.sh quest5`

## Boss Battle Rules

Quest 5 is only valid if all prior quests are marked complete.

Boss battle requires evidence of:

1. End-to-end flow: passage lookup -> study plan -> image -> persona chat -> hymn + job polling.
2. Route-level regression tests passing.
3. Service-level tests passing.
4. Frontend production build success.
5. Updated runbook/release note notes for new endpoints and env toggles.
