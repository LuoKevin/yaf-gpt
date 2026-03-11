#!/usr/bin/env bash
set -euo pipefail

quest="${1:-}"
python_bin="python3"
if [[ -x ".venv/bin/python" ]]; then
  python_bin=".venv/bin/python"
fi

if [[ -z "${quest}" ]]; then
  echo "usage: bash scripts/workflow_game_checks.sh <quest0|quest1|quest2|quest3|quest4|quest5>"
  exit 1
fi

case "${quest}" in
  quest0)
    test -f docs/workflow_game.md
    test -f docs/workflow_scorecard_template.md
    ;;
  quest1)
    "${python_bin}" -m unittest backend.tests.test_api_routes
    ;;
  quest2)
    "${python_bin}" -m unittest backend.tests.test_passage_image_service backend.tests.test_api_routes
    npm --prefix frontend run build
    ;;
  quest3)
    "${python_bin}" -m unittest backend.tests.test_persona_chat_service backend.tests.test_api_routes
    npm --prefix frontend run build
    ;;
  quest4)
    "${python_bin}" -m unittest backend.tests.test_hymn_service backend.tests.test_media_providers backend.tests.test_api_routes
    npm --prefix frontend run build
    ;;
  quest5)
    test -f scripts/smoke_e2e.py
    "${python_bin}" -m unittest discover -s backend/tests -p 'test_*.py'
    npm --prefix frontend run build
    ;;
  *)
    echo "unknown quest '${quest}'"
    exit 1
    ;;
esac

printf "Quest %s checks passed.\n" "${quest}"
