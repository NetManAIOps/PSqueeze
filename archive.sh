#!/usr/bin/env bash
cd "$(dirname "$0")"
basename=$(basename $(realpath $(dirname "$0")))
echo ${basename}
if [ -z "$(git status --porcelain)" ]; then
  # Working directory clean
  commit_id=$(git rev-parse HEAD)
  git archive --format=zip --prefix=${basename}/ HEAD > ${basename}_code_${commit_id}.zip
else
  # Uncommitted changes
  git status
fi