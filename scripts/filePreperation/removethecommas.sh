find . -type f -name '*[:,"]*' |
  while IFS= read -r; do
    mv -- "$REPLY" "${REPLY//[:,\"]}"
  done
