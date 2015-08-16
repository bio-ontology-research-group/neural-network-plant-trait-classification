#!/bin/sh

if [ ! -f allPhotos ]; then
	mkdir allPhotos
fi

if [ ! -f ~/datasets ]; then
	mkdir ~/datasets
fi

cd allPhotos
cat ../downloadLinks.txt | parallel --gnu "wget {}"
cd ..
mv ./allPhotos ~/datasets
cd ~/datasets
find . -type f -name '*[:,"]*' |
  while IFS= read -r; do
    mv -- "$REPLY" "${REPLY//[:,\"]}"
  done
