#!/bin/bash

DIR="tensorrt_llm/examples"
SOURCE="tensorrt_llm==0.9.0.dev0"
TARGET="tensorrt_llm==0.9.0.dev1"

find "$DIR" -type f \( -name "requirements.txt" -o -name "constraints.txt" \) | while read -r file; do
    sed -i "s/$SOURCE/$TARGET/g" "$file"
done
