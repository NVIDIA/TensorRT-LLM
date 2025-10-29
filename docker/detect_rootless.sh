#!/bin/sh

if [ "$(docker context inspect --format '{{.Endpoints.docker.Host}}' "$(docker context show)")" = "unix:///run/user/$(id -u)/docker.sock" ]; then
    echo 1
else
    echo 0
fi
