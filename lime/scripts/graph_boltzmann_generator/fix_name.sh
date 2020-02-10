#!/bin/sh

git filter-branch --env-filter '

CORRECT_NAME="Yuanqing Wang"
CORRECT_EMAIL="wangyq@wangyq.net"

if [ "$GIT_COMMITTER_NAME" = "Flora Zhao" ]
then
    export GIT_COMMITTER_NAME="Yuanqing Wang"
    export GIT_COMMITTER_EMAIL="wangyq@wangyq.net"
fi
' --tag-name-filter cat -- --branches --tags
