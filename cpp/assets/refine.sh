#!/bin/bash

USAGE="./refine.sh <src> <dst>"
if [[ $# -ne 2 ]]
then
    echo "$USAGE"
    exit 1
fi

SRC="$1"
DST="$2"
cp "$SRC" "$DST"
gmsh -refine "$DST"