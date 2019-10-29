#!/usr/bin/env bash

if [ $# -ne 1 ]; then
	echo "usage: clean <dir>"
	exit 1
fi

dir=$1

if [ -d $dir ] || [ -f $dir ]; then
	echo cleaning $dir
	rm -rf $dir
	echo "done"
else
	echo $dir not exist
fi
