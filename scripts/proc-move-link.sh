#!/usr/bin/env bash
for link in $(find ./imagenet-raw -type l)
do
  loc=$(dirname $link)
  dir=$(readlink $link)
  mv $dir $loc
  rm $link
done


