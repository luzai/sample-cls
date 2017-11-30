#!/usr/bin/env bash
for var  in $(cat ./src/corrupt)
do 
	echo $var
	rm data/imagenet-raw/$var -r 
done
