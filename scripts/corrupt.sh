#!/usr/bin/env bash
cd /mnt/nfs1703/kchen/imagenet-raw/n00005787

for i in $(find . -name '*.JPEG'); do
#    var=$(identify $i | sed -nE 's/.*JPEG JPEG (.*)x([0-9]+) .*/\1 \2/p')
#    echo $var
#    echo $var >> size.txt
#    var=$(jpeginfo -c $i | grep ERROR)
#    echo $var
    echo $i
    convert $i /dev/null -regard-warnings
    if [ $? -eq 0 ];then
        echo $i
        echo $i >> corrupt.txt
    fi
done

