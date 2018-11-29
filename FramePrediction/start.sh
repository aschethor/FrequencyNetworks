#!/bin/bash

log_dir=Logs

if [ ! -d "$log_dir" ]
then
	mkdir $log_dir
fi

timestamp=`date "+%d.%m.%Y-%H:%M:%S.%3N"`
log_file="$log_dir/$timestamp.log"

if [[ $# -ge 1 ]]
then
	exec=$1
else
	exec=main.py
fi

install_dir=~/anaconda3/bin

printf "executing $exec\n" | tee $log_file

$install_dir/python -u $exec 2>&1 | tee -a $log_file
