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

install_dir=~/anaconda3/envs/env_pytorch/bin

source activate env_pytorch

nohup $install_dir/python -u $exec >> $log_file 2>&1 &

printf "executing $exec - process ID: $!\n" | tee $log_file

