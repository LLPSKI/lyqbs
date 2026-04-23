#!/bin/bash

if [ "$3" == "--profile" ]; then
    export ENABLE_PROFILING=1
    export NCCL_PROFILER_PLUGIN=./libnccl-profiler-inspector.so
    export NCCL_INSPECTOR_ENABLE=1
    export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500
    export NCCL_INSPECTOR_DUMP_DIR=./nccl_logs
    export NCCL_INSPECTOR_DUMP_VERBOSE=0

    rm -rf ./nccl_logs*
fi

export OMP_NUM_THREADS=1
torchrun --nproc-per-node=$2 $1