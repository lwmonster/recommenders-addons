#!/bin/bash
rm -rf ./ckpt
sh stop.sh
sleep 1

PS_LIST="localhost:2220,localhost:2221,localhost:2222,localhost:2223"
WORKER_LIST="localhost:2231,localhost:2232,localhost:2233,localhost:2234"
CHIEF="localhost:2230"

# Start 4 PS nodes
for i in 0 1 2 3; do
  python movielens-1m-keras-ps.py --ps_list="$PS_LIST" --worker_list="$WORKER_LIST" --chief="$CHIEF" --task_mode="ps" --task_id=$i &
  sleep 1
done

# Start 4 Worker nodes
for i in 0 1 2 3; do
  python movielens-1m-keras-ps.py --ps_list="$PS_LIST" --worker_list="$WORKER_LIST" --chief="$CHIEF" --task_mode="worker" --task_id=$i &
  sleep 1
done

# Start Chief (foreground)
python movielens-1m-keras-ps.py --ps_list="$PS_LIST" --worker_list="$WORKER_LIST" --chief="$CHIEF" --task_mode="chief" --task_id=0
echo "ok"
