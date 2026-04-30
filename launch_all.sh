#!/bin/bash
# Launch 4 parallel GPU jobs for all 16 LongBench datasets.
# Balanced by actual measured runtime (greedy bin-packing).
#
# Usage:  bash launch_all.sh

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

# ── Dataset → GPU assignment (balanced by actual runtime from previous run) ──
# GPU 0: narrativeqa(10507) + triviaqa(2069) + qasper(1155) + multifieldqa_en(1114) = 14,845s (~4.1h)
# GPU 1: repobench-p(7394) + musique(2432) + hotpotqa(1935) + passage_retrieval_en(1909) = 13,670s (~3.8h)
# GPU 2: multi_news(5274) + qmsum(4331) + passage_count(2332) + 2wikimqa(1227) = 13,164s (~3.7h)
# GPU 3: gov_report(5025) + lcc(4546) + trec(2156) + samsum(1913) = 13,640s (~3.8h)

GPU0_TASKS="triviaqa,qasper,multifieldqa_en,narrativeqa"
GPU1_TASKS="repobench-p,musique,hotpotqa,passage_retrieval_en"
GPU2_TASKS="multi_news,qmsum,passage_count,2wikimqa"
GPU3_TASKS="gov_report,lcc,trec,samsum"

echo "============================================"
echo "Launching 4 parallel LongBench evaluations"
echo "============================================"
echo "GPU 0 (~4.1h): $GPU0_TASKS"
echo "GPU 1 (~3.8h): $GPU1_TASKS"
echo "GPU 2 (~3.7h): $GPU2_TASKS"
echo "GPU 3 (~3.8h): $GPU3_TASKS"
echo "============================================"

mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python Results/run_longbench_sticky.py --gpu 0 --tasks "$GPU0_TASKS" > logs/gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python Results/run_longbench_sticky.py --gpu 1 --tasks "$GPU1_TASKS" > logs/gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python Results/run_longbench_sticky.py --gpu 2 --tasks "$GPU2_TASKS" > logs/gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python Results/run_longbench_sticky.py --gpu 3 --tasks "$GPU3_TASKS" > logs/gpu3.log 2>&1 &

echo ""
echo "All 4 jobs launched. Logs in LongBenchSticky/logs/"
echo "Monitor:  tail -f logs/gpu*.log"
echo "Check:    grep 'Total:' logs/gpu*.log"
echo ""

wait
echo "============================================"
echo "All jobs finished!"
echo "============================================"
grep "Total:" logs/gpu*.log
grep "Done\|score:" logs/gpu*.log
