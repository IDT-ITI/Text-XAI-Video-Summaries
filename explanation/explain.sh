# -*- coding: utf-8 -*-

numbers=($(seq 1 50))
for num in "${numbers[@]}"; do
  python explain.py -m models/tvsum.pkl -v ../data/TVSum/video_"$num".mp4
done

numbers=($(seq 1 25))
for num in "${numbers[@]}"; do
  python explain.py -m models/summe.pkl -v ../data/SumMe/video_"$num".mp4
done
