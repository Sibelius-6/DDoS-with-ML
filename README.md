# Machine Learning-based DDoS Detection

**Note**: 
- Due to the time limit and the memory limit of the computer, we only sampled a small portion of data in our experiment. Thus our results might have some bias.
- We remove the PortMap in the test process of classifier since this attack type has not been seen in the train process.
- As in Dec. 2019, when we were using CICFlowMeter to process the pcap files, there was an one-hour time difference from UNB. When extracting different attack files, you must be aware of the one-hour shift.

**Goal**: Our experiment has two phases: correctly identify attacks from benign and classify the attack types. Also, we introduce TDA as a tool to evaluate the performance of attack detection.

## AE.py
as described in Section 2.2 of [report](work_report.pdf).

## ae+tda.py
as described in Section 2.2.3 of [report](work_report.pdf).

## MLPclassifier.py
as described in Section 2.3 of [report](work_report.pdf).

## other_classifier.py
as described in Section 2.3 of [report](work_report.pdf).

## utils.py
Some frequently used functions.
