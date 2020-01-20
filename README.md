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

## Data files

* attack.csv: a sampled portion of attacks (with a small amount of benign as well), mostly DNS attacks
* all_benign.csv: all benign traffics, which are extracted by the convention in original paper (details in report)
* train/*.csv: A.csv => type A attacks. On training day
* test/*.csv: same convention as above, but on testing day
* Note that all data files (except for all_benign) are sampled data, then there might be some discretion to the results in the report. Ideally all the models should be running against all data files. 
