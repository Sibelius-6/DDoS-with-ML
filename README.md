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
First process the pcap files via [CICFlowMeter](http://netflowmeter.ca/). The output is csv files. They are 1-to-1 correspondance. For example, `A.pcap` -> `A.csv`.

The source pcap files from UNB-DDoS2019 are spanning whole capturing period, something like
```
file_1.pcap/file_1.csv   9:00 - 9:10
file_2.pcap/file_2.csv   9:10 - 9:13
.
.
.
```
Since CICFlowMeter generates the timestamp of each traffic flow, we can know the attack type if this traffic is attack.

### attack.csv
a sampled portion of attacks (with a small amount of benign as well). Here I just used one pcap/csv file serving illustrating purposes. Since the time duration of this file, let's say `file_X`, falls between the DNS attack period specified in UNB paper, thus the attack traffics of this file are all DNS attacks. Also, note that there are some benign traffics in this file.

**Note**: 
### all_benign.csv
all benign traffics, which are extracted by the convention in original paper (details in report). Here is a pseudo python code on how you can extract all benign traffics:

```python
li = [] # initialize the empty dataframe 

for file in *.csv: # here *.csv represents the list of all csv files [file_1.csv, file_2.csv, ...]
    df = pd.read_csv(file) # store csv into dataframe
    df = df[(df['Src IP'] != '172.16.0.5') & (df['Dst IP'] != '172.16.0.5')] # if src/dst ip != 172.16.0.5, then it's benign
    li.append[df]
    
benign = pd.concat(li)
benign.to_csv('all_benign.csv')
```

### train/*.csv
A.csv => type A attacks. On training day. (Here I also picked some files as attack files, similar to how I picked attack.csv).

### test/*.csv
same convention as above, but on testing day.


