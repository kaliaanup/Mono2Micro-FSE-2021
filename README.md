## Mono2Micro

This folder contains replication material for the FSE paper accepted. 

**Title**: Mono2Micro: A Practical and Effective Tool for Decomposing Monolithic Java Applications to Microservices [Industry Track]

**Authors**: Anup K. Kalia, Jin Xiao, Rahul Krishna, Saurabh Sinha, Maja Vukovic, Debasish Banarjee

**Affiliation**: IBM T. J. Watson Research Center, Yorktown Heights, NY

- Data Converters. This converts the data format in Mono2Micro to the formats relevant SOTA: FoSCI, COGCN, Bunch and MEM

- Datasets Runtime. This contains data for 4 open-source applications for all the relevant baselines. The data contains both input and output. 

    - acmeair
    - daytrader
    - jpetstore
    - plants

- Experimental Results.  This contains the details of experiments we conducted using the output from various approaches.

- Survey results. This contains the survey outcomes that we conducted in IBM. This does not contains participants details.


Unfortunately we cannot release the code for Mono2Micro. Feel free to use the trial version of Mono2Micro here.

https://www.ibm.com/cloud/mono2micro


Please cite our paper as the following.

```

@inproceedings{Kalia+FSE+2021,
author={Anup K. Kalia, Jin Xiao, Rahul Krishna, Saurabh Sinha, Maja Vukovic, Debasish Banerjee},
title={Mono2Micro: A Practical and Effective Tool for Decomposing Monolithic Java Applications to Microservices},
booktitle={ACM Foundations of Software Engineering (FSE)},
year={2021},
publisher={ACM},
address={Athens,Greece},
pages={1--11},
}

```
