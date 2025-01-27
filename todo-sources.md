
## Feature Engineering
* Compute the channel gain matrix from data
    - Search Info
* Compute the interference matrix [Usage](https://vbn.aau.dk/ws/portalfiles/portal/415102733/Learning_to_Allocate_Radio_Resources_in_Mobile_6G_in_X_Subnetworks_fv.pdf)
    - Discover how they are created. 

* Treat problem like a graph coloring

## Model Architecture
* DNN 
    - Distributed Channel (Sub-Band) allocation [Source](https://vbn.aau.dk/ws/portalfiles/portal/415102733/Learning_to_Allocate_Radio_Resources_in_Mobile_6G_in_X_Subnetworks_fv.pdf)

* GNN - Graph Neural Network
    - Approach: [Meta-Learning and GNN](https://ieeexplore.ieee.org/document/10319409)
    - Band Allocation = Coloring Problem

    * Resource
        - [Russian Dissertation](https://disser.spbu.ru/files/2024/disser_en_sun_qiushi.pdf)

* Multi-Step Solution -> smaller model 
    1. Determine Allocation
        + use mixed methods -> DNN to generate a good graph representation and coloring algo to optimize with edge removal [source](https://vbn.aau.dk/ws/portalfiles/portal/415102733/Learning_to_Allocate_Radio_Resources_in_Mobile_6G_in_X_Subnetworks_fv.pdf)
    2. Determine Power

## Training Method
- Supervised
- Offline Reinforcement Learning
- Reinforcement Learning

## Dataset

* **Obtain (Sub-)Optimal Target Samples**
    - The channel gains, and UE locations with respect to the BS are randomly distributed in the area to produce the channel vector for each UE. The **PSO algorithm** is used to determine the associated optimally assigned powers, which are calculated and stored in the dataset. [Source](https://disser.spbu.ru/files/2024/disser_en_sun_qiushi.pdf?page=38)

    - Use waterfilling algorithm for sub-optimal (in general conditions)
    - Use sequential iterative subband allocation (SISA) for sub-band sub-optimal allocation
    - Use graph coloring algorithm for sub-band assignment (CGC) for sub-band sub-optimal allocation.

* ¿Change Parameters?

## Sources to analyze

* [Graph with related / relevant papers](https://www.connectedpapers.com/main/5653dc094cb70d89258dab412fe4a7c3c760554c/Rate%20conforming-Sub%20band-Allocation-for-In%20factory-Subnetworks%3A-A-Deep-Neural-Network-Approach/graph)

* https://ieeexplore.ieee.org/abstract/document/4509705?casa_token=67aNnEhkfL0AAAAA:6GiIg9W2ifoZxdO_FuNqARdD5di-Dv1QiQJiFFb23OOV6I2djIOhf3fkQjvPnPyWlDNJ3nw6cw

## Surveys, General Methods, etc
* https://ieeexplore.ieee.org/abstract/document/8447187?casa_token=PypjV_eiIzkAAAAA:qLtGsvmpaBcks7K6Rl90QDVxLQURRKlM4dzjIW1_a0qDCiENB68VJMYRPcaqcWl88chOuHcivA
* Comparison of Sub-Band Allocation Algoritms in sub-networkds (https://arxiv.org/pdf/2403.11891)

## Baselines and Methods
* Dynamic Resource Allocation for Heterogeneous Services in Cognitive Radio Networks With Imperfect Channel Sensing
* [Deep-Learning-Based Resource Allocation for Transmit Power Minimization in Uplink NOMA IoT Cellular Networks](https://ieeexplore.ieee.org/abstract/document/10064048?casa_token=SejFmBUgjQwAAAAA:w_kOo4g6ZR-2YebYIqHOKTLk8b9hUmdckuoXf8egsdqDfhyZktYqE3xb40v3iIWBFNQz9dN1wQ)

* Rate-conforming Sub-band Allocation for In-factory Subnetworks: A Deep Neural Network Approach
    - They use the same simulation parameters (only change is center frequency)

* GNN-Based Meta-Learning Approach for Adaptive Power Control in Dynamic D2D Communications

* [Learning to Dynamically Allocate Radio Resources in Mobile 6G in-X Subnetworks](https://vbn.aau.dk/ws/portalfiles/portal/415102733/Learning_to_Allocate_Radio_Resources_in_Mobile_6G_in_X_Subnetworks_fv.pdf)

## Timetable