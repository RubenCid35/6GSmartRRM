
# Publication Study

In this document, we compare different publications and their suitability for the problem we want to solve.
Most publications propose a solution that may be effective but requires modifications to become coherent with
the project aims and constraints.

## **Sub-Band Assignment and Power Control for IoT Cellular Networks via Deep Learning**

**Paper**: [Sub-Band Assignment and Power Control for IoT Cellular Networks via Deep Learning](https://ieeexplore.ieee.org/abstract/document/9682739)

<table border="1" style="border-collapse: collapse; width: 100%;">
    <tr>
        <th style="padding: 10px; background-color: #f2f2f2;">Aspect</th>
        <th style="padding: 10px; background-color: #f2f2f2;">Details</th>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Tasks</strong></td>
        <td style="padding: 10px;">Subband Allocation, Power Control</td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>System</strong></td>
        <td style="padding: 10px;">IoT Cellular Networks (MIMO Systems)</td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Objective</strong></td>
        <td style="padding: 10px;">Maximize the achievable sum rate of IoT users with low complexity</td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Method</strong></td>
        <td style="padding: 10px; padding-bottom: 5px;">
            <ul>
                <li>Two-Stage Optimization Method with Deep Learning Models</li>
                <li>SubBand Allocation stage with CNN architecture </li>
                <li>Power   Allocation stage with FNN architecture </li>
            </ul>
        </td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Validity</strong></td>
        <td style="padding: 10px; padding-bottom: 5px;">
            <p style="color: red; font-weight: 600; padding-left: 10px; align-items: center; justify-items: center; display: flex">
                &#10006; Not Valid
            </p>
            <ul>
                <li>The assumption is that users can't share subbands.</li>
                <li markdown="1">In our case, <strong>K < N</strong>, so subband sharing is necessary.</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Required <br> Modifications</strong></td>
        <td style="padding: 10px;">
            Adapt the subband allocation stage of the approach for our system.
        </td>
    </tr>
</table>

## **Deep-Learning-Based Resource Allocation for Transmit Power Minimization in Uplink NOMA IoT Cellular Networks**

**Paper**: [Deep-Learning-Based Resource Allocation for Transmit Power Minimization in Uplink NOMA IoT Cellular Networks](https://ieeexplore.ieee.org/document/10064048)

<table border="1" style="border-collapse: collapse; width: 100%;">
    <tr>
        <th style="padding: 10px; background-color: #f2f2f2;">Aspect</th>
        <th style="padding: 10px; background-color: #f2f2f2;">Details</th>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Tasks</strong></td>
        <td style="padding: 10px;">Subband Allocation, Power Control</td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>System</strong></td>
        <td style="padding: 10px;">IoT Cellular Networks with K users and N subbands (swapped notation)</td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Objective</strong></td>
        <td style="padding: 10px;">Balance between power minimization and rate constraint satisfaction</td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Method</strong></td>
        <td style="padding: 10px; padding-bottom: 5px;">
            <ul>
                <li>Two-Step Optimization Method with Deep Learning Models</li>
                <li>SubBand Allocation stage based on GA search (genetic algorithm) </li>
                <li>Power   Allocation stage with DNN architecture with unsupervised learning</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Validity</strong></td>
        <td style="padding: 10px; padding-bottom: 5px;">
            <p style="color: green; font-weight: 600; padding-left: 10px; align-items: center; justify-items: center; display: flex">
                &#9989; Valid
            </p>
            <ul>
                <li>The assumptions of the method can be extended to the current constraints.</li>
                <li>The subband allocation step uses a non-optimal method (we cann't assure its optimality) and it's computationaly expensive</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Required <br> Modifications</strong></td>
        <td style="padding: 10px;">
            Adapt the subband allocation stage of the approach for our system.
        </td>
    </tr>
</table>


## **An Energy-Efficient Downlink Resource Allocation In Cellular IoT H-CRANs**

**Paper**: [An Energy-Efficient Downlink Resource Allocation In Cellular IoT H-CRANs](https://ieeexplore.ieee.org/document/10425468)

<table border="1" style="border-collapse: collapse; width: 100%;">
    <tr>
        <th style="padding: 10px; background-color: #f2f2f2;">Aspect</th>
        <th style="padding: 10px; background-color: #f2f2f2;">Details</th>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Tasks</strong></td>
        <td style="padding: 10px;">Subband Allocation, Power Control</td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>System</strong></td>
        <td style="padding: 10px;">NOMA-Based Vehicular Communication Networks</td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Objective</strong></td>
        <td style="padding: 10px;">Maximize the sum rate of vehicular users while ensuring fairness and low latency</td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Method</strong></td>
        <td style="padding: 10px; padding-bottom: 5px;">
            <ul>
                <li>Deep Reinforcement Learning (Deep-Q Learning)</li>
                <li>Learning to dynamically allocate resources and power</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Validity</strong></td>
        <td style="padding: 10px; padding-bottom: 5px;">
            <p style="color: #ff9c04; font-weight: 600; padding-left: 10px; align-items: center; justify-items: center; display: flex">
                &#10071; Partial Valid
            </p>
            <ul>
                <li>Different Type of Networks with lower density NOMA systems</li>
                <li>Allows dynamic allocation to deal with rapid changes, although it may respond uncorrectly to small variations</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Required <br> Modifications</strong></td>
        <td style="padding: 10px;">
            <ul style="list-style-type: '';">
                <li>The <strong>latency-aware RRM</strong> is a good approach that can adapted to factory settings.</li>
                <li>We could consider a model-based reinforcement learning approach to work with better scenarios. </li>
                <li>Requires changes to model hyper-dense and mobile in-X subnetworks settings</li>
            </ul>
        </td>
    </tr>
</table>

## **An Energy-Efficient Downlink Resource Allocation In Cellular IoT H-CRANs**

**Paper**: [An Energy-Efficient Downlink Resource Allocation In Cellular IoT H-CRANs](https://ieeexplore.ieee.org/document/10425468)

<table border="1" style="border-collapse: collapse; width: 100%;">
    <tr>
        <th style="padding: 10px; background-color: #f2f2f2;">Aspect</th>
        <th style="padding: 10px; background-color: #f2f2f2;">Details</th>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Tasks</strong></td>
        <td style="padding: 10px;">Resource Allocation; Network Slicing</td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>System</strong></td>
        <td style="padding: 10px;">NOMA-Based Vehicular Communication Networks</td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Objective</strong></td>
        <td style="padding: 10px;">Maximize the sum rate of vehicular users while ensuring fairness and low latency</td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Method</strong></td>
        <td style="padding: 10px; padding-bottom: 5px;">
            <ul>
                <li>Deep Reinforcement Learning (Deep-Q Learning)</li>
                <li>Learning to dynamically allocate resources and power</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Validity</strong></td>
        <td style="padding: 10px; padding-bottom: 5px;">
            <p style="color: #ff9c04; font-weight: 600; padding-left: 10px; align-items: center; justify-items: center; display: flex">
                &#10071; Partial Valid
            </p>
            <ul>
                <li>Different Type of Networks with lower density NOMA systems</li>
                <li>Allows dynamic allocation to deal with rapid changes, although it may respond uncorrectly to small variations</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Required <br> Modifications</strong></td>
        <td style="padding: 10px;">
            <ul style="list-style-type: '';">
                <li>The <strong>latency-aware RRM</strong> is a good approach that can adapted to factory settings.</li>
                <li>We could consider a model-based reinforcement learning approach to work with better scenarios. </li>
                <li>Requires changes to model hyper-dense and mobile in-X subnetworks settings</li>
            </ul>
        </td>
    </tr>
</table>

## User Subgrouping and Power Control for Multicast Massive MIMO Over Spatially Correlated Channels

**Paper**: [User Subgrouping and Power Control for Multicast Massive MIMO Over Spatially Correlated Channels](https://ieeexplore.ieee.org/abstract/document/9844673?casa_token=RsvjUtHI0yEAAAAA:35DC4HSJgIlSoc9U-XC_ydwzoZWuIt3BCiiTviVXUU7jBLKjT5EdpU0f7Honeb386HdxJmTP_g)

<table border="1" style="border-collapse: collapse; width: 100%;">
    <tr>
        <th style="padding: 10px; background-color: #f2f2f2;">Aspect</th>
        <th style="padding: 10px; background-color: #f2f2f2;">Details</th>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Tasks</strong></td>
        <td style="padding: 10px;">Resource Allocation, Network Slicing</td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>System</strong></td>
        <td style="padding: 10px;">MIMO Network with muticast users</td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Objective</strong></td>
        <td style="padding: 10px;">Optimize resource allocation for network slicing to enhance system performance and user experience</td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Method</strong></td>
        <td style="padding: 10px; padding-bottom: 5px;">
            <ul>
                <li>Deep Reinforcement Learning (DRL)</li>
                <li>Utilizes a DRL-based approach to dynamically allocate resources across network slices</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Validity</strong></td>
        <td style="padding: 10px; padding-bottom: 5px;">
            <p style="color: #ff9c04; font-weight: 600; padding-left: 10px; align-items: center; justify-items: center; display: flex">
                &#10071; Partially Valid
            </p>
            <ul>
                <li>Focuses on network slicing in RAN with Massive MIMO, which differs from in-X subnetwork scenarios</li>
                <li>Employs DRL for resource allocation, which is relevant but may need adaptation for hyper-dense and mobile in-X subnetworks</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td style="padding: 10px;"><strong>Required <br> Modifications</strong></td>
        <td style="padding: 10px;">
            The DRL-based resource allocation approach can be adapted to handle the unique challenges of in-X subnetworks, such as high density and mobility. Incorporating considerations for rapid interference variations and dynamic sub-band allocation may be necessary.
        </td>
    </tr>
</table>


Dynamic Frequency Planning for Autonomous Mobile 6G in-X Subnetworks
Dragonfly approach for resource allocation in industrial wireless networks
Slice-aware Resource Allocation and Admission Control for Smart Factory Wireless Networks
