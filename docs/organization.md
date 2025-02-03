## Experiments
### Naming Convention

The naming convention for experiments follows the format:

```{prototype}-{task}-{architecture}-{version}-{hyper}```

Components:

* **Prototype**: Prototype Name or ID

* **Task**: Specifies the task performed.
    - Possible values:
        - alloc: Allocation task
        - power: Power task
        - both: Both tasks

* **Architecture**: Describes the type of architecture used.
    - Possible values:
        - dnn: Deep Neural Network
        - mixed: Mixed architectures
        - gnn: Graph Neural Network

* **Version**: Indicates the version of the experiment in the format `{major}-{minor}`.
    * Major: Refers to significant changes such as architectural updates or new proposals.
    * Minor: Refers to minor modifications, such as changes in the number of neurons, layers, or regularization techniques.

* **Hyper**: Specifies whether hyperparameter optimization was applied.
    * Possible values:
        * hyper: Hyperparameter optimization applied
        * base: Base configuration without optimization

## Timeline (Plazo)
### Prototype Development

* Publication Methods: 2 weeks
* Custom Methods: 2 weeks