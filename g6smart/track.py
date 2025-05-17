import matplotlib.pyplot as plt
import wandb.sdk
from IPython.display import clear_output

import wandb

# Force Login on Module Load
wandb.login(force=True)
Run = wandb.sdk.wandb_run.Run
def setup_wandb(name: str, group: str, config: dict[str, float], id: str = None) -> Run:
    """Initialize a WandDB with a given set of experiment properties. If there was any running run,
    it finishes it quietly.

    Args:
        name (str): name of the experiment
        group (str): experiment group. For instance, this can be used to differentiate between model architectures.
        config (dict[str, float]): training configuration. It includes training parameters and architecture parameters.
        id (str, optional): Experiment ID. If it set, a previous experiment will be resumed. Defaults to None.

    Returns:
        Run: experiment wandb run object. It allows to access experiment metadata, such as, the id.
    """
    try:
       wandb.finish(quiet = True)
    except Exception:
       pass

    config['name'] = name
    return wandb.init(
        project="6GSmartRRM",
        name   = name,
        id     = id,
        group  = group,
        config = config,
        resume = "allow" if id is None else "must"
    )

def real_time_plot(*metrics):
    """Creates a real-time visualization in a notebook cell. Each time, it is called it clears the previous plot and updates it.
    """
    names = ['training', 'validation']
    assert len(metrics) % 2 == 0, "A odd pair of metrics is required"
    clear_output(wait=True)  # Clear the previous plot

    fig, ax = plt.subplots(1, 2, figsize=(14, 4))  # Two subplots, stacked vertically
    # Plot loss
    for i, loss in enumerate(metrics[:len(metrics) // 2]):
      ax[0].plot(loss, label = f"loss: {names[i]}")
    ax[0].set_title('Real-Time Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    # Plot metric (e.g., SINR or accuracy)
    for i, metric in enumerate(metrics[len(metrics) // 2:]):
      ax[1].plot(metric, label = f"loss: {names[i]}")
    ax[1].set_title('Real-Time Metric')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Bit Rate (Mbps)')
    ax[1].legend()

    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.show()
