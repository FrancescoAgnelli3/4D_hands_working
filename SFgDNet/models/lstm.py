"""LSTM-based baseline built on the generic TimeSpaceModel."""

from .baselines import TimeSpaceModel


class LSTMForecast(TimeSpaceModel):
    """Temporal-only LSTM baseline applied independently to each node."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        horizon: int,
        num_layers: int = 1,
        num_nodes: int = 42,
    ) -> None:
        super().__init__(
            input_size=input_channels,
            n_nodes=num_nodes,
            horizon=horizon,
            hidden_size=hidden_channels,
            output_size=output_channels,
            t_layers=num_layers,
            gnn_layers=0,
            gnn_type=None,
            temporal_type="lstm",
        )
