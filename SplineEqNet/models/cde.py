import torch
import torch.nn as nn

######################
# So you want to train a Neural CDE model?
# Let's get started!
######################

import math
import torch
import torchcde


######################
# A CDE model looks like
#
# z_t = z_0 + \int_0^t f_\theta(z_s) dX_s
#
# Where X is your data and f_\theta is a neural network. So the first thing we need to do is define such an f_\theta.
# That's what this CDEFunc class does.
# Here we've built a small single-hidden-layer neural network, whose hidden layer is of width 128.
######################
class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()
        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


######################
# Next, we need to package CDEFunc up into a model that computes the integral.
######################
class NeuralCDE_single(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, interpolation="cubic"):
        super(NeuralCDE_single, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation

    def forward(self, coeffs):
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        ######################
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              t=X.interval)

        z_T = z_T[:, 1]
        pred_y = self.readout(z_T)
        return pred_y
    
class NeuralCDE(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, interpolation="cubic", horizon=10):
        super(NeuralCDE, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.interpolation = interpolation
        self.horizon = horizon
        self.single_step_model = NeuralCDE_single(input_channels, hidden_channels, output_channels, interpolation)

    def forward(self, initial_path):
        """
        initial_path: Tensor of shape (batch, window, input_channels)
        Returns: predictions of shape (batch, horizon, output_channels)
        """

        # Initial path coefficients
        pred_y_seq = []

        current_path = initial_path  # shape (B, W, C)
        for _ in range(self.horizon):

            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(current_path)

            # Predict next step
            pred_y = self.single_step_model(coeffs)  # shape (B, output_channels)
            pred_y_seq.append(pred_y.unsqueeze(1))  # shape (B, 1, output_channels)

            # Prepare next input path (append pred_y and remove oldest timestep)
            # We'll assume predicted output has same dimensions as input
            pred_y_expanded = pred_y.unsqueeze(1)  # shape (B, 1, C)
            current_path = torch.cat([current_path[:, 1:], pred_y_expanded], dim=1)  # shape (B, W, C)

        # Concatenate predictions along time dimension
        pred_y_seq = torch.cat(pred_y_seq, dim=1)  # shape (B, horizon, output_channels)
        return pred_y_seq
    

class NeuralCDE_fast(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, horizon = 10, interpolation="cubic"):
        super(NeuralCDE_fast, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels*horizon)
        self.interpolation = interpolation
        self.output_channels = output_channels
        self.horizon = horizon

    def forward(self, coeffs):
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(coeffs)
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        ######################
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              t=X.interval)

        z_T = z_T[:, 1]
        pred_y = self.readout(z_T).reshape(-1, self.horizon, self.output_channels)
        return pred_y

class MultiNeuralCDE(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, interpolation="cubic", horizon=10, fast = True):
        super(MultiNeuralCDE, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.interpolation = interpolation
        self.horizon = horizon

        if fast:
            self.cde_model = NeuralCDE_fast(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                interpolation=interpolation,
                horizon=horizon
            )
        else:
            self.cde_model = NeuralCDE(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                interpolation=interpolation,
                horizon=horizon
            )

    def forward(self, x, edges):
        """
        x: Tensor of shape (B, N, T, D)
        Returns: Tensor of shape (B, N, F, D)
        """
        B, T, N, D = x.shape

        # Merge B and N into one dimension
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * N, T, D)

        # Run through NeuralCDE (which expects shape (B*N, T, D))
        out = self.cde_model(x_reshaped)  # shape: (B*N, F, D)

        # Reshape back to (B, N, F, D)
        out = out.reshape(B, N, self.horizon, self.output_channels).permute(0, 2, 1, 3)
        return out
