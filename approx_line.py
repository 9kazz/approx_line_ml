import numpy as np

# LAYERS
#--------------------------------------------------------------

class Layer:
    n_out: int           # count of neurons in layer
    acts:  np.ndarray    # neuron's activations in layer
    learning_rate: float # coefficient for changing parameters in gradient descent after one backward pass
    input: np.ndarray    # previous layer's activations

    weights: np.ndarray
    bias:    np.ndarray
    grad_weight: np.ndarray
    grad_bias:   np.ndarray

    def __init__(self, n_in, n_out, learning_rate=0.01):
        rng = np.random.default_rng()

        self.learning_rate = learning_rate
        self.n_out = n_out
        self.acts  = np.zeros(n_out)
        self.input = np.zeros(n_in)

        self.weights = rng.normal(
            0,
            np.sqrt(2 / n_in),
            size=(n_out, n_in)
        )
        self.bias        = np.zeros(n_out)
        self.grad_weight = np.zeros((n_out, n_in))
        self.grad_bias   = np.zeros(n_out)

    # max(0, x)
    @staticmethod
    def relu(vals: np.ndarray) -> np.ndarray:
        assert vals.ndim == 1
        return np.maximum(vals, 0)

    # x<0 --> 0 | x>0 --> 1
    @staticmethod
    def deriv_relu(layer_vals: np.ndarray) -> np.ndarray:
        assert layer_vals.ndim == 1
        return (layer_vals > 0).astype(layer_vals.dtype)

    @staticmethod
    def loss_func(output: np.ndarray, ref: np.ndarray) -> float:
        assert output.ndim == ref.ndim == 1
        assert output.shape == ref.shape
        return np.sum((output - ref) ** 2)

    # calculate layer activations via activations of previous layer
    def go_forward(self, input: np.ndarray, act_func=True) -> np.ndarray:
        assert input.ndim == 1
        assert input.shape[0] == self.weights.shape[1]

        self.input = input
        vals = self.weights @ input + self.bias

        if act_func:
            self.acts = self.relu(vals)
        else:
            self.acts = vals

        return self.acts

    # calculate gradient of loss function in previous layer's values space
    def pass_error_back(self, prev_acts: np.ndarray, error: np.ndarray) -> np.ndarray:
        assert prev_acts.ndim == 1
        assert error.ndim == 1
        assert prev_acts.shape[0] == self.weights.shape[1]
        assert error.shape[0] == self.weights.shape[0]

        return self.weights.T @ error * self.deriv_relu(prev_acts)
        
    # calculate gradient of loss function in layer's weights space
    def calc_weight_grad(self, error: np.ndarray) -> np.ndarray:
        assert error.ndim == 1
        assert error.shape[0] == self.n_out

        self.grad_weight = np.outer(error, self.input)
        return self.grad_weight

    # calculate gradient of loss function in layer's bias space
    def calc_bias_grad(self, error: np.ndarray) -> np.ndarray:
        assert error.ndim == 1
        assert error.shape[0] == self.n_out

        self.grad_bias = error.copy()
        return self.grad_bias

    # update weights and bias after backward pass
    def update_params(self):
        self.weights -= self.learning_rate * self.grad_weight
        self.bias    -= self.learning_rate * self.grad_bias

# NETWORK
#--------------------------------------------------------------

class Network:
    layers: list[Layer]
    n_in:   int
    n_out:  int
    input:  np.ndarray
    output: np.ndarray

    def __init__(self, layer_sizes: list[int], learning_rate: float = 0.01):
        if len(layer_sizes) < 2:
            raise ValueError("Network must contain input and output layers")

        if any(size <= 0 for size in layer_sizes):
            raise ValueError("Layer sizes must be positive")

        self.n_in  = layer_sizes[0]
        self.n_out = layer_sizes[-1]

        self.layers = []

        for idx in range(len(layer_sizes) - 1):
            self.layers.append(
                Layer(layer_sizes[idx], layer_sizes[idx + 1], learning_rate)
            )

        self.input  = np.zeros(self.n_in)
        self.output = np.zeros(self.n_out)

    def n_layers(self) -> int:
        return len(self.layers)
    
    def forward_pass(self) -> np.ndarray:
        result = self.input
        for idx in range(self.n_layers() - 1):
            result = self.layers[idx].go_forward(result)

        result = self.layers[-1].go_forward(result, act_func=False) # no activation function at output layer

        self.output = result
        return result

    def output_error(self, ref: np.ndarray) -> np.ndarray:
        assert ref.ndim == 1 and ref.shape == self.layers[-1].acts.shape
        output_layer = self.layers[-1]
        return 2 * (output_layer.acts - ref) # no activation function at output layer

    def backward_pass(self, ref: np.ndarray) -> None:
        assert ref.ndim == 1
        error = self.output_error(ref)

        for idx in range(self.n_layers() - 1, -1, -1):
            layer = self.layers[idx]

            layer.calc_weight_grad(error)
            layer.calc_bias_grad(error)

            if idx > 0:
                error = layer.pass_error_back(self.layers[idx-1].acts, error)

    def update_layers(self) -> None:
        for layer in self.layers:
            layer.update_params()


#==============================================================

def generate_dataset(train_size: int, control_size: int, noise_std: float, dots_cnt: int = 7
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng()

    x_points = np.arange(dots_cnt, dtype=float)  # x = 0, 1, ..., dots_cnt-1

    def generate_samples(size: int) -> tuple[np.ndarray, np.ndarray]:
        inputs = np.zeros((size, dots_cnt))
        refs   = np.zeros((size, 2))

        for i in range(size):
            a = rng.uniform(-1, 1)
            b = rng.uniform(-1, 1)

            noise = rng.normal(0, noise_std, size=dots_cnt)

            y_points = a * x_points + b + noise

            inputs[i] = y_points

            # least-squares line
            x_mean = np.mean(x_points)
            y_mean = np.mean(y_points)

            a_fit = np.sum((x_points - x_mean) * (y_points - y_mean)) / np.sum((x_points - x_mean) ** 2)
            b_fit = y_mean - a_fit * x_mean

            refs[i] = np.array([a_fit, b_fit])

        return inputs, refs

    train_inputs, train_refs     = generate_samples(train_size)
    control_inputs, control_refs = generate_samples(control_size)

    return train_inputs, train_refs, control_inputs, control_refs

def train(network: Network, train_inputs: np.ndarray, train_refs: np.ndarray) -> None:
    print("=" * 26, "TRAINING", "=" * 26)
    
    shuffled_set = np.arange(train_set)
    rng = np.random.default_rng()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        rng.shuffle(shuffled_set)

        for idx in range(len(shuffled_set)):
            network.input = train_inputs[idx]

            output = network.forward_pass()
            epoch_loss += Layer.loss_func(output, train_refs[idx])

            network.backward_pass(train_refs[idx])
            network.update_layers()

        epoch_loss /= train_set

        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1:3d}: train loss = {epoch_loss:.6f}")
    
    print("=" * 60)
    print()

def validation(network: Network, control_inputs: np.ndarray, control_refs: np.ndarray) -> tuple[float, float, float, list[np.ndarray]]:
    total_loss = 0.0
    a_error = 0.0
    b_error = 0.0

    predictions = []

    for idx in range(control_set):
        network.input = control_inputs[idx]

        output = network.forward_pass()
        ref = control_refs[idx]

        predictions.append(output.copy())

        total_loss += Layer.loss_func(output, ref)

        a_error += abs(output[0] - ref[0])
        b_error += abs(output[1] - ref[1])

    mean_loss = total_loss / control_set
    mean_a_error = a_error / control_set
    mean_b_error = b_error / control_set

    # statistic
    stat(mean_loss, mean_a_error, mean_b_error, predictions)
    return mean_loss, mean_a_error, mean_b_error, predictions

def stat(mean_loss, mean_a_error, mean_b_error, predictions: list[np.ndarray]) -> None:
    print("=" * 20, "CONTROL SET RESULTS", "=" * 19)
    
    print(f"Samples:       {control_set}")
    print(f"Mean loss:     {mean_loss:.6f}")
    print(f"Mean error a:     {mean_a_error:.6f}")
    print(f"Mean error b:     {mean_b_error:.6f}")

    print("=" * 60)    
    print()

def print_predictions(predictions: list[np.ndarray], cnt: int = 10) -> None:
    print("=" * 25, "PREDICTIONS", "=" * 24)

    print(
        f"{'idx':>4} | "
        f"{'a ref':>9} {'a pred':>9} {'a err %':>9} | "
        f"{'b ref':>9} {'b pred':>9} {'b err %':>9}"
    )
    print("-" * 82)

    for idx in range(min(cnt, control_set)):
        pred = predictions[idx]
        ref  = control_refs[idx]

        a_err = abs(pred[0] - ref[0]) / max(abs(ref[0]), 1e-8) * 100
        b_err = abs(pred[1] - ref[1]) / max(abs(ref[1]), 1e-8) * 100

        print(
            f"{idx:4d} | "
            f"{ref[0]:9.4f} {pred[0]:9.4f} {a_err:9.2f} | "
            f"{ref[1]:9.4f} {pred[1]:9.4f} {b_err:9.2f}"
        )

    print("=" * 82)
    print()


# START
#===============================================================
# approximate line parameters (y = ax + b) for given dot set

train_set   = 1000
control_set = 200
noise_std   = 0.2
n_epochs    = 100
input_dots  = 7

train_inputs, train_refs, control_inputs, control_refs = generate_dataset(train_set, control_set, noise_std, input_dots)

network = Network([input_dots, 5, 3, 2])

train(network, train_inputs, train_refs)
mean_loss, mean_a_error, mean_b_error, predictions = validation(network, control_inputs, control_refs)
print_predictions(predictions, 20)