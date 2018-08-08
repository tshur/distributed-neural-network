from model.unit import INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER

class Layer:
    """A base class for defining a particular Layer in the network.

    Simply put, a Layer consists of a list of Units and is interconnected
    by adjacent Layers. For most of the Layer functions, the Layer simply
    calls the corresponding Unit functions for each Unit in the Layer."""

    def __init__(self, base_unit, previous_layer, num_units, activation, layer_type):
        if isinstance(previous_layer, int):  # Signifies input layer
            input_units = previous_layer
            previous_layer = None
        else:
            input_units = previous_layer.num_units

        self.activation = activation()  # Given ActivationFunction
        self.base_unit = base_unit  # e.g., NonlinearUnit
        self.units = [
            self.base_unit(i, input_units + 1, None, layer_type, self.activation)
            for i in range(num_units)
        ]  # A Layer is a list of units
        self.num_units = num_units
        self.layer_type = layer_type  # i.e., INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER

        if previous_layer:
            previous_layer.connect_to(self)  # Connect previous layer to this one

    def connect_to(self, next_layer):
        """Connect the Layer to the given next_layer by connecting unit outputs."""
        if not self.layer_type == OUTPUT_LAYER:
            for unit in self.units:
                unit.outputs = next_layer.units

    def prepare(self):
        """Prepare the network by clearing cached inputs and error terms."""
        for unit in self.units:
            unit.prepare()

    def propagate_input(self, features):
        """Feed the input features forward through the Layer."""
        for unit in self.units:
            if unit.is_input_unit():
                unit.feed_input(features)
            unit.output()

    def propagate_error(self, target):
        """Propagate the error backwards through the Layer (backpropagation)."""
        for unit in self.units:
            unit.calculate_error(target)

    def propagate_attention(self):
        """Propagate attention backwards through the Layer. Experimental."""
        for unit in self.units:
            unit.calculate_attention()

    def update_weights(self, sample, learning_rate, momentum=0.3):
        """Update the Layer weights according to given momentum and learning_rate."""
        for unit in self.units:
            for i in range(unit.dim):
                # Compute the unit's delta_weight based on cached unit information
                delta = (learning_rate * unit.error * unit.x[i] +
                              momentum * unit.delta_prev[i])
                unit.w[i] += delta  # Update weight
                unit.delta_prev[i] = delta  # Stored for momentum calculation

    def get_weight_updates(self, sample, learning_rate, network_weight_updates):
        """Gets the calculated weight updates without modifying the actual Layer.

        Same as `update_weights()` except modifies `network_weight_updates`,
        a deque, instead of the actual network weights.
        """
        initializing = True if network_weight_updates[0] is None else False  # If first run

        for unit in self.units:
            for i in range(unit.dim):
                delta = learning_rate * unit.error * unit.x[i]  # weight delta
                if initializing:
                    network_weight_updates.append(delta)
                else:
                    network_weight_updates.append(delta + network_weight_updates.popleft())

    def update_from_stored_changes(self, network_weight_updates):
        """Updates the Layer from a deque() of given weight updates (deltas)."""
        for unit in self.units:
            for i in range(unit.dim):
                unit.w[i] += network_weight_updates.popleft()

    def set_weights(self, network_weights):
        """Sets the network weights based on a deque() of given weights.

        This is done in order of Layers from input to output, and for
        each Layer, in order of Units from lowest index to highest.
        """
        for unit in self.units:
            for i in range(unit.dim):
                unit.w[i] = network_weights.popleft()

    def get_weights(self, network_weights):
        """Fills `network_weights` with the current weights.

        This is done in order of Layers from input to output, and for
        each Layer, in order of Units from lowest index to highest.
        """
        for unit in self.units:
            for i in range(unit.dim):
                network_weights.append(unit.w[i])

    def save_weights(self):
        """Temporarily stores the current Layer weights in each Unit.

        This weights will not be modified or used for future training,
        but reserved for later finalization.
        """
        for unit in self.units:
            unit.save_weights()

    def finalize_weights(self):
        """Sets previously saved weights as the current weights of the network.

        These saved weights must have been stored earlier with the
        `save_weights()` method.
        """
        for unit in self.units:
            unit.finalize_weights()