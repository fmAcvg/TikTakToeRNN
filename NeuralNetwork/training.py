from numpy import np

global current_Loss



def backpropagate(model, model_gradients, learning_rate):
    for i, layer in enumerate(model.layers):
        for j, neuron in enumerate(layer.neurons):
            # Update weights and biases using computed gradients
            neuron.weights -= learning_rate * model_gradients.get(f'layer_{i}_neuron_{j}_weights', 0)
            neuron.bias -= learning_rate * model_gradients.get(f'layer_{i}_neuron_{j}_bias', 0)
    return model_gradients
    


def train_on_data_set(model, current_postion, correct_move, learning_rate):
    def loss_function(predicted, correct):
        return -np.sum(correct * np.log(predicted + 1e-15))  # Cross-entropy loss with epsilon for stability   

    # Forward pass
    # Get model predictions
    # Assuming model has a method 'predict' that returns probabilities
    predicted_move = model.predict(current_postion) 
    loss = loss_function(predicted_move[1], correct_move)



            