import numpy as np
''' 
    Code to do forward propagation for a neural network with 2 hidden layer.
    Each hidden layer has two nodes,node_0 and node_1.
    Generate the prediction by multiplying hidden_layer_outputs by weights['output']
    and computing their sum.Use relu function to get the final prediction result.
'''
input_data = np.array([[2, 3],[1,1]])
weights = {'node_0_0': np.array([2, 4]),
           'node_0_1': np.array([4, -5]),
           'node_1_0': np.array([0, 1]),
           'node_1_1': np.array([1, 1]),
           'output': np.array([5, 1])}


def relu(input):
    ''' relu activation function
        Outputs the input if input is +ve , else outputs 0
    '''
    output = max(input, 0)
    return (input)

def predict_with_network(input_data,weights):
    # Calculate node 0 in the first hidden layer
    node_0_0_input = (weights['node_0_0'] *input_data).sum()
    node_0_0_output = relu(node_0_0_input)

    # Calculate node 1 in the first hidden layer
    node_0_1_input = (weights['node_0_1'] *input_data).sum()
    node_0_1_output = relu(node_0_1_input)

    # Put node values into array: hidden_0_outputs
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])

    # Calculate node 0 in the second hidden layer
    node_1_0_input = (weights['node_1_0']*hidden_0_outputs).sum()
    node_1_0_output =relu(node_1_0_input)

    # Calculate node 1 in the second hidden layer
    node_1_1_input = (weights['node_1_1']*hidden_0_outputs).sum()
    node_1_1_output = relu(node_1_1_input)

    # Put node values into array: hidden_1_outputs
    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])

    # Calculate model output: model_output
    model_output = (weights['output']*hidden_1_outputs).sum()

    # Return model_output
    return(model_output)


# Empty list to store prediction results
results = []
for input_data_row in input_data:
    # Append prediction to results
    results.append(predict_with_network(input_data_row, weights))

# Print results
print("Final predicted values are ",results)
