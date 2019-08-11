import numpy as np
''' 
    Code to do forward propagation for a neural network with 1 hidden layer.
    The hidden layer has two nodes,node_0 and node_1.
    Calculate the value in node 0 by multiplying input_data by its weights['node_0'] 
    and computing their sum. Use relu function to get the final result. 
    This is the 1st node in the hidden layer.
    Calculate the value in node 1 using input_data and weights['node_1'].
    Use relu function to get the final result.
    This is the 2nd node in the hidden layer.
    Put the hidden layer values into an array.
    Generate the prediction by multiplying hidden_layer_outputs by weights['output']
    and computing their sum.Use relu function to get the final result.
'''
input_data = np.array([[2, 3],[3,2]])
weights = {'node_0': np.array([1, 1]),
           'node_1': np.array([-1, 1]),
           'output': np.array([2, -1])}


def relu(input):
    ''' relu activation function
        Outputs the input if input is +ve , else outputs 0
    '''
    output = max(input, 0)
    return (input)

def predict_with_network(input_data_row, weights):
    '''
    Function to calculate the predicted output for the network
    :param input_data_row: Contains the values of the initial nodes
    :param weights: Weights for all the layers
    :return: Returns the predicted output value
    '''

    # Calculate node 0 value
    node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_input)
    print("Value for node_0 of hidden layer is ",node_0_output)

    # Calculate node 1 value
    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)
    print("Value for node_1 of hidden layer is ",node_1_output)


    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])

    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = relu(input_to_final_layer)

    # Return model output
    return (model_output)


# Empty list to store prediction results
results = []
for input_data_row in input_data:
    # Append prediction to results
    results.append(predict_with_network(input_data_row, weights))

# Print results
print("Final predicted values are ",results)
