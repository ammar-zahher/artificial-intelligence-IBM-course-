import numpy as np

weight = np.round(np.random.uniform(size=6), decimals=2)
biases = np.round(np.random.uniform(size=3), decimals=2)
print("weights:", weight)
print("biases:", biases)
x_1 = 0.5
x_2 = 0.85
print("x1 is {} and x2 is {}".format(x_1, x_2))
z_first_node = (x_1 * weight[0]) + (x_2 * weight[2]) + biases[0]
print(f'{"="*40}z_first node{"="*40}')
print(
    "The weighted sum of the inputs at the first node in the hidden layer is {}".format(
        z_first_node
    )
)
z_second_node = (x_1 * weight[1]) + (x_2 * weight[3]) + biases[1]
print(f'{"="*40}z_second node{"="*40}')
print(
    "The weighted sum of the inputs at the second node in the hidden layer is {}".format(
        z_second_node
    )
)
a_first = 1.0 / (1.0 + np.exp(-z_first_node))
print(f'{"="*40}a_11 node{"="*40}')
print(
    "The activation of the first node in the hidden layer is {}".format(
        np.round(a_first, decimals=2)
    )
)
a_second = 1.0 / (1.0 + np.exp(-z_second_node))
print(f'{"="*40}a_12 node{"="*40}')
print(
    "The activation of the second node in the hidden layer is {}".format(
        np.round(a_second, decimals=2)
    )
)
z_output_node = (a_first * weight[4]) + (a_second * weight[5]) + biases[2]
print(f'{"="*40}z_output node{"="*40}')
print(
    "The weighted sum of the inputs at the output node is {}".format(
        np.round(z_output_node, decimals=2)
    )
)
a_output = 1.0 / (1.0 + np.exp(-z_output_node))
print(f'{"="*40}a_output node{"="*40}')
print("The activation of the output node is {}".format(np.round(a_output, decimals=2)))
