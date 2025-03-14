def sec_to_matrix(array_data, t, x):
    return array_data.view(t, x)

def matrix_to_sec(matrix_data):
    return matrix_data.view(-1)