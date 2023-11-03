import numpy as np

def postprocess(out, labels, outputNames):
    # DECLARE OUTPUTS DICTIONARY
    outputs = dict()

    # Compute the softmax probabilities without torch
    probabilities = np.exp(out[outputNames[0]]) / np.sum(np.exp(out[outputNames[0]]), axis=1, keepdims=True)

    # Add the probabilities and predicted label to the outputs dictionary
    outputs['output'] = probabilities.tolist()

    # RETURN OUTPUTS
    return outputs
