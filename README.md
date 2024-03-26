# ChatBot Module

## Overview

This ChatBot module utilizes natural language processing (NLP) techniques and deep learning with PyTorch to create a conversational agent. It takes a dataset from `intents.json`, tokenizes and stems the data, and then trains a neural network using PyTorch's `torch.nn` module.

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Training the ChatBot

1. Ensure you have installed the required dependencies.
2. Run `train.py` to train your own ChatBot module. Aim to minimize the loss during training.
3. Monitor the training progress to ensure the loss is decreasing over time.

```bash
python train.py
```

### Chatting with the ChatBot

1. After training your ChatBot, run `main.py` to start chatting with it using your own module.
2. Make sure to specify the correct model name when running `main.py`.

```bash
python main.py
```

## Additional Notes

- The `intents.json` file contains the dataset used for training the ChatBot. Ensure it is structured correctly with intent tags and corresponding patterns.
- Adjust the hyperparameters and architecture of the neural network in the `NeuralNet` class within `module.py` to optimize performance.
- Experiment with different preprocessing techniques and model architectures to improve the ChatBot's conversational abilities.

## Contributors

- MerlynCoding
