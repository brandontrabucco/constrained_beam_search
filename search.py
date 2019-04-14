"""Author: Brandon Trabucco, Copyright 2019
Implements a constrained beam search that supprts multiple variables.
TensorFlow."""


import tensorflow as tf
import numpy as np


class Beam(object):
    """Engine for storing the current results of the beam search."""

    def __init__(self, state, logits):
        """Assign the stateful contents of the beam search.
        Args:
            state: a data structure of tensors with various shapes and dtypes
            logits: a float32 tensor of batch_size, beam_size"""
        self.state = state
        self.logits = logits

    def update(self, proposal_function, constraint_function, gather_function):
        """Proposes next states for the beam search.
        Args:
            proposal_function: a function of the state
            constraint_function: a boolean function of next_states
            gather_function: a functions of the indices and next_states
        Returns: 
            beam: the next beam in a constrained beam search"""
        next_states, update = proposal_function(self.state)
        next_logits = tf.expand_dims(self.logits, 2) + tf.expand_dims(update, 1)
        values, indices = tf.math.top_k(tf.reshape(tf.exp(next_logits) * constraint_function(next_states), [
            tf.shape(next_logits)[0], tf.shape(next_logits)[1] * tf.shape(next_logits)[2]]), k=8)
        return Beam(gather_function(indices, next_states), values)


def count_backend(ids, lengths):
    """NumPy backend for the count function.
    Args:
        ids: a tensor with last axis a variable length sequence
        lengths: a tensor containint the lengths of ids
    Returns:
        mask: a binary mask zero when any element appears twice"""
    flat_ids = np.reshape(ids, [-1, ids.shape[-1]])
    flat_lengths = np.reshape(lengths, [-1, lengths.shape[-1]])
    mask = np.ones(flat_ids.shape[0])
    for i in range(flat_ids.shape[0]):
        unique, counts = np.unique(ids[i, :flat_lengths[i]], return_counts=True)
        if any(counts > 1):
            mask[i] = 0.0
    return mask.reshape(ids.shape[:-1])


def make_count_constraint(transform):
    """Returns zero when any element appears twice
    Args:
        transform: a function mapping state to ids, lengths
    Returns:
        constraint: a function of the state"""
    return lambda states: tf.py_function(count_backend, transform(states), tf.float32)


def repeat_backend(ids, lengths):
    """NumPy backend for the repeat function.
    Args:
        ids: a tensor with last axis a variable length sequence
        lengths: a tensor containint the lengths of ids
    Returns:
        mask: a binary mask zero when any element is repeated"""
    flat_ids = np.reshape(ids, [-1, ids.shape[-1]])
    flat_lengths = np.reshape(lengths, [-1, lengths.shape[-1]])
    mask = np.ones(flat_ids.shape[0])
    for i in range(flat_ids.shape[0]):
        sequence = ids[i, :flat_lengths[i]]
        if any(np.equal(sequence[:-1], sequence[1:])):
            mask[i] = 0.0
    return mask.reshape(ids.shape[:-1])


def make_repeat_constraint(transform):
    """Returns zero when any element appears twice
    Args:
        transform: a function mapping state to ids, lengths
    Returns:
        constraint: a function of the state"""
    return lambda states: tf.py_function(repeat_backend, transform(states), tf.float32)


def make_proposal_function(cell, transform, reverse_transform):
    """Builds a proposal function using a keras cell
    Args:
        cell: a keras call the operates on a sequence
        transform: a function mapping states to keras cell state and inputs
        reverse_transform: a function mapping to next states
    Returns:
        proposal: a function of the state"""
    return lambda states: reverse_transform(states, *cell(transform(states)))