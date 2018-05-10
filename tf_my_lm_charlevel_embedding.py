import gzip
import math
import os
import pickle
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn  # rnn stuff temporarily in contrib, moving back to code in TF 1.1

from NMT_commons import read_data_files_as_chars, convert_to_one_line, build_char_dataset, minibatch_generator, negativeLogProb, perplexity, make_zip_results, sample_from_probabilities, minute

tf.set_random_seed(0)
outputFileName = 'linux_embedding.log'
sys.stdout = open(outputFileName, 'w')

# configs
keep_prop = .8  # of course for training .. give 1 at validation
sequence_length = 30
batch_size = 200
embedding_size = 50
vocabulary_size = -1  # will be initialized by the text encoder
internal_state_size = 512
stacked_layers = 3
learning_rate = 0.001  # fixed learning rate for optimizer
grad_clip = 10

validation_steps = 1000
num_epochs = 10000

log_epoch_every_mins = 3
validation_every_mins = 5
chechpoint_every_mins = 15

# data_root_path = './tuts/Martin RNN/shakespeare/**/*.txt'
data_root_path = "linux.txt"

concat_text = read_data_files_as_chars(data_root_path)
data_length = len(concat_text)
print("data length in chars ", data_length)
print("-" * 100 + '\n', "sample data", convert_to_one_line(''.join(concat_text[:500])))
print("-" * 100 + '\n', ''.join(concat_text[:500]))

encoded_data, counts, dictionary, reverse_dictionary = build_char_dataset(concat_text)
train, validation = encoded_data[:data_length - data_length // 10], encoded_data[data_length - data_length // 10:]

# noinspection PyRedeclaration
vocabulary_size = len(counts)

print("model hyperparams", {
    'keep_prop': keep_prop,
    'sequence_length': sequence_length,
    'batch_size': batch_size,
    'vocabulary_size': vocabulary_size,
    'embedding_size': embedding_size,
    'internal_state_size': internal_state_size,
    'stacked_layers': stacked_layers,
    'learning_rate': learning_rate,
    'grad_clip': grad_clip, }
      )

with gzip.open("reverse_dictionary.pkl.gz", 'w') as f:
    pickle.dump(reverse_dictionary, f)

validation_generator = minibatch_generator(validation, nb_epochs=1, gen_batch_size=1, gen_seq_len=1)

graph = tf.Graph()
with graph.as_default():
    keep_prop_tf = tf.placeholder(tf.float32, name='keep_prop_tf')  # , name='keep_prop_tf'
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name='embeddings')

    softmax_weights = tf.Variable(tf.truncated_normal([internal_state_size, vocabulary_size], stddev=1.0 / math.sqrt(internal_state_size)), name='weights')
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]), name='biases')

    # repeat it stacked_layers
    dropcells = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(internal_state_size), output_keep_prob=keep_prop_tf) for _ in range(stacked_layers)]
    multi_cell = rnn.MultiRNNCell(dropcells, state_is_tuple=True)

    with tf.variable_scope('train'):
        '''inputs'''
        batch_seq_input = tf.placeholder(tf.int32, [batch_size, sequence_length])  # [ batch_size, sequence_length ] --- None will share the graph between validation and training
        embed_batch_seq_input = tf.nn.embedding_lookup(embeddings, batch_seq_input)  # [ batch_size, sequence_length,embedding_size ] instead of one hot vector
        '''expected outputs = same sequence shifted by 1 since we are trying to predict the next character'''
        batch_seq_labels = tf.placeholder(tf.uint8, [batch_size, sequence_length])  # [ batch_size, sequence_length ]
        # one_hot_batch_seq_labels = tf.one_hot(batch_seq_labels, vocabulary_size, 1.0, 0.0)  # [ batch_size, sequence_length, vocabulary_size ] # i will use sparse cross entropy

        # When using state_is_tuple=True, you must use multicell.zero_state
        # to create a tuple of  placeholders for the input states (one state per layer).
        # When executed using session.run(zerostate), this also returns the correctly
        # shaped initial zero state to use when starting your training loop.
        zero_state_train = multi_cell.zero_state(batch_size, dtype=tf.float32)

        print(zero_state_train)
        out_states_train, hidden_state_train = tf.nn.dynamic_rnn(multi_cell, embed_batch_seq_input, dtype=tf.float32, initial_state=zero_state_train)
        # out_states_train: [ batch_size, sequence_length, internal_state_size ]
        # hidden_state_train:  [ batch_size, internal_state_size*stacked_layers ] # this is the last state in the sequence

        # Softmax layer implementation:
        # Flatten the first two dimension of the output [ batch_size, sequence_length, vocabulary_size ] => [ batch_size x sequence_length, vocabulary_size ]
        # then apply softmax readout layer. This way, the weights and biases are shared across unrolled time steps.
        # From the readout point of view, a value coming from a cell or a minibatch is the same thing

        out_states_flattened_train = tf.reshape(out_states_train, [-1, internal_state_size])  # [ batch_size x sequence_length, internal_state_size ]
        logits_train = tf.nn.xw_plus_b(out_states_flattened_train, softmax_weights, softmax_biases)  # [ batch_size x sequence_length, internal_state_size >>> project on vocabulary_space ]

        batch_seq_softmax_train = tf.nn.softmax(logits_train)  # [ batch_size x sequence_length, vocabulary_size ]
        batch_seq_pred_train = tf.reshape(tf.argmax(batch_seq_softmax_train, 1), [batch_size, -1])  # [ batch_size, sequence_length ]

        # labels
        labels_flattened = tf.cast(tf.reshape(batch_seq_labels, [-1]), dtype=tf.int32)  # [ batch_size x sequence_length ] .... dont leave it int8

        # optimization
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train, labels=labels_flattened)  # [ batch_size x sequence_length ] vs  [ batch_size x sequence_length, vocabulary_size ] so  losses are [ batch_size x sequence_length ]
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # returns grads_and_vars is a list of tuples [(gradient, variable)]
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        # check this
        # zip([(2,3),(4,5),(4,5),(4,5),(4,5)])  will be <zip at 0xb3de450648> and as list [((2, 3),), ((4, 5),), ((4, 5),), ((4, 5),), ((4, 5),)] !!

        # a,b=zip(*[(2,3),(4,5),(4,5),(4,5),(4,5)]) will be ((2, 4, 4, 4, 4),(3, 5, 5, 5, 5))
        # the same as so * is unpack operator
        # a,b=zip((2,3),(4,5),(4,5),(4,5),(4,5))

        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=grad_clip)
        # computes the global norm and then shrink all gradients with the same ratio clip_norm/global_norm only if global_norm > clip_norm

        train_step = optimizer.apply_gradients(list(zip(gradients, variables)))  # zip to relate variable to gradient as list of tuples

        # stats for display
        mean_loss = tf.reduce_mean(loss)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(batch_seq_labels, tf.cast(batch_seq_pred_train, tf.uint8)), tf.float32))
        loss_summary = tf.summary.scalar("batch_loss", mean_loss)
        acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
        summaries = tf.summary.merge([loss_summary, acc_summary])

    with tf.variable_scope('valid'):  # batch size is always 1
        '''inputs'''
        val_seq_input = tf.placeholder(tf.int32, [1, None], name='val_seq_input')  # [ 1, dynamic sequence_length ] --- None will share the graph between validation and training
        embed_val_seq_input = tf.nn.embedding_lookup(embeddings, val_seq_input)  # [ 1, sequence_length,embedding_size ] instead of one hot vector

        zero_state_valid = multi_cell.zero_state(1, dtype=tf.float32)

        out_states_valid, hidden_state_valid = tf.nn.dynamic_rnn(multi_cell, embed_val_seq_input, dtype=tf.float32, initial_state=zero_state_valid)
        # out_states_valid: [ 1, sequence_length, internal_state_size ]
        # hidden_state_valid:  [ 1, internal_state_size*stacked_layers ] # this is the last state in the sequence

        hidden_state_valid = tf.identity(hidden_state_valid, name='hidden')  # just to give it a name

        # Softmax layer implementation:
        # Flatten the first two dimension of the output [ 1, sequence_length, vocabulary_size ] => [ 1 x sequence_length, vocabulary_size ]
        # then apply softmax readout layer. This way, the weights and biases are shared across unrolled time steps.
        # From the readout point of view, a value coming from a cell or a minibatch is the same thing
        out_states_flattened_valid = tf.reshape(out_states_valid, [-1, internal_state_size])  # [ 1 x sequence_length, internal_state_size ]

        logits_valid = tf.nn.xw_plus_b(out_states_flattened_valid, softmax_weights, softmax_biases)  # [ 1 x sequence_length, vocabulary_size ] >>> project on vocabulary space

        final_predictions = tf.nn.softmax(logits_valid, name="final_predictions")  # [ batch_size, sequence_length ]

    saver = tf.train.Saver(max_to_keep=1)

# Init Tensorboard stuff. This will save Tensorboard information into a different
# folder at each run named 'log/<timestamp>/'.
timestamp = str(math.trunc(time.time()))
summary_writer = tf.summary.FileWriter("log/" + timestamp + "-training")
validation_writer = tf.summary.FileWriter("log/" + timestamp + "-validation")

# Init for saving models. They will be saved into a directory named 'checkpoints'.
# Only the last checkpoint is kept.
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")

import time

execution_start = time.time()

checkpoint_last = time.time()
epoch_log_last = time.time()
summary_last = time.time()

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    # training loop
    istate = sess.run(zero_state_train)  # initial zero input state (a tuple)
    v_istate = sess.run(zero_state_valid)  # initial zero input state (a tuple)
    step = 0

    for batch, label, epoch in minibatch_generator(train, nb_epochs=num_epochs, gen_batch_size=batch_size, gen_seq_len=sequence_length):
        current_time = time.time()

        # train on one minibatch
        feed_dict = {batch_seq_input: batch, batch_seq_labels: label, keep_prop_tf: keep_prop}
        # This is how you add the input state to feed dictionary when state_is_tuple=True.
        # zerostate is a tuple of the placeholders for the NLAYERS=3 input states of our
        # multi-layer RNN cell. Those placeholders must be used as keys in feed_dict.
        # istate is a tuple holding the actual values of the input states (one per layer).
        # Iterate on the input state placeholders and use them as keys in the dictionary
        # to add actual input state values.
        for i, v in enumerate(zero_state_train):
            feed_dict[v] = istate[i]

        if (current_time - summary_last) > validation_every_mins * minute:
            summary_last = time.time()

            _, predictions, softmax_probabilities, ostate, smm, ls, acc = sess.run([train_step, batch_seq_pred_train, batch_seq_softmax_train, hidden_state_train, summaries, mean_loss, accuracy], feed_dict=feed_dict)
            t_perplexity = float(perplexity(softmax_probabilities, np.reshape(label, [-1])))
            print('step : %d epoch : %d Minibatch perplexity: %.2f' % (step, epoch, t_perplexity))
            print("train :{}:loss {},acc{}".format(epoch, ls, acc))

            # save training data for Tensorboard
            summary_writer.add_summary(smm, step)
            summary_per = tf.Summary(value=[
                tf.Summary.Value(tag="perplexity", simple_value=t_perplexity),
            ])
            summary_writer.add_summary(summary_per, step)

            # do some validation
            v_istate = sess.run(zero_state_valid)  # initial zero input state (a tuple)
            valid_logprob = 0
            for i in range(validation_steps):
                v_batch, v_label, _ = next(validation_generator)

                feed_dict = {val_seq_input: v_batch, keep_prop_tf: 1}  # no dropout for validation
                for _i, v in enumerate(zero_state_valid):
                    feed_dict[v] = v_istate[_i]
                prediction, v_ostate = sess.run([final_predictions, hidden_state_valid], feed_dict=feed_dict)
                valid_logprob = valid_logprob + negativeLogProb(prediction, v_label)

                v_istate = v_ostate
            v_perplexity = float(2 ** (valid_logprob / validation_steps))
            print('step : %d epoch : %d validation perplexity: %.2f' % (step, epoch, v_perplexity))

            summary = tf.Summary(value=[
                tf.Summary.Value(tag="perplexity", simple_value=v_perplexity),
            ])
            validation_writer.add_summary(summary, step)

            v_istate = sess.run(zero_state_valid)  # initial zero input state (a tuple)
            # display a short text generated with the current weights and biases (every 150 batches)
            print(("=" * 50) + "generation" + ("=" * 50))
            next_feed = np.array([[dictionary['n']]])
            for k in range(2000):
                feed_dict = {val_seq_input: next_feed, keep_prop_tf: 1}  # no dropout for validation
                for _i, v in enumerate(zero_state_valid):
                    feed_dict[v] = v_istate[_i]

                probabilities, v_ostate = sess.run([final_predictions, hidden_state_valid], feed_dict=feed_dict)
                sample = sample_from_probabilities(probabilities, topn=10 if epoch <= 1 else 2, vocabulary_size=vocabulary_size)
                print(reverse_dictionary[sample], end="")
                next_feed = np.array([[sample]])  # feedback
                v_istate = v_ostate
            print("\n", ("=" * 50) + "end" + ("=" * 50))

        else:
            _, ostate = sess.run([train_step, hidden_state_train], feed_dict=feed_dict)

        # save a checkpoint
        if (current_time - checkpoint_last) > chechpoint_every_mins * minute:
            checkpoint_last = time.time()
            saved_file = saver.save(sess, 'checkpoints/rnn_train_' + timestamp, global_step=step)
            print("Saved file: " + saved_file)
            make_zip_results("linux_em", step, outputFileName)

        if (current_time - epoch_log_last) > log_epoch_every_mins * minute:
            epoch_log_last = time.time()

            print("step :", step, "epoch :", epoch, )

        # loop state around
        istate = ostate
        step += 1
        sys.stdout.flush()

execution_end = time.time()


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("this took {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


timer(execution_start, execution_end)
summary_writer.close()
validation_writer.close()
