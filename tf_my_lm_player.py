import gzip
import pickle

import numpy as np
import tensorflow as tf

from NMT_commons import sample_from_probabilities

model = 5
rand_init = False

seedword = '#inclu'  # if rand_init == False

path_pieces = [
    ("linux_out", '1522494838-185000'),
    ("shake_out", '1522455228-95000'),
    ('final_word_level', "1522489752-65000"),  # operates on embedding
    ("linux2_out", '1522855612-25000'),  # operates on embedding
    ("linux3_out", '1523103812-25000'),

    ("final_linux", '1523182986-108012'),
    ("final_linux_em", '1523183860-105491'),
    ("word_level", '1523232165-36030'),
][model]

with gzip.open('.\\runs\\' + path_pieces[0] + '\\reverse_dictionary.pkl.gz', 'rb') as f:
    reverse_dictionary = pickle.load(f)

dictionary = dict(zip(reverse_dictionary.values(), reverse_dictionary.keys()))
vocabulary_size = len(dictionary)
print("vocabulary", vocabulary_size)
stacked_layers = 3
internal_state_size = 256 if model == 2 else 512

meta_graph = '.\\runs\\{}\checkpoints\\rnn_train_{}.meta'.format(*path_pieces)
variable_state = '.\\runs\\{}\checkpoints\\rnn_train_{}'.format(*path_pieces)

zero_state_tuple = ('valid/MultiRNNCellZeroState/DropoutWrapperZeroState/GRUCellZeroState/zeros:0',
                    'valid/MultiRNNCellZeroState/DropoutWrapperZeroState_1/GRUCellZeroState/zeros:0',
                    'valid/MultiRNNCellZeroState/DropoutWrapperZeroState_2/GRUCellZeroState/zeros:0')

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(meta_graph)
    new_saver.restore(sess, variable_state)
    state_tuple = sess.run(zero_state_tuple)

    if rand_init:
        feed = np.random.randint(vocabulary_size)  # random word id
        feed = np.array([[feed]])  # shape [batch_size, sequence_length] with batch_size=1 and sequence_length=1
    else:

        feed = np.zeros([1, len(seedword)])

        for i, letter in enumerate(seedword):
            feed[0, i] = dictionary[letter]

    # initial values
    y = feed
    h = np.zeros([1, internal_state_size * stacked_layers], dtype=np.float32)  # [ batch_size, INTERNALSIZE * NLAYERS]
    for i in range(1000000000):

        feed_dict = {'valid/val_seq_input:0': feed, 'keep_prop_tf:0' if model > 2 else 'Placeholder:0': 1}  # no dropout for validation
        for _i, v in enumerate(zero_state_tuple):
            feed_dict[v] = state_tuple[_i]
        prediction, state_tuple = sess.run(['valid/final_predictions:0', 'valid/hidden:0'], feed_dict=feed_dict)

        # If sampling is be done from the topn most likely characters, the generated text
        # is more credible and more "english". If topn is not set, it defaults to the full
        # distribution (ALPHASIZE)

        next_feed = sample_from_probabilities(prediction, topn=2000 if model == 2 or model == 7 else 2, vocabulary_size=vocabulary_size, is_word_level=model == 2 or model == 7)
        feed = np.array([[next_feed]])  # shape [batch_size, sequence_length] with batch_size=1 and sequence_length=1
        next_feed = reverse_dictionary[next_feed]
        print(next_feed, end=" " if model == 2 or model == 7 else "")

#  was still much better only to remain in this area as the part of the empire of africa until the death of justinian i
# the islamic republic of mali was filled by a republic and a tribe of indigenous tribes from this one is a peculiar list
# of deities representing the great majority of the muslim population the lake itself has developed a wide variety of religious
# beliefs including that of the large muslim muslim state the muslim population runs from sunni bush means connected with the
#  population and the population of the country about a quarter of the arab population there are about three zero zero zero zero muslims worldwide
# in december the jewish community of palestinian state hosts the arab population there is a population

# the country main article demographics of lebanon religion main article islam religion its existence in islam is the ancient arabic
#  language meaning a descendant of arab or muslim descent
# culture of afghanistan many of whom have adherents the schools of islam generally speaking the well and less advanced
#  regions of india are called villages and thus the majority of these schools
