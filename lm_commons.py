import collections
import glob

import numpy as np

from zipFileHelperClassHelper import ZipFile

# # time tracking
minute = 60

def negativeLogProb(predictions, labels):  # [batch_size*seq_length , vocabulary] as labels [batch_size*seq_length]
    """Log-probability of the true labels in a predicted batch."""
    predictions[predictions < 1e-10] = 1e-10  # wont go -infinity
    return np.sum(-np.log2(predictions[np.arange(labels.shape[0]), labels])) / labels.shape[0]  # single value


def perplexity(predictions, labels):
    """perplexity of the model."""
    return 2 ** negativeLogProb(predictions, labels)


def minibatch_generator(data, nb_epochs, gen_batch_size, gen_seq_len):
    """
                        thanks to (martin_gorner) repo
    Divides the data into batches of sequences so that all the sequences in one batch
    continue in the next batch. This is a generator that will keep returning batches
    until the input data has been seen nb_epochs times. Sequences are continued even
    between epochs, >>apart from one, the one corresponding to the end of raw_data.<< accepted approximation
    The remainder at the end of raw_data that does not fit in an full batch is ignored.
    """
    data = np.array(data)
    data_len = data.shape[0]
    # using (data_len-1) because we must provide for the sequence shifted by 1 too
    steps_per_epoch = (data_len - 1) // (gen_batch_size * gen_seq_len)

    assert steps_per_epoch > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."

    rounded_data_len = steps_per_epoch * gen_batch_size * gen_seq_len
    xdata = np.reshape(data[0:rounded_data_len], [gen_batch_size, steps_per_epoch * gen_seq_len])  # [....####] => [....,####]
    ydata = np.reshape(data[1:rounded_data_len + 1], [gen_batch_size, steps_per_epoch * gen_seq_len])

    # batch generator
    for epoch in range(nb_epochs):
        for step in range(steps_per_epoch):
            x = xdata[:, step * gen_seq_len:(step + 1) * gen_seq_len]
            y = ydata[:, step * gen_seq_len:(step + 1) * gen_seq_len]
            # this will circulate DOWN for epoch > 0
            x = np.roll(x, -epoch, axis=0)  # to continue continue continue the text from epoch to epoch (do not reset rnn state! except the last bottom sample)
            y = np.roll(y, -epoch, axis=0)

            yield x, y, epoch


def read_data_files_as_chars(directory):
    concat_text = []
    file_matches = glob.glob(directory, recursive=True)
    for directory in file_matches:
        print("Loading file " + directory)
        with open(directory, "r") as file:
            try:
                concat_text.extend(file.read().lower() + "\n\n\n")
            except ValueError:
                _ = None

    return concat_text


def read_data_files_as_words(directory):
    concat_text = ""
    file_matches = glob.glob(directory, recursive=True)
    for file_dir in file_matches:
        print("Loading file " + file_dir)
        with open(file_dir, "r") as file:
            try:
                concat_text += file.read().lower() + "\n\n\n"
            except ValueError:
                _ = None

    return concat_text.split(' ')


def build_char_dataset(corpus):
    '''
    extract features by mapping each word to number and vice versa
    make count array of the most common words
    make a dictionary for word to num             word -> rank
    make reverse dictionary       word <- rank
    map each word to its number(rank or 0 for UNK) as list of words
    :param corpus: the text
    :return: dictionaries and the encoded data
    '''
    count = collections.Counter(corpus).items()
    _dictionary = dict()  # map common words to >> number according to frequency 0 for UNK 1 for THE
    for word, _ in count:
        _dictionary[word] = len(_dictionary)

    data = list(map(lambda _word: _dictionary[_word], corpus))  # same as words list but of numbers corresponding to each word
    _reverse_dictionary = dict(zip(_dictionary.values(), _dictionary.keys()))
    return data, count, _dictionary, _reverse_dictionary


def build_word_dataset(corpus, vocabulary_size):
    '''
    extract features by mapping each word to number and vice versa
    make count array of the most common words
    make a dictionary for word to num             word -> rank
    make reverse dictionary       word <- rank
    map each word to its number(rank or 0 for UNK) as list of words
    :param corpus: the text
    :return: dictionaries and the encoded data
    '''
    count = [['UNK', -1]]  # frequency of the most common 20000 word
    count.extend(collections.Counter(corpus).most_common(vocabulary_size - 1))
    dictionary = dict()  # map common words to >> number according to frequency 0 for UNK 1 for THE
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()  # same as words list but of numbers corresponding to each word
    unk_count = 0
    for word in corpus:
        if word in dictionary:  # is it a common word ?
            index = dictionary[word]  # it's rank
        else:
            index = 0  # UNK is mapped to 0
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count  # ['UNK', -1] => ['UNK', unk_count]
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def convert_to_one_line(text):
    return text.replace("\n", "\\n").replace("\t", "\\t")


def sample_from_probabilities(probabilities, topn, vocabulary_size, is_word_level=False):
    """Roll the dice to produce a random integer in the [0..vocabulary_size] range,
        according to the provided probabilities. If topn is specified, only the
        topn highest probabilities are taken into account.
        :param probabilities: a list of size vocabulary_size with individual probabilities
        :param topn: the number of highest probabilities to consider. Defaults to all of them.
        :return: a random integer
        """
    probabilities = probabilities[-1, :]

    p = np.squeeze(probabilities)
    p[np.argsort(p)[:-topn]] = 0  # leave only the topn , zero otherwise
    if is_word_level:  # cut the UNK
        p[0] /= 1000
    p = p / np.sum(p)  # normalize
    return np.random.choice(vocabulary_size, 1, p=p)[0]  # get one sample


def make_zip_results(filename, step, outputFileName):
    myzipfile = ZipFile('{}{}.zip'.format(filename, step))
    myzipfile.addDir('log/')
    myzipfile.addDir('checkpoints/')
    myzipfile.addFile('reverse_dictionary.pkl.gz')
    myzipfile.addFile(outputFileName)
    myzipfile.print_info()
