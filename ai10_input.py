import os
import math
import tensorflow as tf

BOARD_SIZE = 8

NUM_CLASSES = 1
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

class OthelloRecord(object): pass

def read_logs(filename_queue):
    """Reads and parses examples from the board log files

    Args:
        filename_queue: A queue of strings with the filenames to read from

    Returns:
        An objects representing a single example, with the following fields:
            height: number of rows in the result (8)
            width: number of columns in the result (8)
            depth: number of channels in the result (1)
            key: a scalar string Tensor describing the filename & record number
                for this example
            score: an float32 Tensor with final score of the board in the range
                -64..64 normalized to 0..1.
            int8boards: a tuple of 8 [height, width, depth] int8 Tensors with the
                board data. There are 8 for rotation invariance.
        Each object is a rotation or flipped version of the original board
    """

    result = OthelloRecord()
    score_bytes = 1
    result.height = 8
    result.width = 8
    result.depth = 1
    image_bytes = result.height * result.width * result.depth
    record_bytes = score_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)

    result.score = tf.cast(
        tf.strided_slice(record_bytes, [0], [score_bytes]), tf.int8)
    result.score = tf.add(
        tf.divide(
            tf.to_float(result.score),
            128
        ),
        0.5
    )

    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [score_bytes],
                         [score_bytes + image_bytes]),
        [result.depth, result.height, result.width])

    # Goes from [depth, height, width] to [height, width, depth]
    board_a = tf.transpose(depth_major, [1, 2, 0])
    # Generates all the extra rotations
    board_b = tf.image.rot90(board_a, 1)
    board_c = tf.image.rot90(board_a, 3)
    board_d = tf.image.rot90(board_a, 2)
    board_e = tf.image.flip_left_right(board_a)
    board_f = tf.image.flip_up_down(board_a)
    board_g, board_h = tf.image.flip_up_down(board_b), tf.image.flip_up_down(board_c)

    result.int8boards = [
        (board_a, result.score),
        (board_b, result.score),
        (board_c, result.score),
        (board_d, result.score),
        (board_e, result.score),
        (board_f, result.score),
        (board_g, result.score),
        (board_h, result.score)
    ]

    return result

def _generate_board_and_score_batch(board, score, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Args:
        image: 3D Tensor of size [height, width, 1] of type.float32
        label: 1D Tensor of size type.int32
        min_queue_examples: int32, minimum number of smaples to retain
            in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue
    
    Returns:
        boards: Boards. 4D Tensor of size [batch_size, height, width, 1]
        scores: Scores. 1D Tensor of size [batch_size]
    """

    num_preprocess_threads = 16
    if shuffle:
        boards, score_batch = tf.train.shuffle_batch_join(
            board, # [board, score]
            batch_size=batch_size,
            #num_threads=num_preprocess_threads,
            capacity=min_queue_examples + batch_size,
            min_after_dequeue=min_queue_examples
            )
    else:
       boards, score_batch = tf.train.batch_join(
            board, # [board, score],
            batch_size=batch_size,
            #num_threads=num_preprocess_threads,
            capacity=min_queue_examples + batch_size
            )

    tf.summary.image('images', boards)

    return boards, tf.reshape(score_batch, [batch_size])

def inputs(eval_data, data_dir, batch_size):
    """Construct input for AI10 evaluation using the Reader ops.

    Args:
        eval_data: bool, indicating if one should use the train or eval data set.
        data_dir: Path to the log data directory.
        batch_size: Number of images per batch.

    Returns:
        boards: Boards. 4D Tensor for size [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1]
        scores: Scores. 1D Tensor of [batch_size] size.
    """
    if not eval_data:
        filenames = [os.path.join(data_dir, file) for file in \
                     os.listdir(data_dir) if file.endswith('.oth')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.oth')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    
    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_logs(filename_queue)
    float_boards = tuple(
        (tf.cast(board, tf.float32), 
        score)
    for board, score in read_input.int8boards)
    
    for board, score in float_boards:
        score.set_shape([1])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    return _generate_board_and_score_batch(float_boards, read_input.score,
                                           min_queue_examples, batch_size,
                                           shuffle=False)
