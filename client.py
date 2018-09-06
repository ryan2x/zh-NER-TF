import os
import numpy as np
from utils import get_logger
from data import pad_sequences, batch_yield
from tensorflow.contrib.crf import viterbi_decode


class BiLSTM_CRF_Client(object):
    def __init__(self, args, embeddings, tag2label, vocab, paths, config):
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.hidden_dim = args.hidden_dim
        self.embeddings = embeddings
        self.CRF = args.CRF
        self.update_embedding = args.update_embedding
        self.dropout_keep_prob = args.dropout
        self.optimizer = args.optimizer
        self.lr = args.lr
        self.clip_grad = args.clip
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.shuffle = args.shuffle
        # self.model_path = paths['model_path']
        # self.summary_path = paths['summary_path']
        # self.logger = get_logger(paths['log_path'])
        # self.result_path = paths['result_path']
        self.config = config

    def demo_one(self, server, sent, verbose=None):
        label_list = []
        for seqs, labels in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            if verbose:
                print('seqs', type(seqs), len(seqs), len(seqs[0]), seqs)
                print('batch_size', self.batch_size)
            label_list_, _ = self.predict_one_batch(server, seqs, verbose=verbose)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        tag = [label2tag[label] for label in label_list[0]]
        return tag

    def predict_one_batch(self, server, seqs, verbose=None):
        """
        :param server:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        import tensorflow as tf

        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)
        if verbose:
            print('feed_dicts', type(feed_dict), feed_dict.keys())

        host, port = server.split(':')
        stub, request = serving_client(host, port)

        input_seq_length = len(seqs[0])
        tag_length = len(self.tag2label)

        request.inputs['word_ids'].CopyFrom(
            tf.contrib.util.make_tensor_proto(feed_dict["word_ids"],
                                              shape=[1, len(seqs[0])], dtype=np.int32))
        request.inputs['sequence_lengths'].CopyFrom(
            tf.contrib.util.make_tensor_proto(feed_dict["sequence_lengths"],
                                              shape=[len(feed_dict["sequence_lengths"])], dtype=np.int32))
        request.inputs['dropout'].CopyFrom(
            tf.contrib.util.make_tensor_proto(feed_dict["dropout"], shape=[], dtype=np.float32))

        print(request)

        # sync requests
        result_future = stub.Predict(request, 30.)
        print(result_future)

        logits = np.array(result_future.outputs['logits'].float_val).reshape((1, input_seq_length, tag_length))
        transition_params = np.array(result_future.outputs['transition_params'].float_val).reshape((tag_length, tag_length))
        print(logits)
        print(logits.shape, logits.dtype)
        print(transition_params)
        print(transition_params.shape, transition_params.dtype)

        # logits, transition_params = client.run(feed_dict)
        label_list = []
        for logit, seq_len in zip(logits, seq_len_list):
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
            label_list.append(viterbi_seq)
        if verbose:
            print('logits', type(logits), logits.shape, logits.dtype)
            print('transition_params', type(transition_params), transition_params.shape, transition_params.dtype)
            print('label_list', label_list)
        return label_list, seq_len_list

    def get_feed_dict(self, seqs, dropout=None):
        """

        :param seqs:
        :param labels:
        :param lr:
        :param dropout:
        :return: feed_dict
        """
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)

        feed_dict = {"word_ids": word_ids,
                     "sequence_lengths": seq_len_list}
        if dropout is not None:
            feed_dict["dropout"] = dropout

        return feed_dict, seq_len_list


def serving_client(host, port, name=None, method=None):
    from grpc.beta import implementations
    from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
    from tensorflow_serving.apis import prediction_service_pb2

    if not name:
        name = 'BiLSTM_CRF'
    if not method:
        method = 'predict_word_ids'
    print('server', host, port)

    # create the RPC stub
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    # stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # create the request object and set the name and signature_name params
    request = predict_pb2.PredictRequest()
    request.model_spec.name = name
    request.model_spec.signature_name = method

    return stub, request


def test101(**kwargs):
    import argparse
    from utils import str2bool
    from data import read_dictionary, tag2label, random_embedding
    import tensorflow as tf

    print('test101', kwargs)

    ## Session configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory

    ## hyperparameters
    parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
    parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
    parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
    parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
    parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')
    parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
    parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
    parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
    parser.add_argument('--pretrain_embedding', type=str, default='random',
                        help='use pretrained char embedding or init it randomly')
    parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
    parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
    parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
    parser.add_argument('--demo_model', type=str, default='1521112368', help='model for test and demo')
    args = parser.parse_args([])

    ## get char embeddings
    word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))
    if args.pretrain_embedding == 'random':
        embeddings = random_embedding(word2id, args.embedding_dim)
    else:
        embedding_path = 'pretrain_embedding.npy'
        embeddings = np.array(np.load(embedding_path), dtype='float32')

    paths = {}

    client = BiLSTM_CRF_Client(args, embeddings, tag2label, word2id, paths, config=config)

    demo_sent = kwargs.get("demo_sent")
    demo_sent = list(demo_sent.strip())
    print('demo_sent', len(demo_sent))
    demo_data = [(demo_sent, ['O'] * len(demo_sent))]

    ret1 = client.demo_one(kwargs.get("server"), demo_data, verbose=True)

    print('result-1', ret1)

    from utils import get_entity

    PER, LOC, ORG = get_entity(ret1, demo_sent)
    print('PER: {}\nLOC: {}\nORG: {}'.format(PER, LOC, ORG))


def test102(**kwargs):
    print('test102', kwargs)

    import tensorflow as tf

    server = kwargs.get("server")

    host, port = server.split(':')
    stub, request = serving_client(host, port, "half_plus_two", "serving_default")

    values = kwargs.get('values')
    if not values:
        values = [34.55, 233.233]
    request.inputs['x'].CopyFrom(
        tf.contrib.util.make_tensor_proto(values, shape=[len(values)], dtype=np.float32))

    # sync requests
    result_future = stub.Predict(request, 30.)
    # print(result_future)
    result = np.array(result_future.outputs['y'].float_val)
    print('result', result)


def test103(**kwargs):
    print('test103', kwargs)

    import tensorflow as tf

    server = kwargs.get("server")
    method = kwargs.get("method")

    host, port = server.split(':')
    if not method:
        stub, request = serving_client(host, port, "counter", "get_counter")
    elif method in ['incr_counter', 'reset_counter']:
        stub, request = serving_client(host, port, "counter", method)
    elif method == "incr_counter_by":
        stub, request = serving_client(host, port, "counter", method)
        delta = float(kwargs.get('delta', 33.45))
        request.inputs['delta'].CopyFrom(
            tf.contrib.util.make_tensor_proto(delta, shape=[], dtype=tf.float32))
        print(request.inputs['delta'])
    else:
        stub, request = serving_client(host, port, "counter", "get_counter")

    # sync requests
    result_future = stub.Predict(request, 30.)
    # print(result_future)

    counter = np.array(result_future.outputs['output'].float_val)
    print(counter)


if __name__ == '__main__':
    demo_sent = '在周恩来总理的领导下，有当时中共中央主管科学工作的陈毅、国务院副总理兼国家计委主任李富春具体领导下，在北京召开了包括中央各部门、各有关高等学校和中国科学院的科学技术工作人员大会，动员制定十二年科学发展远景规划。'
    test101(server="localhost:8501", demo_sent=demo_sent)
