import tensorflow as tf
import numpy as np
import os, argparse, time, random
from model import BiLSTM_CRF
from utils import str2bool, get_logger, get_entity
from data import read_corpus, read_dictionary, tag2label, random_embedding


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
parser.add_argument('--pretrain_embedding', type=str, default='random', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1521112368', help='model for test and demo')
args = parser.parse_args()


## get char embeddings
word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))
if args.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, args.embedding_dim)
else:
    embedding_path = 'pretrain_embedding.npy'
    embeddings = np.array(np.load(embedding_path), dtype='float32')


## read corpus and get training data
if args.mode != 'demo':
    train_path = os.path.join('.', args.train_data, 'train_data')
    test_path = os.path.join('.', args.test_data, 'test_data')
    train_data = read_corpus(train_path)
    test_data = read_corpus(test_path); test_size = len(test_data)


## paths setting
paths = {}
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
output_path = os.path.join('.', args.train_data+"_save", timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))


## training model
if args.mode == 'train':
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()

    ## hyperparameters-tuning, split train/dev
    # dev_data = train_data[:5000]; dev_size = len(dev_data)
    # train_data = train_data[5000:]; train_size = len(train_data)
    # print("train data: {0}\ndev data: {1}".format(train_size, dev_size))
    # model.train(train=train_data, dev=dev_data)

    ## train model on the whole training data
    print("train data: {}".format(len(train_data)))
    model.train(train=train_data, dev=test_data)  # use test_data as the dev_data to see overfitting phenomena

## testing model
elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    print("test data: {}".format(test_size))
    model.test(test_data)

## demo
elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        while(1):
            print('Please input your sentence:')
            demo_sent = input()
            if demo_sent == '' or demo_sent.isspace():
                print('See you next time!')
                break
            else:
                demo_sent = list(demo_sent.strip())
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                tag = model.demo_one(sess, demo_data)
                PER, LOC, ORG = get_entity(tag, demo_sent)
                print('PER: {}\nLOC: {}\nORG: {}'.format(PER, LOC, ORG))

## demo
elif args.mode == 'demo1':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    demo_sent = '在周恩来总理的领导下，有当时中共中央主管科学工作的陈毅、国务院副总理兼国家计委主任李富春具体领导下，在北京召开了包括中央各部门、各有关高等学校和中国科学院的科学技术工作人员大会，动员制定十二年科学发展远景规划。'
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        #
        demo_sent = list(demo_sent.strip())
        demo_data = [(demo_sent, ['O'] * len(demo_sent))]
        tag = model.demo_one(sess, demo_data, verbose=True)
        PER, LOC, ORG = get_entity(tag, demo_sent)
        print('PER: {}\nLOC: {}\nORG: {}'.format(PER, LOC, ORG))

## demo
elif args.mode == 'save':
    import shutil

    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    #
    model_version = "1"
    export_path = os.path.join('.', args.train_data+"_export", model_version)
    shutil.rmtree(export_path, ignore_errors=True)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    #
    with tf.Session(config=config) as sess:
        print('============= save =============')
        saver.restore(sess, ckpt_file)
        print('model restored')

        graph = tf.get_default_graph()

        # word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")

        word_ids = graph.get_tensor_by_name('word_ids:0')
        sequence_lengths = graph.get_tensor_by_name('sequence_lengths:0')
        dropout_pl = graph.get_tensor_by_name('dropout:0')
        print(word_ids, sequence_lengths, dropout_pl)

        logits = model.logits
        transition_params = model.transition_params
        # out_classes = graph.get_tensor_by_name('lstm/initial_state:0')

        tensor_info_word_ids = tf.saved_model.utils.build_tensor_info(word_ids)
        tensor_info_sequence_lengths = tf.saved_model.utils.build_tensor_info(sequence_lengths)
        tensor_info_dropout = tf.saved_model.utils.build_tensor_info(dropout_pl)

        tensor_info_logits = tf.saved_model.utils.build_tensor_info(logits)
        tensor_info_transition_params = tf.saved_model.utils.build_tensor_info(transition_params)
        # tensor_info_output = tf.saved_model.utils.build_tensor_info(output_tuple)

        #
        # feed_dict = {"word_ids": word_ids, "sequence_lengths": sequence_lengths, "dropout": dropout_pl}
        # logits, transition_params = sess.run([model.logits, model.transition_params],
        #                                      feed_dict=feed_dict)
        print('logits', logits)
        print('transition_params', transition_params)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'word_ids': tensor_info_word_ids,
                        'sequence_lengths': tensor_info_sequence_lengths,
                        'dropout': tensor_info_dropout},
                outputs={'logits': tensor_info_logits, "transition_params": tensor_info_transition_params},
                # outputs={'logits': logits, "transition_params": transition_params},
                # outputs={'Placeholder_1': tensor_info_output},
                # outputs={'Placeholder_1': tensor_info_logits, "Placeholder_2": tensor_info_transition_params},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        legacy_init_op = tf.group(
            tf.tables_initializer(), name='legacy_init_op')

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_word_ids':
                    prediction_signature,
            },
            legacy_init_op=legacy_init_op
        )

        # export the model
        builder.save(as_text=True)

    print("model saved")

