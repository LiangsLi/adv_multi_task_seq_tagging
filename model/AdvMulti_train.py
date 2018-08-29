import os
import datetime
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import sys
import shutil
import random
from sklearn.metrics import accuracy_score
import argparse
import pathlib
from voc import Vocab, Tag
from config import MODEL_TYPE, ADV_STATUS

from AdvMulti_model import MultiModel
import data_helpers

# ==================================================
#  numpy 数组形式的单字、单词向量
# 单字向量使用预训练向量，单词向量是字向量的简单组合
init_embedding = Vocab('../data/insurance_wordvec.wv', '../data/20_data/vocab.txt', single_task=False,
                       bi_gram=True).word_vectors
parser = argparse.ArgumentParser()
parser.add_argument('--vocab_size', default=init_embedding.shape[0], type=int)

# Data parameters
parser.add_argument('--word_dim', default=100, type=int)
parser.add_argument('--lstm_dim', default=100, type=int)
parser.add_argument('--num_classes', default=4, type=int)
parser.add_argument('--num_corpus', default=20, type=int)
parser.add_argument('--embed_status', default=True, type=bool)
parser.add_argument('--gate_status', default=False, type=bool)

# predict ? train ?
parser.add_argument('--predict', default=False, type=bool)
parser.add_argument('--train', default=True, type=bool)

# Model Hyperparameters[t]
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--l2_reg_lambda', default=0.000, type=float)
parser.add_argument('--adv_weight', default=0.06, type=float)
parser.add_argument('--clip', default=5, type=int)

# Training parameters
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--batch_size_big', default=64, type=int)
parser.add_argument('--batch_size_huge', default=64, type=int)
# step
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--num_epochs_private', default=1000, type=int)
parser.add_argument('--evaluate_every', default=10, type=int)

# Early Stop
parser.add_argument('--all_early_stop_step', default=200, type=int)
parser.add_argument('--private_early_stop_step', default=200, type=int)

# Misc Parameters
parser.add_argument('--allow_soft_placement', default=True, type=bool)
parser.add_argument('--log_device_placement', default=False, type=bool)
parser.add_argument('--gpu_growth', default=True, type=bool)
FLAGS = parser.parse_args()
# if FLAGS.embed_status is False:
#     不使用预训练词向量
#     init_embedding = None

# print("\nParameters:")
# for attr, value in sorted(FLAGS.__private_stop_flags.items(), reverse=True):
#     print("{}={} \n".format(attr.upper(), value))
# print("")

time_format = '%y%m%d%H%M'
time_stamp = time.strftime(time_format, time.localtime())
# define log file
logger_file_path = pathlib.Path('../log')
if not logger_file_path.exists():
    logger_file_path.mkdir()
logger = logging.getLogger('record')
fh = logging.FileHandler(str(logger_file_path / time_stamp))
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.setLevel(logging.INFO)

if MODEL_TYPE == 'Model1':
    reuse_status = True
    sep_status = True
elif MODEL_TYPE == 'Model2':
    reuse_status = False
    sep_status = False
elif MODEL_TYPE == 'Model3':
    reuse_status = True
    sep_status = False
else:
    # print'choose the correct multi_model, the listed choices are Model1, Model2, Model3'
    logger.warning('Wrong Model Choosen {}'.format(MODEL_TYPE))
    sys.exit()

# stats = [FLAGS.embed_status, FLAGS.gate_status, ADV_STATUS]
# posfix = map(lambda x: 'Y' if x else 'N', stats)
# posfix.append(MODEL_TYPE)
# if ADV_STATUS:
#     posfix.append(str(FLAGS.adv_weight))

# Load data
# 用来保存模型的数据，每个列表长度等于任务的数量
train_data_iterator = []
dev_data_iterator = []
test_data_iterator = []
dev_df = []
test_df = []

TRAIN_FILE = []
DEV_FILE = []
TEST_FILE = []
DROP_OUT = []
for i in range(1, FLAGS.num_corpus + 1):
    TRAIN_FILE.append('../data/20_data/' + str(i) + '/train.csv')
    DEV_FILE.append('../data/20_data/' + str(i) + "/dev.csv")
    TEST_FILE.append('')
    DROP_OUT.append(0.65)

print("Loading data...")
# 加载数据。保存到对应列表
for i in range(FLAGS.num_corpus):
    # task data 0
    train_data_iterator.append(data_helpers.BucketedDataIterator(pd.read_csv(TRAIN_FILE[i])))
    # task data 1
    dev_df.append(pd.read_csv(DEV_FILE[i]))
    # task data 2
    dev_data_iterator.append(data_helpers.BucketedDataIterator(dev_df[i]))
    # task data 3
    # test_df.append(pd.read_csv(TEST_FILE[i]))
    # task data 4
    # test_data_iterator.append(data_helpers.BucketedDataIterator(test_df[i]))

logger.info('-' * 50)

shared_train_stop_step = [FLAGS.num_epochs] * FLAGS.num_corpus
# shared_train_best_step = [0] * FLAGS.num_corpus
# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
    )
    
    session_conf.gpu_options.allow_growth = FLAGS.gpu_growth
    
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        
        # build model
        model = MultiModel(batch_size=FLAGS.batch_size,
                           vocab_size=FLAGS.vocab_size,
                           word_dim=FLAGS.word_dim,
                           lstm_dim=FLAGS.lstm_dim,
                           num_classes=FLAGS.num_classes,
                           num_corpus=FLAGS.num_corpus,
                           lr=FLAGS.lr,
                           clip=FLAGS.clip,
                           l2_reg_lambda=FLAGS.l2_reg_lambda,
                           adv_weight=FLAGS.adv_weight,
                           init_embedding=init_embedding,
                           gates=FLAGS.gate_status,
                           adv=False,
                           reuseshare=reuse_status,
                           sep=sep_status)
        
        # Output directory for models
        model_name = 'multi_task_' + str(FLAGS.num_corpus) + '_' + time_stamp
        # try:
        #     递归删除model 参数保存目录
        # shutil.rmtree(os.path.join(os.path.curdir, "models", model_name))
        # except:
        #     pass
        out_dir = os.path.abspath(os.path.join(os.path.pardir, "checkpoints", model_name))
        
        print("Writing to {}\n".format(out_dir))
        
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        # modeli_embed_adv_gate_diff_dropout
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_all = []
        for i in range(1, FLAGS.num_corpus + 1):
            filename = 'task' + str(i)
            checkpoint_all.append(os.path.join(checkpoint_dir, filename))
        
        checkpoint_private = []
        checkpoint_shared = []
        
        for i in range(1, FLAGS.num_corpus + 1):
            private_filename = 'task_private_' + str(i)
            shared_filename = 'task_shared_' + str(i)
            checkpoint_private.append(os.path.join(checkpoint_dir, private_filename))
            checkpoint_shared.append(os.path.join(checkpoint_dir, shared_filename))
        checkpoint_shared.append(os.path.join(checkpoint_dir, 'shared_train_done'))
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        shared_vars = []
        task_private_vars = []
        for var in tf.global_variables():
            if var.name.split('/')[0] in ['embedding', 'shared', 'domain']:
                shared_vars.append(var)
        
        for i in range(1, FLAGS.num_corpus + 1):
            var_prefix = 'task' + str(i)
            temp = []
            for var in tf.global_variables():
                if var.name.startswith(var_prefix):
                    temp.append(var)
            task_private_vars.append(temp)
        
        shared_model_saver = tf.train.Saver(shared_vars, max_to_keep=20)
        task_private_saver = []
        for i in range(FLAGS.num_corpus):
            task_private_saver.append(tf.train.Saver(task_private_vars[i], max_to_keep=20))
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
        
        summ_path = pathlib.Path('../summaries')
        if not summ_path.exists():
            summ_path.mkdir()
        summ_path = summ_path / time_stamp
        if not summ_path.exists():
            summ_path.mkdir()
        
        shared_train_writer = tf.summary.FileWriter(str(summ_path / 'shared_train_loss'), sess.graph)
        private_train_writer = tf.summary.FileWriter(str(summ_path / 'private_train_loss'))
        dev_writer = tf.summary.FileWriter(str(summ_path / 'dev_loss'))
        metric_P_writer = tf.summary.FileWriter(str(summ_path / 'metric_P'))
        metric_R_writer = tf.summary.FileWriter(str(summ_path / 'metric_R'))
        metric_F1_writer = tf.summary.FileWriter(str(summ_path / 'metric_F1'))
        metric_new_F1_writer = tf.summary.FileWriter(str(summ_path / 'new_F1'))
        ####################################
        
        # Initialize all variables 执行初始化
        sess.run(tf.global_variables_initializer())
        
        # Initialize all the op
        # basictask is for basic cws loss, task is for combination loss, privatetask is for cws loss only on solo params
        # predictTask is for prediction, task_data is for storage of loading from .csv
        basictask = []
        task = []
        privateTask = []
        predictTask = []
        task_data = []
        # 把各种 op 保存起来，注意这里没有训练 domain scope 的 op（ 在model.domain_op）
        for i in range(FLAGS.num_corpus):
            """
            # task_basic_op 是 optimizer op（更新全图梯度）
            # global_basic_step 是 optimizer step
            # losses 是 reduce_mean(crf_log_likelihood)（仅仅只 CRF loss）
            """
            basictask.append([model.task_basic_op[i], model.global_basic_step[i], model.losses[i]])
            if model.adv:
                """
                task_op 是 optimizer op（更新除了 domain scope 之外的全图梯度）
                global_step ：optimizer step
                loss_com 是CRF loss 加入对抗损失之后的新 loss
                """
                task.append([model.task_op[i], model.global_step[i], model.loss_com[i]])
            """
            task_op_ss : optimizer op (更新私有计算图（私有 bi-LSTM ，私有 CRF）)
            global_step :
            losses : CRF loss
            """
            privateTask.append([model.task_op_ss[i], model.global_pristep[i], model.losses[i]])
            # basictask > task > privatetask
            # CRF 计算相关
            predictTask.append([model.scores[i], model.transition[i]])
            # 数据
            # task_data.append([train_data_iterator[i], dev_df[i], dev_data_iterator[i], test_df[i], test_data_iterator[i]])
            task_data.append(
                [train_data_iterator[i], dev_df[i], dev_data_iterator[i]])
            # [train_data_iterator[i], dev_df[i], dev_data_iterator[i], test_df[i], test_data_iterator[i]])
        
        
        def train_step_all(x_batch, y_batch, y_class_batch, seq_len_batch, id):
            """训练（使用对抗）"""
            step, loss_cws, loss_adv, loss_hess = model.train_step_task(sess,
                                                                        x_batch, y_batch, seq_len_batch, y_class_batch,
                                                                        DROP_OUT[id - 1],
                                                                        # 除去判别器的优化器：
                                                                        task[id - 1][0], task[id - 1][1],
                                                                        task[id - 1][2],
                                                                        # 判别器优化器：
                                                                        model.domain_op, model.global_step_domain,
                                                                        model.D_loss, model.H_loss
                                                                        )
            
            time_str = datetime.datetime.now().isoformat()
            print("Task_{}: {}: step {}, loss_cws {:g}, loss_adv {:g}, loss_hess {:g}".format(id, time_str, step,
                                                                                              loss_cws, loss_adv,
                                                                                              loss_hess))
            
            return step
        
        
        def train_step_basic(x_batch, y_batch, seq_len_batch, id):
            """训练（不使用对抗）"""
            step, loss, loss_summ = model.train_step_basic(sess,
                                                           x_batch, y_batch, seq_len_batch, DROP_OUT[id - 1],
                                                           basictask[id - 1][0],
                                                           basictask[id - 1][1],
                                                           basictask[id - 1][2], id
                                                           )
            
            time_str = datetime.datetime.now().isoformat()
            print("ALL_train: Task_{}: {}: step {}, loss {:g}".format(id, time_str, step, loss))
            # test_summ_writer.add_summary(summ, step)
            shared_train_writer.add_summary(loss_summ, step)
            return step
        
        
        def train_step_private(x_batch, y_batch, seq_len_batch, id):
            """训练私有计算图部分"""
            step, loss, loss_summ = model.train_step_pritask(sess,
                                                             x_batch, y_batch, seq_len_batch, DROP_OUT[id - 1],
                                                             # 私有计算图优化器
                                                             privateTask[id - 1][0], privateTask[id - 1][1],
                                                             privateTask[id - 1][2], id)
            
            time_str = datetime.datetime.now().isoformat()
            print("Private train:Task_{}: {}: step: +{}, loss {:g}".format(id, time_str, step, loss))
            # test_summ_writer.add_summary(summ, step + FLAGS.num_epochs)
            private_train_writer.add_summary(loss_summ, step + shared_train_stop_step[id - 1])
            
            return step
        
        
        def final_test_step(step, df, iterator, idx, predict=False, print_predict=True, summary=False):
            """测试predict/test、验证dev"""
            
            def get_begin_end(idxs, tag):
                
                begin = -1
                end = -1
                if FLAGS.num_classes == 4:
                    B_list = []
                    E_list = []
                    M_list = []
                    for i_, item in enumerate(idxs):
                        label = tag.idx2tag[item]
                        if label == 'B':
                            B_list.append(i_)
                        elif label == 'E':
                            E_list.append(i_)
                        elif label == 'M':
                            M_list.append(i_)
                    if len(B_list) == len(E_list) and len(B_list) != 0:
                        begin = B_list[0]
                        end = E_list[0]
                    elif len(B_list) != 0 or len(E_list) != 0:
                        if len(M_list) != 0:
                            if len(B_list) == 0:
                                begin = M_list[0]
                            else:
                                begin = B_list[0]
                            if len(E_list) == 0:
                                end = M_list[-1]
                            else:
                                end = E_list[0]
                    elif len(M_list) != 0:
                        begin = M_list[0]
                        end = M_list[-1]
                elif FLAGS.num_classes == 2:
                    I_list = []
                    for i_, item in enumerate(idxs):
                        if tag.idx2tag[item] == 'I':
                            I_list.append(i_)
                    if len(I_list) != 0:
                        begin = I_list[0]
                        end = I_list[-1]
                if begin > end:
                    begin = -1
                    end = -1
                return begin, end
            
            def get_match_size(pred_one, real_one):
                if (real_one[0], real_one[1]) == (-1, -1) or (pred_one[0], pred_one[1]) == (-1, -1):
                    return 0
                if pred_one[0] <= real_one[0]:
                    b1, e1 = pred_one
                    b2, e2 = real_one
                else:
                    b1, e1 = real_one
                    b2, e2 = pred_one
                match_size = min(e1, e2) - b2 + 1
                if match_size < 0:
                    match_size = 0
                return match_size
            
            def get_metric_out(data, test=False, print_metric=True):
                # y: I:0 O:1
                # cor_num = 0
                Tags = Tag()
                old_P_values = []
                old_R_values = []
                real_data_num = 0
                pred_data_num = 0
                pred_right_num = 0
                for y_pred, y in data:
                    old_one_P = [a == b for (a, b) in zip(y_pred, y) if Tags.idx2tag[b] in ['B', 'M', 'E', 'I']]
                    old_one_R = [a == b for (a, b) in zip(y_pred, y) if Tags.idx2tag[a] in ['B', 'M', 'E', 'I']]
                    if len(old_one_P) == 0:
                        pp = 0
                    else:
                        pp = np.mean(old_one_P)
                    if len(old_one_R) == 0:
                        rr = 0
                    else:
                        rr = np.mean(old_one_R)
                    old_P_values.append(pp)
                    old_R_values.append(rr)
                    
                    """
                    统计每一个知识点的准确率和召回率
                    p = TP/(TP + FP)          # FP 表示抽取出来但是抽取错误的，或者误抽取的
                    r = TP/(TP + FN)          # TP 表示抽取出来并且和标注数据比对正确，正确的标注判断如下面的论述 pred_right_num
                                              # (TP + FN) 标注数据的知识点总数 real_data_num
                                              # (TP + FP) 预测数据的知识点总数 pred_data_num
                    f = 2 * p * r / (p + r)
                    """
                    flag1, flag2, flag3 = False, False, False
                    pred_begin, pred_end = get_begin_end(y_pred, Tags)
                    real_begin, real_end = get_begin_end(y, Tags)
                    if pred_begin != -1 and pred_end != -1:
                        pred_data_num += 1
                        flag1 = True
                    if real_begin != -1 and real_end != -1:
                        real_data_num += 1
                        flag2 = True
                    match_size = get_match_size((pred_begin, pred_end), (real_begin, real_end))
                    try:
                        retio = match_size / (pred_end - pred_begin + real_end - real_begin + 2 - match_size)
                    except ZeroDivisionError as e:
                        logger.warning("Warning: " + str(e))
                        logger.warning(
                            "pb:{},pe:{},rb:{},re:{},ms:{}".format(pred_begin, pred_end, real_begin, real_end,
                                                                   match_size))
                        raise RuntimeError("stop:ZeroDivisionError")
                    if retio >= 0.8:
                        pred_right_num += 1
                        flag3 = True
                    if (not flag1 and flag3) or (not flag2 and flag3):
                        logger.warning("Warning: flag1/2=False but flag3=True")
                        logger.warning(
                            "pb:{},pe:{},rb:{},re:{},ms:{},r:{}".format(pred_begin, pred_end, real_begin, real_end,
                                                                        match_size, retio))
                        raise RuntimeError("stop:flag1/2=False but flag3=True")
                
                old_P = np.mean(old_P_values)
                old_R = np.mean(old_R_values)
                
                if (old_P + old_R) == 0:
                    old_F = 0
                else:
                    old_F = 2 * old_P * old_R / (old_P + old_R)
                
                if pred_data_num == 0:
                    new_P = 0
                else:
                    new_P = pred_right_num / pred_data_num
                
                if real_data_num == 0:
                    new_R = 0
                else:
                    new_R = pred_right_num / real_data_num
                
                if (new_P + new_R) == 0:
                    new_F = 0
                else:
                    new_F = 2 * new_P * new_R / (new_P + new_R)
                
                if print_metric:
                    # print('lhy P: ', old_P)
                    # print('lhy R: ', old_R)
                    print('old F: ', old_F)
                    print('new F: ', new_F)
                    # print('###' * 10)
                if test:
                    return (old_P, old_R, old_F), (new_P, new_R, new_F)
                else:
                    return old_F, new_F
            
            N = df.shape[0]
            # N 是 dev 数据样本数量
            # predictTask[idx][0]、[1]是训练好的 CRF 的 scores、transitions
            samples, batch_num, loss_summ = model.fast_all_predict(sess, N, iterator,
                                                                   predictTask[idx - 1][0],
                                                                   predictTask[idx - 1][1],
                                                                   basictask[idx - 1][2],
                                                                   idx,
                                                                   summary)
            
            (old_P, old_R, old_F), (new_P, new_R, new_F) = get_metric_out(samples, test=True)
            
            if summary:
                dev_writer.add_summary(loss_summ, step)
                for metric_value, writer in zip([old_F, new_F], [metric_F1_writer, metric_new_F1_writer]):
                    metric_summ = model.get_metric_summary(sess, metric_value, idx)
                    writer.add_summary(metric_summ, step)
            if predict:
                print("Now predict:")
                raise RuntimeError('predict模式需要新的数据迭代器，predict 模式下正确标签未知')
                # todo：predict mode
            else:
                if print_predict:
                    print("Dev  Task: ", idx)
                    print("Batch num: ", batch_num, "Sample num: ", N)
                    random.shuffle(samples)
                    # num_ = min(N // 10, 10)
                    for (pred_one, true_one) in samples[:2]:
                        print('right  : ', true_one)
                        print('predict: ', pred_one)
                        print('---' * 10)
            
            return old_F, new_F
        
        
        ########################################################
        # train loop
        if FLAGS.train:
            best_accuary = [0.0] * FLAGS.num_corpus
            best_step_all = [0] * FLAGS.num_corpus
            best_pval = [0.0] * FLAGS.num_corpus
            best_rval = [0.0] * FLAGS.num_corpus
            best_fval = [0.0] * FLAGS.num_corpus
            best_step_private = [0] * FLAGS.num_corpus
            private_stop_flag = [False] * FLAGS.num_corpus
            all_stop_flag = [False] * FLAGS.num_corpus
            
            logger.info('-------------train starts:{}--------------'.format(time_stamp))
            for i in range(FLAGS.num_epochs):
                # 逐个语料训练
                # 每一个 epoch 只运行不同语料的一个 batch
                if sum(all_stop_flag) == FLAGS.num_corpus:
                    shared_save_path = shared_model_saver.save(sess, checkpoint_shared[-1])
                    logger.info("Shared train : Early stop triggered in epoch:{}".format(i))
                    logger.info(">>>>>>Saved shared model to path:{}".format(shared_save_path))
                    # print("Shared train : Early stop triggered in epoch:{}".format(i))
                    break
                for j in range(1, FLAGS.num_corpus + 1):
                    if all_stop_flag[j - 1]:
                        continue
                    
                    if model.adv:  # 对抗
                        raise RuntimeError('暂不支持对抗')
                        # if j == 1:
                        #     x_batch, y_batch, y_class, seq_len_batch = task_data[j - 1][0].next_batch(
                        #         FLAGS.batch_size_big,
                        #         round=j - 1,
                        #         classifier=True)
                        # elif j == 2:
                        #     x_batch, y_batch, y_class, seq_len_batch = task_data[j - 1][0].next_batch(
                        #         FLAGS.batch_size_huge, round=j - 1, classifier=True)
                        # else:
                        #     x_batch, y_batch, y_class, seq_len_batch = task_data[j - 1][0].next_batch(FLAGS.batch_size,
                        #                                                                               round=j - 1,
                        #                                                                               classifier=True)
                        # with adv
                        # 使用某个 CRF 的损失更新计算图（不包含对抗）的梯度
                        # current_step = train_step_all(x_batch, y_batch, y_class, seq_len_batch, j)
                    else:  # 非对抗
                        x_batch, y_batch, seq_len_batch = task_data[j - 1][0].next_batch(FLAGS.batch_size)
                        # without adv
                        # 使用某个CRF 的损失计算
                        current_step = train_step_basic(x_batch, y_batch, seq_len_batch, j)
                    
                    if current_step % FLAGS.evaluate_every == 0:
                        # dev:
                        # print("***" * 10)
                        # print('current_step: ', current_step)
                        # 在dev数据集上验证模型：
                        old_f, new_f = final_test_step(current_step, task_data[j - 1][1], task_data[j - 1][2], j,
                                                       summary=True)
                        if best_accuary[j - 1] < new_f:
                            best_accuary[j - 1] = new_f
                            best_step_all[j - 1] = current_step
                            shared_save_path = shared_model_saver.save(sess, checkpoint_shared[j - 1])
                            private_save_path = task_private_saver[j - 1].save(sess, checkpoint_private[j - 1])
                            logger.info(
                                "Shared train: Task {} got better F1:{} in step {}".format(j, new_f, current_step))
                            logger.info(">>>>>>Saved shared model to {}, private mode to {}".format(shared_save_path,
                                                                                                    private_save_path))
                        
                        elif current_step - best_step_all[j - 1] > FLAGS.all_early_stop_step:
                            logger.info("Shared train : Task {} early stop in step:{}".format(j, current_step))
                            all_stop_flag[j - 1] = True
                            shared_train_stop_step[j - 1] = current_step
            
            if sum(all_stop_flag) != FLAGS.num_corpus:
                logger.info('-----------Shared Train ends-------------')
                shared_save_path = shared_model_saver.save(sess, checkpoint_shared[-1])
                logger.info(
                    '>>>>>>Finally,saved shared model to {} after epochs:{}'.format(shared_save_path, FLAGS.num_epochs))
            
            for i in range(FLAGS.num_corpus):
                logger.info(
                    'After shared train, Task{} best step is {} and F1:{:.2f}'.format(i + 1, best_step_all[i],
                                                                                      best_accuary[i] * 100))
            
            for j in range(1, FLAGS.num_corpus + 1):
                task_private_saver[j - 1].restore(sess, checkpoint_private[j - 1])
            
            shared_model_scores = []
            shared_f1s = []
            for shared_ckp in checkpoint_shared:
                logger.info('use: ' + shared_ckp)
                shared_model_saver.restore(sess, shared_ckp)
                # 测试：
                temp = []
                for j in range(1, FLAGS.num_corpus + 1):
                    old_f, new_f = final_test_step(0, task_data[j - 1][1], task_data[j - 1][2], j,
                                                   summary=False, print_predict=False)
                    logger.info(">LoadSharedMode--Task {}, F1 {:.2f}".format(j, new_f * 100))
                    temp.append(new_f)
                shared_model_scores.append(sum(temp))
                shared_f1s.append(temp)
            # 选择最优：
            best_shared_model = np.argmax(shared_model_scores)
            shared_model_saver.restore(sess, checkpoint_shared[best_shared_model])
            shared_best_f1 = shared_f1s[best_shared_model]
            print("##" * 10)
            logger.info("LoadSharedModel> Choose: " + checkpoint_shared[best_shared_model])
            # raise RuntimeError('stop')
            
            for i in range(FLAGS.num_epochs_private):
                stop = True
                for j in range(FLAGS.num_corpus):
                    # stop 初始化为 False
                    if private_stop_flag[j] is False:
                        stop = False
                
                # 持续训练直到所有 private_stop_flag 都是 True（所有 CRF 都取得最优）
                if stop is False:
                    for j in range(1, FLAGS.num_corpus + 1):
                        if private_stop_flag[j - 1]:
                            # 如果对应的 private_stop_flag 是 True，则跳过
                            continue
                        else:
                            # 如果语料 j的 private_stop_flag 是 False，那么就基于 loss[j]对私有计算图训练
                            x_batch, y_batch, seq_len_batch = task_data[j - 1][0].next_batch(FLAGS.batch_size)
                            current_step = train_step_private(x_batch, y_batch, seq_len_batch, j)
                            if current_step % FLAGS.evaluate_every == 0:
                                # 在 dev 数据集上验证模型：
                                old_f, new_f = final_test_step(current_step + shared_train_stop_step[j - 1],
                                                               task_data[j - 1][1],
                                                               task_data[j - 1][2], j, summary=True)
                                if best_accuary[j - 1] < new_f:
                                    # 如果是最优
                                    best_accuary[j - 1] = new_f
                                    best_step_private[j - 1] = current_step
                                    private_save_path = task_private_saver[j - 1].save(sess, checkpoint_private[j - 1])
                                    logger.info("Private train: Task {} got better F1:{} in step {}".format(j, new_f,
                                                                                                            current_step))
                                    logger.info(
                                        ">>>>>>Saved shared model to {}, private mode to {}".format(shared_save_path,
                                                                                                    private_save_path))
                                
                                elif current_step - best_step_private[j - 1] > FLAGS.private_early_stop_step:
                                    # 超过2000步都没有取得更好的结果
                                    logger.info(
                                        "Private train : Task {} early stop in step:{}".format(j, current_step))
                                    private_stop_flag[j - 1] = True
                else:
                    logger.info("Private train : Early stop triggered in epoch:{}".format(i))
                    break
        
        logger.info(">>>>>>>>>>>>>>>>Train Done<<<<<<<<<<<<<<<")
        # 结果对比：
        for i in range(FLAGS.num_corpus):
            logger.info('Shared train result: Task{} best F1:{:.2f}'.format(i + 1, shared_best_f1[i] * 100))
            logger.info(
                'After Private train, Task{} best step is {} and F1:{:.2f}'.format(i + 1, best_step_private[i],
                                                                                   best_accuary[i] * 100))
        # 加载私有模块：
        sess.run(tf.global_variables_initializer())
        for j in range(1, FLAGS.num_corpus + 1):
            task_private_saver[j - 1].restore(sess, checkpoint_private[j - 1])
        
        for shared_ckp in checkpoint_shared:
            logger.info('use: ' + shared_ckp)
            shared_model_saver.restore(sess, shared_ckp)
            # 测试：
            for j in range(1, FLAGS.num_corpus + 1):
                old_f, new_f = final_test_step(0, task_data[j - 1][1], task_data[j - 1][2], j,
                                               summary=False, print_predict=False)
                logger.info(">LoadFinalMode--Task {}, F1 {:.2f}".format(j, new_f * 100))
        
        if FLAGS.predict:
            """预测模式"""
            logger.info('-------------Show the results------------')
            print('--load best model')
            shared_model_saver.restore(sess, checkpoint_shared[-1])
            for j in range(1, FLAGS.num_corpus + 1):
                task_private_saver[j - 1].restore(sess, checkpoint_private[j - 1])
            # todo:目前的 predict 实际上是 test，（1）是没有专门的读取没有正确标签的迭代器、（2）是还是需要存放 N 个地方，分别抽取，不能存放1份抽取 N 次
            for i in range(FLAGS.num_corpus):
                print('Task:{}\n'.format(i + 1))
                # task_data[i][3]是测试数据的 dataFrame（pandas 返回），[4]是测试数据的 iterator
                old_f, new_f = final_test_step(0, task_data[i][3], task_data[i][4], i + 1, print_predict=False)
                # evaluate_word_PRF(yp, yt)
                # get_metric_out(yp, yt)
                # todo:将预测输出写到文件中
