import config
import models
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'LSTM', help = 'name of the model')
parser.add_argument('--save_name', type = str)

parser.add_argument('--train_prefix', type = str, default = 'train')
parser.add_argument('--test_prefix', type = str, default = 'dev_dev')
parser.add_argument('--input_theta', type = float, default = -1)
parser.add_argument('--two_phase', action='store_true')


args = parser.parse_args()
model = {
    'MGRE_bert': models.MGRE_bert,
}

con = config.BertConfig(args)
con.load_test_data()
pretrain_model_name = 'checkpoint_bert_relation_exist_cls'
con.testall(model[args.model_name], args.save_name, args.input_theta, args.two_phase, pretrain_model_name)