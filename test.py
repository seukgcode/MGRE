import config
import models
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'LSTM', help = 'name of the model')
parser.add_argument('--save_name', type = str)

parser.add_argument('--train_prefix', type = str, default = 'train')
parser.add_argument('--test_prefix', type = str, default = 'dev_dev')
parser.add_argument('--input_theta', type = float, default = -1)


args = parser.parse_args()
model = {
	'MGRE': models.MGRE,
}

con = config.Config(args)
con.load_test_data()
con.testall(model[args.model_name], args.save_name, args.input_theta)
