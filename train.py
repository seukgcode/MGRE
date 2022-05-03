import config
import models
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'MGRE', help = 'name of the model')
parser.add_argument('--save_name', type = str)

parser.add_argument('--train_prefix', type = str, default = 'dev_train')
parser.add_argument('--test_prefix', type = str, default = 'dev_dev')


args = parser.parse_args()
model = {
	'MGRE': models.MGRE,
}

con = config.Config(args)
con.set_max_epoch(5)
con.load_train_data()
con.load_test_data()
con.train(model[args.model_name], args.save_name)
