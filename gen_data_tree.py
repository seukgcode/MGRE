import os
import json

data_path = './prepro_data'

train_annotated_file_name = os.path.join(data_path, 'dev_train.json')
dev_file_name = os.path.join(data_path, 'dev_dev.json')
test_file_name = os.path.join(data_path, 'dev_test.json')

train_annotated_tree_name = os.path.join(data_path, 'dev_train_tree.json')
dev_tree_name = os.path.join(data_path, 'dev_dev_tree.json')
test_tree_name = os.path.join(data_path, 'dev_test_tree.json')



# Dependency Tree
from nltk.parse.stanford import StanfordDependencyParser
java_path = "/opt/jdk1.8.0_231/bin/java"
os.environ['JAVAHOME'] = java_path
dep_parser=StanfordDependencyParser("jars/stanford-parser.jar",
                                    "jars/stanford-parser-4.2.0-models.jar",
                                    "jars/englishPCFG.ser.gz")

dependency_type = {}
def tree_build(data_file_name, is_training = True, suffix=''):
    # build tree for document
    ori_data = json.load(open(data_file_name))
    data = []

    print('doc_num:', len(ori_data))
    for i in range(len(ori_data)):
        print('doc', i)

        sents = ori_data[i]['sents']
        Ls = ori_data[i]['Ls']

        doc_item = {}

        for j in range(len(sents)):
            print(j)
            parse_tree = dep_parser.parse(sents[j])
            for trees in parse_tree:
                tree = trees

            for k in tree.nodes:
                if k == 0:
                    continue
                item = {}

                if tree.nodes[k]['head'] == 0:
                    item['head'] = -1
                else:
                    item['head'] = tree.nodes[k]['head'] - 1 + Ls[j]
                item['rel'] = tree.nodes[k]['rel']


                doc_item[tree.nodes[k]['address'] - 1 + Ls[j]] = item

                if item['rel'] not in dependency_type:
                    dependency_type[item['rel']] = 0

        data.append(doc_item)

    # saving
    print("Saving files")
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"

    json.dump(data, open(name_prefix + suffix + '_tree.json', 'w'))


tree_build(train_annotated_file_name, is_training = False, suffix = '_train')
tree_build(dev_file_name, is_training = False, suffix = '_dev')
tree_build(test_file_name, is_training = False, suffix = '_test')




# save the dependency map
dependency_map = dict((i, j) for i, j in zip(dependency_type.keys(), range(len(dependency_type))))
json.dump(dependency_map, open('./prepro_data/dependency_map.json', 'w'))


dep_map = json.load(open('./prepro_data/dependency_map.json'))
print("dependency_num:",len(dep_map))

def dep2id(data_file_name):
    data = json.load(open(data_file_name))

    for i in range(len(data)):
        for j in data[i].keys():
            data[i][j]['rel'] = dep_map[data[i][j]['rel']]
    json.dump(data, open(data_file_name, 'w'))


dep2id(train_annotated_tree_name)
dep2id(dev_tree_name)
dep2id(test_tree_name)





# find the path in tree
def get_path(tree, node, node_t=-1):
    path = []
    path.append(node)
    while (node != node_t):
        node = tree[str(node)]['head']
        path.append(node)
    return path


def get_layer(data_file_name, is_training = True, suffix=''):
    data = json.load(open(data_file_name))
    for i in range(len(data)):
        for j in data[i].keys():
            path = get_path(data[i], int(j))
            layer_num = len(path) - 1

            data[i][j]['path'] = path
            data[i][j]['layer'] = layer_num

    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"
    json.dump(data, open(name_prefix + suffix + '_tree.json', 'w'))


get_layer(train_annotated_tree_name, is_training = False, suffix = '_train')
get_layer(dev_tree_name, is_training = False, suffix = '_dev')
get_layer(test_tree_name, is_training = False, suffix = '_test')




def gen_data_RNN(data_file_name, is_training = True, suffix=''):
    ori_data = json.load(open(data_file_name))

    data = []
    for i in range(len(ori_data)):
        doc_item= {}
        for j in ori_data[i].keys():
            item = {}
            layer = ori_data[i][j]['layer']
            item['rel'] = ori_data[i][j]['rel']
            item['head'] = ori_data[i][j]['head']
            item['tail'] = int(j)

            if layer not in doc_item:
                doc_item[layer] = list()
            doc_item[layer].append(item)


        data.append(doc_item)

    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"
    json.dump(data, open(os.path.join(data_path, name_prefix + suffix + '_tree_RNN.json'), 'w'))


gen_data_RNN(train_annotated_tree_name, is_training = False, suffix = '_train')
gen_data_RNN(dev_tree_name, is_training = False, suffix = '_dev')
gen_data_RNN(test_tree_name, is_training = False, suffix = '_test')





# find the SDP between two nodes
def path_find(path1, path2):
    path1 = path1[::-1]
    path2 = path2[::-1]

    temp = 0
    for l1, l2 in zip(path1, path2):
        if (l1 == l2):
            temp += 1

    if temp == min(len(path1),len(path2)):
        temp -= 1

    return path1[temp:], path2[temp:]


def gen_data_LSTM(is_training = True, suffix=''):
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"

    path_len_max = 0

    tree_data = json.load(open(name_prefix + suffix + '_tree.json'))
    ent_data = json.load(open(os.path.join(data_path, name_prefix + suffix + '.json')))

    data = []
    for i in range(len(ent_data)):
        ent_id = {}
        for j in range(len(ent_data[i]['vertexSet'])):
            layer_min = 100
            id_min = 0

            for k in range(ent_data[i]['vertexSet'][j][0]['pos'][0],ent_data[i]['vertexSet'][j][0]['pos'][1]):
                if str(k) in tree_data[i]:
                    if tree_data[i][str(k)]['layer'] < layer_min:
                        layer_min = tree_data[i][str(k)]['layer']
                        id_min = k
            ent_id[j] = id_min

        item_doc = {}
        for m in range(len(ent_id)):
            head = str(ent_id[m])
            item = {}
            for n in range(len(ent_id)):
                if n == m:
                    continue

                tail = str(ent_id[n])
                path_head, path_tail = path_find(tree_data[i][head]['path'], tree_data[i][tail]['path'])
                item[n] = {'path_head': path_head, 'path_tail':path_tail}
                if len(path_head) > path_len_max:
                    path_len_max = len(path_head)
                if len(path_tail) > path_len_max:
                    path_len_max = len(path_tail)
            item_doc[m] = item

        data.append(item_doc)

    print('path_len_max:',path_len_max)

    json.dump(data, open(os.path.join(data_path, name_prefix + suffix + '_tree_LSTM.json'), 'w'))


gen_data_LSTM(is_training = False, suffix = '_train')
gen_data_LSTM(is_training = False, suffix = '_dev')
gen_data_LSTM(is_training = False, suffix = '_test')
