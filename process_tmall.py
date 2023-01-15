import argparse
import time
import csv
import pickle
import operator
import datetime
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tmall', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()
print(opt)
df = pd.read_csv('data_format1/user_log_format1.csv')
df=df[['item_id','cat_id']].drop_duplicates()
i2c=dict(zip(df.item_id.values,df.cat_id.values))

print(df.cat_id.describe())
with open('tmall_data.csv', 'w') as tmall_data:
    with open('tmall/raw/dataset15.csv', 'r') as tmall_file:
        header = tmall_file.readline()
        tmall_data.write(header)
        for line in tmall_file:
            data = line[:-1].split('\t')
            if int(data[2]) > 120000:
                break
            tmall_data.write(line)

print("-- Starting @ %ss" % datetime.datetime.now())
with open('tmall_data.csv', "r") as f:
    reader = csv.DictReader(f, delimiter='\t')
    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        sessid = int(data['SessionId'])
        if curdate and not curid == sessid:
            date = curdate
            sess_date[curid] = date
        curid = sessid
        item = int(data['ItemId'])
        curdate = float(data['Time'])

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    date = float(data['Time'])
    sess_date[curid] = date
print("-- Reading data @ %ss" % datetime.datetime.now())


# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    if len(filseq) < 2 or len(filseq) > 40:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

# Split out test set based on dates
dates = list(sess_date.items())
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

# the last of 100 seconds for test
splitdate = maxdate - 100

print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
tra_sess = filter(lambda x: x[1] < splitdate, dates)
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
print(len(tra_sess))    # 186670    # 7966257
print(len(tes_sess))    # 15979     # 15324
print(tra_sess[:3])
print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
# Convert training sessions to sequences and renumber items to start from 1
# def obtian_tra():
#     train_ids = []
#     train_seqs = []
#     train_dates = []
#     item_ctr = 1
#     for s, date in tra_sess:
#         seq = sess_clicks[s]
#         outseq = []
#         for i in seq:
#             if i in item_dict:
#                 outseq += [item_dict[i]]
#             else:
#                 outseq += [item_ctr]
#                 item_dict[i] = item_ctr
#                 item_ctr += 1
#         if len(outseq) < 2:  # Doesn't occur
#             continue
#         train_ids += [s]
#         train_dates += [date]
#         train_seqs += [outseq]
#     print('item_ctr')
#     print(item_ctr)     # 43098, 37484
#     return train_ids, train_dates, train_seqs

# # Convert test sessions to sequences, ignoring items that do not appear in training set
# def obtian_tes():
#     test_ids = []
#     test_seqs = []
#     test_dates = []
#     for s, date in tes_sess:
#         seq = sess_clicks[s]
#         outseq = []
#         for i in seq:
#             if i in item_dict:
#                 outseq += [item_dict[i]]
#         if len(outseq) < 2:
#             continue
#         test_ids += [s]
#         test_dates += [date]
#         test_seqs += [outseq]
#     return test_ids, test_dates, test_seqs

# tra_ids, tra_dates, tra_seqs = obtian_tra()
# tes_ids, tes_dates, tes_seqs = obtian_tes()
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    train_cates=[]
    item_ctr = 1
   
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        outcate=[]
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
                
            try:
                outcate+=[i2c[int(i)]]
            except:
                outcate+=[183]
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
        train_cates += [outcate]
    print(item_ctr)     # 43098, 37484
    
    return train_ids, train_dates, train_seqs,train_cates


# Convert test sessions to sequences, ignoring items that do not appear in training set

def obtian_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    test_cates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        outcate = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            try:
                outcate+=[i2c[int(i)]]
            except:
                outcate+=[183]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
        test_cates += [outcate]
        
    return test_ids, test_dates, test_seqs,test_cates
tra_ids, tra_dates, tra_seqs,tra_cates = obtian_tra()
tes_ids, tes_dates, tes_seqs,tes_cates = obtian_tes()

def process_seqs(iseqs, idates,icates,save_item):
    out_seqs = []
    out_cates=[]
    labs = []

    ids = []
    all_items = []
    all_cates = []
    for id, seq, date,cate in zip(range(len(iseqs)), iseqs, idates,icates):
        all_items += seq
        all_cates += cate
        for i in range(len(seq)-1,0,-1):
            tar = seq[-i]
            labs += [tar]
            ctar = cate[-i]
            
            out_seqs += [seq[:-i]]
            out_cates += [cate[:-i]]
            ids += [id]
    
    if save_item:
        all_items = pd.DataFrame(all_items,columns=['item'])
        all_items = all_items.item.value_counts().reset_index()
        all_items.columns=['item','item_counts']
        item_dict = dict(zip(all_items.item.values,all_items.item_counts.values))
        pickle.dump(item_dict, open(f'{opt.dataset}_item_dict.pkl', 'wb'))

        print(len(item_dict.keys()))

    return out_seqs,  labs, ids,out_cates

print(tra_seqs[:10])

tr_seqs, tr_labs, tr_ids,tr_cates = process_seqs(tra_seqs, tra_dates,tra_cates,True)
te_seqs, te_labs, te_ids,te_cates = process_seqs(tes_seqs, tes_dates,tes_cates,False)

tra = (tr_seqs, tr_labs, tr_cates)
tes = (te_seqs, te_labs, te_cates)
print('train_test')
print(len(tr_seqs))
print(len(te_seqs))
print(tr_seqs[:3], tr_labs[:3])
print(te_seqs[:3], te_labs[:3])
all = 0

for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('avg length: ', all * 1.0/(len(tra_seqs) + len(tes_seqs)))

if not os.path.exists('tmall'):
    os.makedirs('tmall')
pickle.dump(tra, open('tmall/train.txt', 'wb'))
pickle.dump(tes, open('tmall/test.txt', 'wb'))
pickle.dump((tra_seqs,tra_cates), open('tmall/all_train_seq.txt', 'wb'))

# Namespace(dataset='Tmall')
# Splitting train set and test set
# item_ctr
# 40728
# train_test
# 351268
# 25898
# avg length:  6.687663052493478
