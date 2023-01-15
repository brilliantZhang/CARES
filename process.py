import argparse
import time
import csv
import pickle
import operator
import datetime
import os
import pandas as pd
from sklearn import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()
print(opt)

dataset = 'sample_train-item-views.csv'
if opt.dataset == 'diginetica':
    dataset = 'train-item-views.csv'
    df=pd.read_csv('product-categories.csv',delimiter=';')
    i2c=dict(zip(df.itemId.to_list(),df.categoryId.to_list()))
elif opt.dataset =='yoochoose':
    dataset = 'yoochoose-clicks.dat'
    df = pd.read_csv(dataset,names=['sessionId','timestamp','item_id','category'])
    le = preprocessing.LabelEncoder()
    df['category']=df['category'].replace(0,-1)
    df['category']=df['category'].astype(str)
    df['category']=le.fit_transform(df['category'])+1
    i2c=dict(zip(df.item_id.values,df.category.values))

print("-- Starting @ %ss" % datetime.datetime.now())
with open(dataset, "r") as f:
    if opt.dataset == 'yoochoose':
        reader = csv.DictReader(f, fieldnames=['sessionId','timestamp','item_id','category'], delimiter=',')
    else:
        reader = csv.DictReader(f, delimiter=';')
        
    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        sessid = data['sessionId']
        if curdate and not curid == sessid:
            date = ''
            if opt.dataset == 'yoochoose':
                date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
            else:
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date[curid] = date
        curid = sessid
        if opt.dataset == 'yoochoose':
            item = data['item_id']
        else:
            item = data['itemId'], int(data['timeframe'])
        curdate = ''
        if opt.dataset == 'yoochoose':
            curdate = data['timestamp']
        else:
            curdate = data['eventdate']

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    date = ''
    if opt.dataset == 'yoochoose':
        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
    else:
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
        for i in list(sess_clicks):
            sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
            sess_clicks[i] = [c[0] for c in sorted_clicks]
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
    if len(filseq) < 2:
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

# 7 days for test
splitdate = 0
if opt.dataset == 'yoochoose':
    splitdate = maxdate - 86400 * 1  # the number of seconds for a day：86400
else:
    splitdate = maxdate - 86400 * 7

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
                outcate+=[1]
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
                outcate+=[1]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
        test_cates += [outcate]
        
    return test_ids, test_dates, test_seqs,test_cates


tra_ids, tra_dates, tra_seqs,tra_cates = obtian_tra()
tes_ids, tes_dates, tes_seqs,tes_cates = obtian_tes()

df=pd.DataFrame(tra_dates+tes_dates,columns=['date'])
df['timeid']=df.date.rank(method='dense',ascending=True).astype(int)
time_map=dict(zip(df['date'].unique(),df['timeid'].unique()))

def process_seqs(iseqs, idates,icates,save_item):
    out_seqs = []
    out_dates = []
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
            out_dates += [time_map[date]]
            ids += [id]
        # for i in range(1, len(seq)):
        #     tar = seq[-i]
        #     labs += [tar]
        #     out_seqs += [seq[:-i]]
        #     out_cates += [cate[:-i]]
        #     out_dates += [date]
        #     ids += [id]
    if save_item:
        all_items = pd.DataFrame(all_items,columns=['item'])
        all_items = all_items.item.value_counts().reset_index()
        all_items.columns=['item','item_counts']
        item_dict = dict(zip(all_items.item.values,all_items.item_counts.values))
        pickle.dump(item_dict, open(f'{opt.dataset}_item_dict.pkl', 'wb'))

        # all_cates = pd.DataFrame(all_cates,columns=['item'])
        # all_cates = all_cates.item.value_counts().reset_index()
        # all_cates.columns=['item','item_counts']
        # cate_dict = dict(zip(all_cates.item.values,all_cates.item_counts.values))
        # pickle.dump(cate_dict, open(f'{opt.dataset}_cate_dict.pkl', 'wb'))

        print(len(item_dict.keys()))

    return out_seqs, out_dates, labs, ids,out_cates

print(tra_seqs[:10])

tr_seqs, tr_dates, tr_labs, tr_ids,tr_cates = process_seqs(tra_seqs, tra_dates,tra_cates,True)
te_seqs, te_dates, te_labs, te_ids,te_cates = process_seqs(tes_seqs, tes_dates,tes_cates,False)

print(len(tr_seqs))
print(len(te_seqs))
print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
print(te_seqs[:3], te_dates[:3], te_labs[:3])
all = 0

tra = (tr_seqs, tr_labs, tr_cates)
tes = (te_seqs, te_labs,te_cates)
for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))
if opt.dataset == 'diginetica':
    if not os.path.exists('diginetica'):
        os.makedirs('diginetica')
    
    pickle.dump(tra, open('diginetica/train.txt', 'wb'))
    pickle.dump(tes, open('diginetica/test.txt', 'wb'))
    pickle.dump((tra_seqs,tra_cates), open('diginetica/all_train_seq.txt', 'wb'))
    
elif opt.dataset == 'yoochoose':
    if not os.path.exists('yoochoose1_4'):
        os.makedirs('yoochoose1_4')
    if not os.path.exists('yoochoose1_64'):
        os.makedirs('yoochoose1_64')
    pickle.dump(tes, open('yoochoose1_4/test.txt', 'wb'))
    pickle.dump(tes, open('yoochoose1_64/test.txt', 'wb'))

    split4, split64 = int(len(tr_seqs) / 4), int(len(tr_seqs) / 64)
    print(len(tr_seqs[-split4:]))
    print(len(tr_seqs[-split64:]))

    tra4, tra64 = (tr_seqs[-split4:], tr_labs[-split4:],tr_cates[-split4:]), (tr_seqs[-split64:], tr_labs[-split64:],tr_cates[-split64:])
    seq4, seq64 = (tra_seqs[tr_ids[-split4]:],tra_cates[tr_ids[-split4]:]), (tra_seqs[tr_ids[-split64]:],tra_cates[tr_ids[-split64]:])

    pickle.dump(tra4, open('yoochoose1_4/train.txt', 'wb'))
    pickle.dump(seq4, open('yoochoose1_4/all_train_seq.txt', 'wb'))

    pickle.dump(tra64, open('yoochoose1_64/train.txt', 'wb'))
    pickle.dump(seq64, open('yoochoose1_64/all_train_seq.txt', 'wb'))
else:
    if not os.path.exists('sample'):
        os.makedirs('sample')
    pickle.dump(tra, open('sample/train.txt', 'wb'))
    pickle.dump(tes, open('sample/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('sample/all_train_seq.txt', 'wb'))

print('Done.')