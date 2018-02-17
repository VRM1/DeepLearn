import numpy as np
from tqdm import tqdm
import json
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from DSAE import DSAEB
import gensim
np.random.seed(7)
''' This DSAE model is for using the NON - embedding layer version of the model'''
    
    
# doe sone hot encoding of labels
def encodeLabels(labels):
    
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    encoded_Y = label_encoder.transform(labels)
    # convert integers to dummy variables (i.e. one hot encoded)
    labels = np_utils.to_categorical(encoded_Y)
    return labels

# method to train the model with product reviews.
def GetReviews(graph,doc2vec):
    
    itm_rev_a = []
    itm_rev_b = []
    pairs = []
    labels = []
    for ids in graph:
        ids = ids.split(',')
        itm_rev_a.append(list(doc2vec[ids[0]]))
        itm_rev_b.append(list(doc2vec[ids[1]]))
        pairs.append(ids)
        labels.append(ids[2])
    itm_rev_a = np.array(itm_rev_a)
    itm_rev_b = np.array(itm_rev_b)
    return (itm_rev_a,itm_rev_b,pairs,labels)

# method that return item-item relationship graph (i.e. substitute, complements etc)
def GetItmGraphs(name,typ):
    
    if typ=='binary':
        end = '_2class_filtered.json'
    else:
        end = '_4class_filtered.json'
    with open('dataset/'+name+end,'r') as fp:
        itm_pairs = json.load(fp)
    # we have two cases now (a) substitute and (b) complement
    subs,compl,=[],[]
    for itms in tqdm(itm_pairs):
        tmp_pairs = [i.strip() for i in itms.strip('(|)').split(',')]
        if itm_pairs[itms] == 1:
            subs.append(','.join(tmp_pairs)+',1')
        else:
            compl.append(','.join(tmp_pairs)+',0')
    subs = subs[:2500000]
    compl = compl[:5000000]
    data = subs + compl
    return data
    
def TrainAmazon(name,batch_size,z_dim,epochs,ld_weight,typ):
    
    # load the d2v trained model, which will serve as input to VAE
    d2v_model = gensim.models.doc2vec.Doc2Vec.load('dataset/'+name+'_reviews'+'.d2v')
#     main_data = [list(d2v_model.docvecs[i]) for i in xrange(len(d2v_model.docvecs))]
    print("Model loaded")
    #  maximum length of each document review
    inpt_dim = len(d2v_model[0])
    # total number of unique words
    inter_dim = 256
    graph_data = GetItmGraphs(name,typ)
    print 'total item pairs:{}, total item with reviews:{}'\
            .format(len(graph_data),len(d2v_model.docvecs))
    # 3K samples for train and 1k for test
    train,test = train_test_split(graph_data,train_size=0.75)
    train,validation = train_test_split(train,train_size=0.80)
    print 'train size:{}, validation size:{}, test size:{}'.format(len(train),len(validation),len(test))
    ''' get the reviews corresponding to the selected 
    item pairs and their labels (substitutes and complements)'''
    train_a,train_b,pairs_train,labels = GetReviews(train,d2v_model)
    valid_a,valid_b,pairs_valid,valid_labels = GetReviews(validation,d2v_model)
    test_a,test_b,pairs_test,test_labels = GetReviews(test,d2v_model)
#     print '# test dataset:{} items'.format(len(test))
#     encoded_revs = [train[itm] for itm in sorted(train.keys())]
    labels = encodeLabels(labels)
    valid_labels = encodeLabels(valid_labels)
    labels_test = encodeLabels(test_labels)
    # get the models
    dsae = DSAEB(name,inter_dim, z_dim, inpt_dim,batch_size)
    DSAE,LinkPredictor,cp = dsae.getModels()
    if ld_weight == 'yes':
        weights_path = 'output/DSAE_'+name+'.hdf5'
        DSAE.load_weights(weights_path)
        print 'loaded pre-trained weights'
    else:
        # train the model
        DSAE.fit([np.array(train_a),np.array(train_b)],[np.array(train_a),np.array(train_b),labels], \
                 shuffle=True, nb_epoch=epochs, batch_size=batch_size, \
                 validation_data=[[np.array(valid_a),np.array(valid_b)],\
                                  [np.array(valid_a),np.array(valid_b),valid_labels]], \
                 callbacks=cp)
      
    
    results = LinkPredictor.evaluate(x=[np.array(test_a),np.array(test_b)],y=[labels_test])
    print results
     

if __name__ == '__main__':
    
    ''' max length of each review is 200 after the filtration.
    So, we use this for padding.'''
    data_typ = ['Musical_Instruments','Electronics','Movies_and_TV','Books']
    batch_size=512
    epochs = 70
    z_dim=100
    # the type of class label (1-> subs, 2-> complement, or the 4 type of classes with direction)
    typ='binary'
    name = data_typ[2]
    ld_weight = 'no'
    TrainAmazon(name,batch_size,z_dim,epochs,ld_weight,typ)