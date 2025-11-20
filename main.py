import argparse
import os
import time
import torch
from models import  Model1
from utils import setup_seed, load_protein, gen_trte_adj_mat, load_data
import pandas as pd
from loss import InstanceLoss, post_proC, loss_cs, instanceloss, thrC
from models import Encoder, Decoder, Model2

def train1(incom_omic_dic, incom_graph_dic):
    model1.train()
    optim.zero_grad()
    z,ae_representations,gcn_representations=model1(incom_omic_dic, incom_graph_dic)

    ins1_loss = sum(ins_criterion(z, aer) for aer in ae_representations)
    ins2_loss = sum(ins_criterion(z, gcnr) for gcnr in gcn_representations)
    i_loss = criterion(z, true_protein)
    i_loss = torch.sqrt(i_loss)

    total_loss = i_loss + alpha1* ins1_loss+ alpha2 * ins2_loss
    total_loss.backward()
    optim.step()
    return  total_loss,z
def train2(x,adj,n_clusters):
    model2.train()
    optimizer.zero_grad()
    H, CH, Coefficient, X_,y,W= model2(x, adj)

    cs_loss=loss_cs(y,W.detach())
    rec_loss = torch.sum(torch.pow(x-X_, 2))
    loss_instance = instanceloss(H, CH)
    loss_coef = torch.sum(torch.pow(Coefficient, 2))

    all_loss = loss_coef*0.01+ +loss_instance*0.01+rec_loss*0.01+cs_loss*0.01
    all_loss.backward()
    optimizer.step()
    return  all_loss.item(),Coefficient

cuda = True if torch.cuda.is_available() else False
dataType = ['rna', 'meth','CN','miRNA']
cancer = ['BLCA','BRCA','KIRC','LUAD','SKCM','STAD','UCEC','UVM','GBM','PAAD' 'GBM']
cancer_dict = {'BRCA': 5, 'BLCA': 5, 'KIRC': 4,'LUAD': 3,'SKCM': 4
               ,'STAD': 3, 'UCEC': 4, 'UVM': 4,'GBM': 3,'PAAD': 2}
data_fold ='./datasets/'
fea_sample_name=object()
sub_name='BLCA'
data_seed=1
setup_seed(data_seed)
alpha1=0.01
alpha2=0.01

omic_dic = {}
incom_omic_dic={}
graph_dic = {}
incom_graph_dic = {}
omic_dims= {}

num_class = 32
############## load data/graph #####################
true_protein,protein_sample_name=load_protein(sub_name)
true_protein = true_protein.cuda()

for type in dataType:
    omic_dic[type] = load_data(type, sub_name)
    sample_names=omic_dic[type].index


    common_samples = sample_names.intersection(protein_sample_name)
    incom_omic_dic[type] = omic_dic[type].loc[common_samples]
    incom_omic_dic[type] = torch.FloatTensor(incom_omic_dic[type].values.astype(float))
    omic_dims[type] = omic_dic[type].shape[1]  # 使用累积添加

    incom_graph_dic[type] = gen_trte_adj_mat(incom_omic_dic[type], num_class)

    incom_omic_dic[type] = incom_omic_dic[type].cuda()
    incom_graph_dic[type] = incom_graph_dic[type][0].cuda()

    omic_dic[type] = torch.FloatTensor(omic_dic[type].values.astype(float))
    graph_dic[type] = gen_trte_adj_mat(omic_dic[type], num_class)
    omic_dic[type] = omic_dic[type].cuda()
    graph_dic[type] = graph_dic[type][0].cuda()

dim_final = true_protein.shape[1]

############## Model 1 #####################
dropout1 = 0.1
lr_e =1e-5
ae_n_enc_1 = 256
ae_n_enc_2 = 128
n_gcn_1 = 128

torch.autograd.set_detect_anomaly(True)
criterion = torch.nn.MSELoss()  # MSE-loss
model1 = Model1(omic_dims, ae_n_enc_1, ae_n_enc_2, n_gcn_1, dim_final, dropout1)
optim = torch.optim.Adam(
    list(model1.parameters()), lr=lr_e)
if cuda:
    model1.cuda()
print("\n############## Training Model1 ##############\n")
epoch = 300
ins_criterion = InstanceLoss(true_protein.shape[0],0.1,"cuda:0")
for i in range(epoch):
    total_loss,_=train1(incom_omic_dic, incom_graph_dic)
    if (i + 1) % 10 == 0:
        print('epoch[', i +1, '/', epoch, ']','total_loss=',total_loss.item())

############## generate complete protein/consistency representation #######################
final_pro,_,_=model1(omic_dic, graph_dic)


############## Model 2 #######################
parser = argparse.ArgumentParser()

parser.add_argument("-learning_rate", type=float, default=0.00001, help="learning rate")
parser.add_argument("-epochs", type=int, default=500, help="number of epochs")
parser.add_argument("-weight_decay", type=float, default=1e-5, help="weight decay")
parser.add_argument("-tau", type=float, default=1, help="temperature")
parser.add_argument("-n", dest='cluster_num', type=int, default=-1, help="cluster number")
parser.add_argument('--q', type=float, default=0.5, help='A parameter q')
args = parser.parse_args()
dropout2 = 0.1
args.cluster_num = cancer_dict[sub_name]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
graph = gen_trte_adj_mat(final_pro, args.cluster_num)[0]
final_pro = final_pro.to(device)
graph = graph.to(device)

criterion_instance = InstanceLoss(len(final_pro), args.tau, device).to(device)
encoder = Encoder(final_pro.shape[1], 128, dropout2).to(device)
decoder = Decoder(128, final_pro.shape[1], dropout2).to(device)
model2 = Model2(encoder, decoder, len(final_pro), 128, cancer_dict[sub_name]).to(device)
optimizer = torch.optim.Adam(
    model2.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay
)
alpha = max(0.4 - (args.cluster_num - 1) / 10 * 0.1, 0.1)
loss_list=[]
print('\n############## Training Model 2 #####################\n')
for epoch in range(args.epochs):
    all_loss, C = train2(final_pro, graph, n_clusters=args.cluster_num)#
    loss_list.append(all_loss)
    if ((epoch + 1 )% 10 == 0):
        print(f'Epoch=[ {epoch+1:03d}/{args.epochs:03d} ],all_loss={all_loss:.4f}')

C = C.detach().cpu().numpy()
commonZ = thrC(C, alpha)    # Threshold

y_pred, L = post_proC(commonZ, args.cluster_num)

df_cluster = pd.DataFrame({
    'sample_names': sample_names,
    'y_pred': y_pred
})
output_dir='./results/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
out_file = output_dir+str(sub_name)+'.csv'
df_cluster.to_csv(out_file, index=False)
print("=================== Final =====================")





