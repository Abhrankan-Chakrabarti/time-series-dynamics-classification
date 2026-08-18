"""Cross-validation of the proposed regimes, including pair-aware LOGO."""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold,LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
from sklearn.base import clone

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'
df=pd.read_csv(OUT/'timeseries_feature_matrix_25.csv'); d=df[df.dataset!='REFERENCE'].copy()
label_map={'alpha':'Persistent_low_frequency','beta':'Persistent_low_frequency','delta':'Intermediate_broadband','gamma':'Intermediate_broadband','kai':'High_frequency_outlier','kappa':'Persistent_low_frequency','lambda':'Persistent_low_frequency','mu':'Persistent_low_frequency','nu':'Intermediate_broadband','phi':'Intermediate_broadband','rho':'Intermediate_broadband','theta':'Intermediate_broadband'}
d['group']=d.signal.str.replace('NLM_sac_ascf_','',regex=False).str.replace('sac_ascf_','',regex=False).map(label_map)
ids=['signal','dataset','group','source_file']; cols=[c for c in d.columns if c not in ids+['n_samples']]
X=d[cols].replace([np.inf,-np.inf],np.nan); y=d.group.to_numpy(); base=d.signal.str.replace('NLM_sac_ascf_','',regex=False).str.replace('sac_ascf_','',regex=False).to_numpy()
model=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('pca',PCA(n_components=.90,svd_solver='full')),('clf',LogisticRegression(C=.5,class_weight='balanced',max_iter=5000,solver='lbfgs',random_state=42))])
svm=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('pca',PCA(n_components=.90,svd_solver='full')),('clf',SVC(C=1,kernel='rbf',class_weight='balanced',random_state=42))])
def pred_cv(est,splitter,X0,y0,g=None):
 p=np.empty(len(y0),object)
 for tr,te in splitter.split(X0,y0,g):
  e=clone(est); e.fit(X0.iloc[tr],y0[tr]); p[te]=e.predict(X0.iloc[te])
 return p
skf=StratifiedKFold(2,shuffle=True,random_state=42); conv=[]
for n,e in [('Logistic_PCA',model),('RBF_SVM_PCA',svm)]:
 p=pred_cv(e,skf,X,y); conv.append([n,accuracy_score(y,p),balanced_accuracy_score(y,p),f1_score(y,p,average='macro')])
conv=pd.DataFrame(conv,columns=['model','accuracy','balanced_accuracy','macro_F1'])
logo=LeaveOneGroupOut(); truth=[]; pred=[]; rows=[]
for fold,(tr,te) in enumerate(logo.split(X,y,base),1):
 held=base[te][0]
 if len(np.unique(y[tr]))<3: rows.append([fold,held,'NOT TESTABLE',np.nan,np.nan,np.nan,'kai is the only high-frequency base signal'])
 else:
  e=clone(model); e.fit(X.iloc[tr],y[tr]); pp=e.predict(X.iloc[te]); truth.extend(y[te]); pred.extend(pp); rows.append([fold,held,'TESTED',accuracy_score(y[te],pp),balanced_accuracy_score(y[te],pp),f1_score(y[te],pp,average='macro'),''])
pair=pd.DataFrame(rows,columns=['fold','held_out','status','accuracy','balanced_accuracy','macro_F1','note'])
# Valid two-class pair-aware test.
m=y!='High_frequency_outlier'; X2=X.loc[m].reset_index(drop=True); y2=y[m]; g2=base[m]; p2=pred_cv(model,logo,X2,y2,g2)
summary=pd.DataFrame([
 ['2-fold stratified','Logistic+PCA',*conv.loc[conv.model=='Logistic_PCA',['accuracy','balanced_accuracy','macro_F1']].iloc[0].tolist(),'Optimistic: SAC/NLM pairs may split'],
 ['2-fold stratified','RBF-SVM+PCA',*conv.loc[conv.model=='RBF_SVM_PCA',['accuracy','balanced_accuracy','macro_F1']].iloc[0].tolist(),'Optimistic: SAC/NLM pairs may split'],
 ['Leave-one-base-out','Logistic+PCA, 3-class',accuracy_score(truth,pred),balanced_accuracy_score(truth,pred),f1_score(truth,pred,average='macro'),'kai-held-out fold is structurally untestable'],
 ['Leave-one-base-out','Logistic+PCA, 2-class',accuracy_score(y2,p2),balanced_accuracy_score(y2,p2),f1_score(y2,p2,average='macro'),'Clean persistent vs intermediate test']
],columns=['validation','model','accuracy','balanced_accuracy','macro_F1','interpretation'])
with pd.ExcelWriter(OUT/'timeseries_cross_validation_results.xlsx') as w:
 summary.to_excel(w,index=False,sheet_name='Summary'); conv.to_excel(w,index=False,sheet_name='Stratified_CV'); pair.to_excel(w,index=False,sheet_name='Pair_Aware_3Class'); pd.DataFrame(confusion_matrix(truth,pred,labels=sorted(set(y))),index=sorted(set(y)),columns=sorted(set(y))).to_excel(w,sheet_name='3Class_Confusion'); pd.DataFrame(confusion_matrix(y2,p2,labels=['Persistent_low_frequency','Intermediate_broadband']),index=['Persistent','Intermediate'],columns=['Persistent','Intermediate']).to_excel(w,sheet_name='2Class_Confusion')
print(summary.to_string(index=False))
