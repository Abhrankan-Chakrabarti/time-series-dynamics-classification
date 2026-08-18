"""PCA, optional UMAP, K-means and hierarchical clustering."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"results"
df=pd.read_csv(OUT/"timeseries_feature_matrix_25.csv")
# The exploratory classification labels are included here for reproducibility.
label_map={
 "alpha":"Persistent_low_frequency","beta":"Persistent_low_frequency",
 "delta":"Intermediate_broadband","gamma":"Intermediate_broadband",
 "kai":"High_frequency_outlier","kappa":"Persistent_low_frequency",
 "lambda":"Persistent_low_frequency","mu":"Persistent_low_frequency",
 "nu":"Intermediate_broadband","phi":"Intermediate_broadband",
 "rho":"Intermediate_broadband","theta":"Intermediate_broadband"}
for i,r in df.iterrows():
    if r.dataset != "REFERENCE":
        base=r.signal.replace("NLM_sac_ascf_","").replace("sac_ascf_","")
        df.loc[i,"group"]=label_map.get(base,"UNASSIGNED")

ids=["signal","dataset","group","source_file"]
cols=[c for c in df.columns if c not in ids+['n_samples']]
X=df[cols].replace([np.inf,-np.inf],np.nan).fillna(df[cols].median())
Z=StandardScaler().fit_transform(X)
pca=PCA(); XP=pca.fit_transform(Z); p2=PCA(2); PC=p2.fit_transform(Z)

try:
 import umap.umap_ as umap
 U=umap.UMAP(n_neighbors=7,min_dist=.15,n_components=2,random_state=42).fit_transform(Z)
except Exception:
 U=PC.copy()

kmrows=[]
for k in range(2,7):
 lab=KMeans(n_clusters=k,n_init=50,random_state=42).fit_predict(Z)
 kmrows.append([k,silhouette_score(Z,lab),adjusted_rand_score(df.group,lab)])
kdf=pd.DataFrame(kmrows,columns=['k','silhouette','ARI_vs_predefined_groups'])
best=int(kdf.loc[kdf.silhouette.idxmax(),'k'])
kl=KMeans(n_clusters=best,n_init=100,random_state=42).fit_predict(Z)
al=AgglomerativeClustering(n_clusters=3).fit_predict(Z)
res=df[ids].copy(); res[['PC1','PC2']]=PC; res[['UMAP1','UMAP2']]=U; res[f'KMeans_k{best}']=kl; res['Agglomerative_k3']=al
load=pd.DataFrame(p2.components_.T,index=cols,columns=['PC1_loading','PC2_loading'])
with pd.ExcelWriter(OUT/'timeseries_PCA_UMAP_clustering.xlsx') as w:
 res.to_excel(w,index=False,sheet_name='Embeddings_Clusters'); kdf.to_excel(w,index=False,sheet_name='KMeans_Evaluation'); load.to_excel(w,sheet_name='PCA_Loadings')
 pd.DataFrame({'PC':np.arange(1,len(pca.explained_variance_ratio_)+1),'explained_variance_ratio':pca.explained_variance_ratio_,'cumulative_variance':np.cumsum(pca.explained_variance_ratio_)}).to_excel(w,index=False,sheet_name='PCA_Variance')

def plot(a,b,title,xlab,ylab,outfile):
 plt.figure(figsize=(10,7))
 for g in df.group.unique():
  m=df.group.values==g; plt.scatter(a[m],b[m],label=g,s=55)
 plt.xlabel(xlab); plt.ylabel(ylab); plt.title(title); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(OUT/outfile,dpi=180); plt.close()
plot(PC[:,0],PC[:,1],'PCA of 25 Time-Series Feature Vectors',f'PC1 ({p2.explained_variance_ratio_[0]*100:.1f}%)',f'PC2 ({p2.explained_variance_ratio_[1]*100:.1f}%)','timeseries_PCA.png')
plot(U[:,0],U[:,1],'UMAP of 25 Time-Series Feature Vectors','UMAP 1','UMAP 2','timeseries_UMAP.png')
plt.figure(figsize=(14,7)); dendrogram(linkage(Z,method='ward'),labels=df.signal.str.replace('sac_ascf_','',regex=False).str.replace('NLM_sac_ascf_','NLM_',regex=False),leaf_rotation=90,leaf_font_size=8); plt.tight_layout(); plt.savefig(OUT/'timeseries_dendrogram.png',dpi=180); plt.close()
