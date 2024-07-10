#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries yang dibutuhkan 
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#import csv dari lokal komputer (file csv)
df = pd.read_csv('D:\Materi Data Science\Project Mini Course\Project 3\marketing_campaign_data.csv')
df.sample(10)


# In[3]:


# Deteksi dan hitung kolom yang null
df.isna().sum()


# - Terdapat 24 nilai null pada fitur Income, maka dari itu kita isi dengan nilai rata-rata (mean)

# ### Handle Missing Value

# In[4]:


# Mengisi nilai mean pada fitur Income
df.fillna(df['Income'].median(), inplace=True)
df.isna().sum()


# In[5]:


#Periksa duplikat data
jlh_data = df.shape[0]
duplikat = df.duplicated().sum()

# Tampilkan hasilnya
print(f'Jumlah data awal = {jlh_data} dengan total duplikasi data = {duplikat}')


# - Tidak terdapat data duplikat

# ### Handling Outlier

# In[6]:


# Cek Outlier Kolom Numerik
col_numeric = df.select_dtypes(include=['int64', 'float64']).columns.tolist()


# In[7]:


# cek outlier fitur numerikal
plt.figure(figsize=(12, 15))
for i in range(0, len(col_numeric)):
    plt.subplot(6, 5, i+1)
    sns.boxplot(y=df[col_numeric[i]], orient='v')
    plt.tight_layout()


# In[8]:


#Handle Outlier menggunakan z.score
# List fitur yang akan di-handle outliernya
features = [
    'Year_Birth', 'Income', 'MntMeatProducts', 'MntSweetProducts', 'MntGoldProds', 
    'NumDealsPurchases', 'NumCatalogPurchases','NumWebVisitsMonth'
]


# In[9]:


# Filter Outlier
print(f'Jumlah baris sebelum memfilter outlier: {len(df)}')

filtered_entries = np.array([True] * len(df))
for col in features:
    zscore = np.abs(stats.zscore(df[col]))  # Menghitung absolute Z-score
    filtered_entries = (zscore < 3) & filtered_entries

df = df[filtered_entries]

print(f'Jumlah baris setelah memfilter outlier: {len(df)}')


# In[10]:


df.info()


# In[10]:


# cek outlier fitur numerikal setelah di filter
plt.figure(figsize=(12, 15))
for i in range(0, len(col_numeric)):
    plt.subplot(6, 5, i+1)
    sns.boxplot(y=df[col_numeric[i]], orient='v')
    plt.tight_layout()


# #### Membuat Tabel Umur

# In[11]:


#membuat tabel umur
df['umur'] = 2024 - df['Year_Birth']

unique_umur = sorted(df['umur'].unique())
print(unique_umur)


# ### Feature Engineering

# ##### Fitur Membership_Duration

# In[12]:


# Mengubah kolom Dt_Customer menjadi datetime dengan format DD-MM-YYYY
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)


# In[16]:


#Feature Engineering untuk Membership_Duration = tahun_sekarang - Dt_Customer
df['Membership_Duration'] = 2024 - df['Dt_Customer'].dt.year
df['Membership_Duration'].value_counts()


# ##### Fitur Cat_Age

# In[15]:


#Feature Engineering untuk umur (Cat_Age) = tahun_sekarang - Year_Birth
bins = [17, 35, 55, 100]
labels = ['Remaja', 'Dewasa', 'Lansia' ]

# Menggunakan pd.cut untuk mengelompokkan umur
df['Cat_Age'] = pd.cut(df['umur'], bins=bins, labels=labels, right=False)

# Menampilkan DataFrame dengan kelompok umur
df['Cat_Age'].value_counts()


# ##### Fitur Total_Children

# In[17]:


#Feature Engineering untuk Total_Children = Kidhome + Teenhome
df['Total_Children'] = (df['Kidhome'] + df['Teenhome'])

#Menampilkan dataframe Total_Children
df['Total_Children'].value_counts()


# ##### Fitur Total_Transaction

# In[18]:


#Feature Engineer untuk Total_Transaction = NumDealsPurchase + NumWebPurchase + NumCatlogPuchase + NumStorePurchase
df['Total_Transaction'] = df['NumDealsPurchases'] + df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases']
#Menampilkan
df['Total_Transaction'].value_counts()


# ##### Fitur Total_Spending

# In[19]:


# Feature Engineering untuk Total_Spending = MntCoke + MntFruits + MntMeatProducts + MntFishProductss + MntSweetProducts + MntGoldProds
df['Total_Spending'] = df['MntCoke'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']

#Menampilkan
df['Total_Spending'].value_counts()


# ##### Total_Accepted_Campaign

# In[20]:


#Feature Engineer untuk Total_Accepted_Campaign = AcceptedCmp1 + AcceptedCmp2 + AcceptedCmp3 + AcceptedCmp4 + AcceptedCmp5 
df['Total_Accepted_Campaign'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5'] + df['Response']

#Menampilkan
df['Total_Accepted_Campaign'].value_counts()


# ##### Feature Conversion Rate (CVR)

# In[21]:


# Menghitung conversion rate untuk setiap entri
df['CVR'] = (df['Total_Transaction'] / df['NumWebVisitsMonth'])

# Menangani kemungkinan nilai NaN atau infinit jika ada NumWebVisitsMonth = 0
df['CVR'].replace([float('inf'), -float('inf')], 0, inplace=True)
df['CVR'].fillna(0, inplace=True)

# Melihat beberapa baris pertama dari dataset yang sudah dimodifikasi
df['CVR'].value_counts()


# In[22]:


# Pilih kolom yang relevan untuk analisis, misalnya 'Marital_Status' dan 'Education'
categories = ['Marital_Status', 'Education']

# Menghitung conversion rate untuk setiap kategori
conversion_rates = {}
for category in categories:
    conversion_rate_by_category = df.groupby(category).apply(lambda x: (x['Response'].sum() / x['NumWebVisitsMonth'].sum()) * 100)
    conversion_rates[category] = conversion_rate_by_category



# In[23]:


# Sorting untuk 'Marital_Status'
marital_status_order = ['Lajang', 'Bertunangan', 'Menikah', 'Janda', 'Duda']
df['Marital_Status'] = pd.Categorical(df['Marital_Status'], categories=marital_status_order, ordered=True)

# Sorting untuk 'Education'
education_order = ['SMA', 'D3', 'S1', 'S2', 'S3']
df['Education'] = pd.Categorical(df['Education'], categories=education_order, ordered=True)


# In[24]:


#Membuat Plot
plt.figure(figsize=(14, 7))
for i, (category, rates) in enumerate(conversion_rates.items(), 1):
    plt.subplot(1, 2, i)
    sns.barplot(x=rates.index, y=rates.values, order=marital_status_order if category == 'Marital_Status' else education_order)
    plt.title(f'Conversion Rate by {category}')
    plt.ylabel('Conversion Rate (%)')
    plt.xlabel(category)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# In[25]:


# Menghitung rata-rata CVR untuk setiap kelompok umur
age_group_cvr = df.groupby('Cat_Age')['CVR'].mean().reset_index()

# Membuat plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Cat_Age', y='CVR', data=age_group_cvr)
plt.title('Conversion Rate by Age Group')
plt.xlabel('Cat Age')
plt.ylabel('Average Conversion Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()

# Menampilkan plot
plt.show()


# In[26]:


# Menghitung rata-rata CVR untuk setiap umur
age_group_cvr1 = df.groupby('umur')['CVR'].mean().reset_index()

#Membuat Scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='umur', y='CVR', data=age_group_cvr1, s=100)
sns.regplot(x='umur', y='CVR', data=df, scatter=False, color='blue')
plt.title('Conversion Rate vs. Age')
plt.xlabel('Age')
plt.ylabel('Conversion Rate')
plt.show()


# ## Feature Encoding

# In[27]:


num_fe = [
    'Total_Children','Total_Spending', 'Total_Transaction', 'Total_Accepted_Campaign', 'CVR'
]


# In[72]:


#Cek Outlier Fitur Engineering
plt.figure(figsize=(12, 15))
for i in range(0, len(num_fe)):
    plt.subplot(6, 5, i+1)
    sns.boxplot(y=df[num_fe[i]], orient='v')
    plt.tight_layout()


# In[74]:


# Filter Outlier
print(f'Jumlah baris sebelum memfilter outlier: {len(df)}')

filtered_entries2 = np.array([True] * len(df))
for col in num_fe:
    zscore = np.abs(stats.zscore(df[col]))  # Menghitung absolute Z-score
    filtered_entries = (zscore < 3) & filtered_entries2

df = df[filtered_entries2]

print(f'Jumlah baris setelah memfilter outlier: {len(df)}')


# In[63]:


#Cek Outlier Fitur Engineering
plt.figure(figsize=(12, 15))
for i in range(0, len(num_fe)):
    plt.subplot(6, 5, i+1)
    sns.boxplot(y=df[num_fe[i]], orient='v')
    plt.tight_layout()


# - fitur Education menggunakan Label Encoding
# - fitur Marital_Status dan Cat_Age menggunakan One Hot Encoding

# In[28]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[29]:


#Label Encoding untuk Education
mapping_education = {
    'SMA' : 0,
    'D3' : 1,
    'S1' : 2,
    'S2' : 3,
    'S3' : 4
}
df['Education'] = df['Education'].map(mapping_education)


# In[30]:


# 2. One Hot Encoding untuk Marital_Status dan Cat_Age menggunakan pd.get_dummies
marital_status_dummies = pd.get_dummies(df['Marital_Status'], prefix='Status')
cat_age_dummies = pd.get_dummies(df['Cat_Age'], prefix='Age')

# Menggabungkan hasil encoding dengan dataframe asli
df = df.drop(columns=['Marital_Status', 'Cat_Age'])
df = pd.concat([df, marital_status_dummies, cat_age_dummies], axis=1)


# In[31]:


df.info()


# In[32]:


df['Education'].value_counts()


# In[33]:


df['Education'] = df['Education'].astype(int)


# In[30]:


df.info()


# In[34]:


df.head(5)


# In[35]:


df_scale = df.copy()


# In[36]:


X = df_scale.drop(columns=['Unnamed: 0', 'ID', 'Year_Birth', 'Education', 'Status_Lajang', 'Status_Bertunangan','Status_Menikah', 'Status_Janda', 'Status_Duda',  'Age_Remaja', 'Age_Dewasa', 'Age_Lansia', 'Dt_Customer', 'Response', 'Complain'])


# In[37]:


X.info()


# In[38]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


# In[39]:


X.head(10)


# # MODELING

# In[40]:


df_model = X.copy()


# ##### PCA

# In[41]:


from sklearn.decomposition import PCA 

# fit pca
pca = PCA(n_components = 2)
pca.fit(df_model)

# pca transformed 
data_pca = pca.transform(df_model)


# In[42]:


data_pca1 = pd.DataFrame(data_pca)


# ##### Mencari Cluster Terbaik 

# In[43]:


# mencari nilai k optimal dengan parameter inertia
from sklearn.cluster import KMeans

inertia = []
k_values = range(1,11)

# fit model
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=123)
    kmeans.fit(data_pca1)
    inertia.append(kmeans.inertia_)


# In[45]:


pip install yellowbrick


# In[44]:


# Visualisasi grafik elbow method
from yellowbrick.cluster import KElbowVisualizer
plt.figure(figsize=(10, 5))

# plot grafik
plt.plot(k_values, inertia ,color='#91bfdb', linewidth= 2.5, marker='o', markerfacecolor='#fc8d59', markersize=10)
plt.title('Inertia Score Elbow for KMeans Clustering', fontsize=14)
plt.xlabel('K',fontsize=12)
plt.ylabel('Inertia',fontsize=12)


# In[45]:


# visualisasi dengan parameter distortion
from yellowbrick.cluster import KElbowVisualizer

# fit model
model = KMeans(random_state=123)
visualizer = KElbowVisualizer(model, metric='distortion', timings=True, locate_elbow=True)
visualizer.fit(data_pca1)       
visualizer.show() 


# - Berdasarkan Distortion Score dan Elbow methode didapatkan jumlah cluster terbaik adalah 4 Cluster

# ##### Clustering Menggunakan K-Means

# In[46]:


from sklearn.cluster import KMeans

# fit model
kmeans = KMeans(n_clusters=4, random_state = 123)
kmeans.fit(data_pca1.values)
data_pca1['cluster'] = kmeans.labels_


# In[47]:


data_pca1.sample(10)


# In[48]:


# Mendefinisikan palet warna
color =  ['steelblue', 'indianred', 'grey', 'orange', 'olive']

# Membuat visualisasi hasil segmentasi klaster
fig, ax = plt.subplots(figsize=(10, 7))
sns.scatterplot(data=data_pca1, x=0, y=1, hue='cluster', palette=color, ax=ax)
plt.title('Cluster Segmentation')
plt.show()


# #### Evaluation

# In[49]:


data_pca1.columns = data_pca1.columns.astype(str)


# In[50]:


from sklearn.metrics import silhouette_score

def visualize_silhouette_layer(data):
    clusters_range = range(2,10)
    results = []

    for i in clusters_range:
        km = KMeans(n_clusters=i, random_state=123)
        cluster_labels = km.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        results.append([i, silhouette_avg])

    result = pd.DataFrame(results, columns=["n_clusters", "silhouette_score"])
    pivot_km = pd.pivot_table(result, index="n_clusters", values="silhouette_score")

    plt.figure(figsize=(5,4))
    sns.heatmap(pivot_km, annot=True, linewidths=1, fmt='.3f', cmap='coolwarm')
    plt.tight_layout()
    plt.title('Silhouette Score of K-means Clustering')
    plt.show()


# In[51]:


visualize_silhouette_layer(data_pca1)


# ## Customer Personality Analysis for Marketing Retargeting

# In[52]:


df['Cluster'] = kmeans.labels_


# In[53]:


feats = ['Recency','Total_Transaction','Total_Spending', 'Income','Cluster','CVR', 'umur']
data_summary = df[feats]


# In[97]:


round(data_summary.groupby('Cluster').agg(['median']),2).round()


# 1. Cluster 0 ( High Spender )
# - Lama Belanja kembali sekitar 54 hari
# - Total Transaksi yang dilakukan sebanyak 20 transaksi
# - Total Spending yang paling besar sebesar 1365000
# - Income yang didapat paling besar sebesar 76800000
# - Konversi paling besar sebanyak 9 %
# - Umur sekitar 55 Tahun
# 
# 2. Cluster 1 ( Risk Of Turn )
# - Lama belanja kembali sekitar 49 hari
# - Total Transaksi paling kecil hanya 7 transaksi
# - Total Spending dan Income paling kecil
# - CVR yang dihasilkan paling kecil hanya 1%
# - Umur sekitar 51 Tahun
# 
# 3. Cluster 2 ( Mid Spender )
# - Lama belanja kembali sekitar 50 Hari
# - Total Transaksi sebanyak 23 transaksi, paling besar
# - Total Spending dan Income paling besar Kedua
# - CVR yang didapat sebesar 5 % Terbesar kedua
# - Umur Sekitar 59 Tahun
# 
# 4. Cluster 3 ( Low Spender )
# - Lama belanja kembali sekitar 50 hari
# - Total Transaksi diatas Cluster 1, paling rendah kedua
# - Total Spending dan Income paling rendah, namun masih diatas Cluster 1
# - CVR yang didapat 3%
# - Umur sekitar 58 Tahun

# In[55]:


def dist_list(lst):
    plt.figure(figsize=[12, 7])
    i = 1
    for col in lst:
        ax = plt.subplot(2, 3, i)
        ax.vlines(df[col].median(), ymin=-1, ymax=4, color='black', linestyle='--')
        g = df.groupby('Cluster')
        x = g[col].median().index
        y = g[col].median().values
        ax.barh(x, y, color=color) 
        plt.title(col)
        i = i + 1

dist_list(['Total_Transaction','Total_Spending', 'Income', 'Recency', 'CVR', 'umur'])
plt.tight_layout()
plt.show()


# In[56]:


# persentase total customer setiap cluster
cluster_count = data_summary['Cluster'].value_counts().reset_index()
cluster_count.columns = ['Cluster', 'count']
cluster_count['percentage (%)'] = round((cluster_count['count']/len(data_summary))*100,2)
cluster_count = cluster_count.sort_values(by=['Cluster']).reset_index(drop=True)
cluster_count


# In[57]:


#visualisasi persentase customer pada setiap cluster
fig, ax = plt.subplots(figsize=(10,6))

bars = plt.bar(x=cluster_count['Cluster'], height= cluster_count['percentage (%)'], color=color)

for bar in bars:
    height = bar.get_height()
    label_x_pos = bar.get_x() + bar.get_width() / 2
    ax.text(label_x_pos, height, s=f'{height} %', ha='center',va='bottom')

plt.title('Percentage of Customer by Cluster')
plt.xticks(range(0,4))


# In[100]:


variable = [ 'NumWebVisitsMonth', 'Total_Accepted_Campaign', 'Membership_Duration', 'NumDealsPurchases','NumStorePurchases']


# In[101]:


# distribusi cluster
plt.figure(figsize=(15, 20))
for i in range(0, len(variable)):
    plt.subplot(5, 2, i+1)
    sns.boxenplot(x='Cluster', y=df[variable[i]], data=df, palette=color).set(title=f'Customer\'s {variable[i]} Distribution by Cluster')
    plt.tight_layout()


# In[ ]:




