from tkinter import E
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import os
from sklearn.cluster import MeanShift, estimate_bandwidth
import glob
from tqdm import tqdm


# --------------------------------------------------------------------- #
# <1> 
# clustering by using Photo.csv - geometric coordinates
# --------------------------------------------------------------------- #

# photo_info = pd.read_csv(1
#     'Photo.csv', 
#     names=["photo_id", "user_id", "latitude", "longitude", "taken_time", "null"]
#     )

# print(photo_info.head())
# # print(photo_info.columns)

# # photo_id = photo_info[0]
# # user_id = photo_info[1]
# # coor_latitude = photo_info[2]
# # coor_longitude = photo_info[3]
# # taken_time = photo_info[4]

# # TODO 
# # sklearn.Meanshift

# coords = photo_info.values[:, [2, 3]]
# bandwidth = estimate_bandwidth(coords, quantile=0.2, n_samples=500)
# clustering_coords = MeanShift(bandwidth=bandwidth, bin_seeding=True)
# clustering_coords.fit(coords)

# labels = clustering_coords.labels_
# labels_unique = np.unique(labels)
# cluster_centers = clustering_coords.cluster_centers_
# n_clusters_ = len(labels_unique)

# np.save('clustering_coords.npy', clustering_coords)
# # print(clustering_longitude)

# --------------------------------------------------------------------- #
# <1 - 2> 
# clustering by using Photo.csv - geometric coordinates -> plot results
# --------------------------------------------------------------------- #

# # import matplotlib.pyplot as plt
# # from itertools import cycle

# # plt.figure(1)
# # plt.clf()

# # colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")

# # for k, col in zip(range(n_clusters_), colors):
# #     my_members = labels == k
# #     cluster_center = cluster_centers[k]
# #     plt.plot(coords[my_members, 0], coords[my_members, 1], col + ".")
# #     plt.plot(
# #         cluster_center[0],
# #         cluster_center[1],
# #         "o",
# #         markerfacecolor=col,
# #         markeredgecolor="k",
# #         markersize=14,
# #     )
# # plt.title("Estimated number of clusters: %d" % n_clusters_)
# # plt.show()
# # plt.savefig('cl')

# --------------------------------------------------------------------- #
# <2> filtering ::  Photos.csv || Given Photo images
# --------------------------------------------------------------------- #

# photo_data = glob.glob('bundler-v0.3-binary/bundler-v0.3-binary/Photos/*.jpg')
# photo_data_ids = []
# for photo_data_name in photo_data:
#     img_name = os.path.basename(photo_data_name)
#     photo_data_ids.append(int(img_name.split('.')[0]))

# new_info = pd.DataFrame(columns=["photo_id", "longitude", "latitude"])

# for photo_data_id in tqdm(photo_data_ids):
#     row_idx = list(photo_info.values[:, 0]).index(photo_data_id)
#     # print(row_idx)
#     row = photo_info[["photo_id", "longitude", "latitude"]].loc[row_idx]
#     # new_info = pd.concat([new_info, row])
#     new_info = new_info.append(row)

#     # print(new_info)
    
# new_info.to_csv('filterd_photos.csv')

# --------------------------------------------------------------------- #
# <3> filterd data --> clustering
# --------------------------------------------------------------------- #
# new_info = pd.read_csv('filterd_photos.csv', usecols=[1, 2, 3])
# print(new_info.head())

# coords = new_info.values[:, [1, 2]]
# print(coords.shape)

# bandwidth = estimate_bandwidth(coords, quantile=0.2, n_samples=500)
# clustering_coords = MeanShift(bandwidth=bandwidth, bin_seeding=True)
# clustering_coords.fit(coords)

# labels = clustering_coords.labels_
# labels_unique = np.unique(labels)
# cluster_centers = clustering_coords.cluster_centers_
# n_clusters_ = len(labels_unique)

# np.save('filterd_clustering_coords.npy', clustering_coords)
# print(clustering_longitude)

# --------------------------------------------------------------------- #
# <4> filterd data --> clustering --> visualization
# --------------------------------------------------------------------- #

# import matplotlib.pyplot as plt
# from itertools import cycle

# plt.figure(1)
# plt.clf()

# colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")

# for i, (k, col) in enumerate(zip(range(n_clusters_), colors)):
#     my_members = labels == k
#     cluster_center = cluster_centers[k]

#     custom_info = pd.DataFrame({"photo_id":new_info.values[my_members, 0], "longitude": new_info.values[my_members, 1], "latitude":new_info.values[my_members, 2]})

#     plt.plot(coords[my_members, 0], coords[my_members, 1], col + ".")
#     plt.plot(
#         cluster_center[0],
#         cluster_center[1],
#         "o",
#         markerfacecolor=col,
#         markeredgecolor="k",
#         markersize=14,
#     )
#     custom_info.to_csv(f'cluster_{i}.csv')

# plt.title("Estimated number of clusters: %d" % n_clusters_)
# plt.show()
# plt.savefig('cl')

# --------------------------------------------------------------------- #
# <5> Filter tag -> check whether textual context can be used or not
# --------------------------------------------------------------------- #

# tag = pd.read_csv('Tag.csv', names=['photo_id', '?', 'tag'])
# filtered = pd.read_csv('filtered_photos.csv')


# filtered_tag = pd.DataFrame()
# given_photo_ids = filtered.values[:, 1]
# pass_count = 0
# for id_ in tqdm(given_photo_ids):
#     id_ = int(id_)
#     try:
#         row_idx = list(tag.values[:, 0]).index(id_)
#         row = tag.loc[row_idx]

#         filtered_tag = filtered_tag.append(row)
#     except:
#         pass_count += 1
#         pass

# filtered_tag.to_csv('tag_filtered.csv')

filtered_tag = pd.read_csv('tag_filtered.csv')

check_filtered_tag = set(filtered_tag['tag'].values)
print(check_filtered_tag)

with open('filterd_tag_list.txt', 'w') as f:
    for tag in check_filtered_tag:
        f.writelines(tag + '\n')
    
f.close()






