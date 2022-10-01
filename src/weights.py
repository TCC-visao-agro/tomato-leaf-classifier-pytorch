from sklearn.utils import class_weight
import numpy as np
class_names = [
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]
class_samples = [
    {'Tomato_Bacterial_spot': 2127}, {'Tomato_Early_blight': 1000}, {'Tomato_Late_blight': 1909},
    {'Tomato_Leaf_Mold': 952}, {'Tomato_Septoria_leaf_spot': 1771}, {'Tomato_Spider_mites_Two_spotted_spider_mite': 1676},
    {'Tomato__Target_Spot': 1404}, {'Tomato__Tomato_YellowLeaf__Curl_Virus': 3208}, {'Tomato__Tomato_mosaic_virus': 373}, {'Tomato_healthy': 1591}
]

class_samples_number = [2127,  1000,  1909, 952,  1771,  1676,
     1404,  3208,  373,  1591
]

class_sample_count = np.unique(class_samples_number, return_counts=True)[1]
weight = 1. / class_samples_number
samples_weight = weight[class_samples_number]
print(samples_weight)
# samples_weight = weight[class_samples_number]
# samples_weight = torch.from_numpy(samples_weight)