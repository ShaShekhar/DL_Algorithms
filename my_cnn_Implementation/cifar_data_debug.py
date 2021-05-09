import pickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(files):
    with open(files,'rb') as f:
        dicts = pickle.load(f,encoding='latin1')
    return dicts
# Create label dictionary
label_dict = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
#data = unpickle('cifar-10-python/cifar-10-batches-py/data_batch_1').get(b'data')
#data type uint8 i.e 1 byte or 8 bit i.e. all bits are on so we have total 255*255*255 = 16581375 approx. 16.5 Million colors.
#print(data.shape)  #shape = (10000,3072)
def load_data_wrapper():
    data = unpickle('cifar-10-python/cifar-10-batches-py/data_batch_2').get('data')
    label = unpickle('cifar-10-python/cifar-10-batches-py/data_batch_2').get('labels')
    training_data = [np.transpose(np.reshape(x,(3,32,32)),(1,2,0)) for i,x in enumerate(data)]
    training_label = [label_dict[x] for x in label]
    for i in range(100):
        ax = plt.subplot(10,10,i+1)
        ax.set_title(training_label[100+i])
        plt.imshow(training_data[100+i])
        plt.xticks([]),plt.yticks([])
    plt.show()
load_data_wrapper()

#data = unpickle('cifar-100-python/train')#.get('data')
#print(set(data['coarse_labels']).intersection(set(data['fine_labels'])))
#dict_keys(['batch_label', 'fine_labels', 'filenames', 'coarse_labels', 'data'])
dict_coarse_label_names = {0:'aquatic_mammals', 1:'fish', 2:'flowers', 3:'food_containers', 4:'fruit_and_vegetables',
                           5:'household_electrical_devices',6:'household_furniture', 7:'insects', 8:'large_carnivores',
                           9:'large_man-made_outdoor_things', 10:'large_natural_outdoor_scenes', 11:'large_omnivores_and_herbivores',
                           12:'medium_mammals', 13:'non-insect_invertebrates', 14:'people', 15:'reptiles', 16:'small_mammals',
                           17:'trees', 18:'vehicles_1', 19:'vehicles_2'}

dict_fine_label_names = {0:'apple',1:'aquarium_fish',2:'baby',3:'bear', 4:'beaver', 5:'bed', 6:'bee', 7:'beetle',8:'bicycle', 9:'bottle',
                        10:'bowl', 11:'boy', 12:'bridge', 13:'bus',14:'butterfly', 15:'camel', 16:'can', 17:'castle', 18:'caterpillar',
                        19:'cattle', 20:'chair', 21:'chimpanzee', 22:'clock', 23:'cloud', 24:'cockroach', 25:'couch', 26:'crab', 27:'crocodile',
                        28:'cup', 29:'dinosaur', 30:'dolphin', 31:'elephant', 32:'flatfish', 33:'forest', 34:'fox', 35:'girl', 36:'hamster',
                        37:'house',38:'kangaroo', 39:'keyboard', 40:'lamp', 41:'lawn_mower',42:'leopard', 43:'lion', 44:'lizard', 45:'lobster',
                        46:'man', 47:'maple_tree',48:'motorcycle', 49:'mountain', 50:'mouse', 51:'mushroom', 52:'oak_tree', 53:'orange',
                        54:'orchid', 55:'otter',56:'palm_tree', 57:'pear', 58:'pickup_truck', 59:'pine_tree', 60:'plain', 61:'plate', 62:'poppy',
                        63:'porcupine',64:'possum', 65:'rabbit', 66:'raccoon', 67:'ray', 68:'road', 69:'rocket',70:'rose', 71:'sea', 72:'seal',
                        73:'shark', 74:'shrew',75:'skunk',76:'skyscraper', 77:'snail', 78:'snake', 79:'spider', 80:'squirrel', 81:'streetcar',
                        82:'sunflower', 83:'sweet_pepper',84:'table', 85:'tank', 86:'telephone', 87:'television', 88:'tiger', 89:'tractor',
                        90:'train', 91:'trout', 92:'tulip',93:'turtle', 94:'wardrobe', 95:'whale', 96:'willow_tree', 97:'wolf',98:'woman', 99:'worm'}

def load_data_wrapper():
    data = unpickle('cifar-100-python/train').get('data')
    training_data = [np.transpose(np.reshape(x,(3,32,32)),(1,2,0)) for i,x in enumerate(data)]
    fine_label = unpickle('cifar-100-python/train').get('fine_labels')
    #coarse_label = unpickle('cifar-100-python/train').get('coarse_labels')
    training_label = [dict_fine_label_names[x] for x in fine_label]
    for i in range(81):
        ax = plt.subplot(9,9,i+1)
        ax.set_title(training_label[200+i])
        plt.imshow(training_data[200+i])
        plt.xticks([]),plt.yticks([])
    plt.show()
#load_data_wrapper()


#Superclass(coarse_label)	               Classes(fine_label)
#aquatic mammals 	                beaver, dolphin, otter, seal, whale
#fish 	                            aquarium fish, flatfish, ray, shark, trout
#flowers 	                        orchids, poppies, roses, sunflowers, tulips
#food containers 	                bottles, bowls, cans, cups, plates
#fruit and vegetables 	            apples, mushrooms, oranges, pears, sweet peppers
#household electrical devices 	    clock, computer keyboard, lamp, telephone, television
#household furniture 	            bed, chair, couch, table, wardrobe
#insects 	                        bee, beetle, butterfly, caterpillar, cockroach
#large carnivores 	                bear, leopard, lion, tiger, wolf
#large man-made outdoor things 	    bridge, castle, house, road, skyscraper
#large natural outdoor scenes 	    cloud, forest, mountain, plain, sea
#large omnivores and herbivores 	camel, cattle, chimpanzee, elephant, kangaroo
#medium-sized mammals 	            fox, porcupine, possum, raccoon, skunk
#non-insect invertebrates 	        crab, lobster, snail, spider, worm
#people 	                        baby, boy, girl, man, woman
#reptiles 	                        crocodile, dinosaur, lizard, snake, turtle
#small mammals 	                    hamster, mouse, rabbit, shrew, squirrel
#trees 	                            maple, oak, palm, pine, willow
#vehicles 1 	                    bicycle, bus, motorcycle, pickup truck, train
#vehicles 2 	                    lawn-mower, rocket, streetcar, tank, tractor
