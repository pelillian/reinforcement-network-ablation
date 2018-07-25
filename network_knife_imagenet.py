import os
import sys
import glob
import re

import pickle as pkl

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
from scipy.spatial.distance import cdist

# from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

if len(sys.argv) < 2 or '-s' not in sys.argv:
	from imageio import imread

	import tensorflow as tf
	from keras.models import load_model
	from keras import backend as K

	from keras.applications.xception import Xception

# silence tf compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ACTIONS = 3

IMAGE_SIZE = (299, 299, 3)

NUM_SLICES = 64

PLOT_WIDTH = 0.25

IMAGE_TYPES =  ['maltese', 'power_drill', 'obelisk']
IMAGE_IDX = [153, 740, 682, 2]

num_images = 0
# populate image lists so we can test these images later
# image lists must all have the same number of images
image_type_list = []
image_file_list = []
image_dict = {}
for image_type in IMAGE_TYPES:
	image_dict[image_type] = []
	num_images = 0
	for image_filename in glob.iglob('images/' + image_type + '/*.png'):
			image_type_list.append(image_type)
			image_file_list.append(image_filename)
			image_dict[image_type].append(image_filename)
			num_images += 1

# given a model, actually compute the actions for each image
def test_network(model, imagefiles_list):
	img_arr = np.empty((len(imagefiles_list),  IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]), dtype=np.float64)
	for img_num, image_filename in enumerate(imagefiles_list):

		img = imread(image_filename, as_gray=False, pilmode="RGB")
		img = img / np.max(img)
		img_arr[img_num] = img

	pred = model.predict(img_arr)
	# only keep columns we're interested in
	pred = pred[:, IMAGE_IDX]

	return pred

def get_masks(original_weights):
	mask_list = []

	# for each group of neurons, create a mask dropping out that group
	SLICE_SIZE = 2048 // NUM_SLICES
	for n in range(NUM_SLICES):
		mask_ = []
		for layer in original_weights:
			if layer.shape == (1, 1, 1536, 2048):
				# set weights to zero
				layer = np.ones_like(layer)
				n_ = n*SLICE_SIZE
				layer[:, :, :, n_:(n_ + SLICE_SIZE)] = 0
				mask_.append(layer)
			elif layer.shape == (2048,):
				# set biases to zero
				layer = np.ones_like(layer)
				n_ = n*SLICE_SIZE
				layer[n_:(n_ + SLICE_SIZE)] = 0
				mask_.append(layer)
			else:
				mask_.append(np.ones_like(layer))
		mask_list.append(mask_)

	return mask_list

def num_to_color(arr):
	new_arr = []
	for item in arr.tolist():
		if item == 0:
			new_arr.append((0, 1, 0))
		else:
			new_arr.append((item, 0, 1 - item))
	return new_arr

def plot_actions(pdf, actions, colors, action_max, title, zero_index=True):
	ind = np.arange(len(actions))
	if not zero_index:
		ind += 1

	ax = plt.subplot(111)

	# use subplots2grid to add the images

	# plt.figure().imshow(plt.imread(image_filename), extent=[-0.5, 0.5, np.max(old_action) + PLOT_WIDTH, np.max(old_action) + 4*PLOT_WIDTH])

	color_0 = num_to_color( np.abs(colors[:, 0] / action_max) )
	ax.bar(ind - PLOT_WIDTH, actions[:, 0], width=PLOT_WIDTH, align='center', color=color_0)

	color_1 = num_to_color( np.abs(colors[:, 1] / action_max) )
	ax.bar(ind,               actions[:, 1], width=PLOT_WIDTH, align='center', color=color_1)

	color_2 = num_to_color( np.abs(colors[:, 2] / action_max) )
	ax.bar(ind + PLOT_WIDTH, actions[:, 2], width=PLOT_WIDTH, align='center', color=color_2)

	# ax2 = plt.figure().add_axes([0, 0.9, 0.1, 0.1])
	# ax2.imshow(plt.imread(image_filename))

	plt.title(title)
	pdf.savefig()
	plt.clf()

def test_network_knife(model, network_file_path, imagefiles_list, imagetypes_list):

	imagefiles_list = np.array(imagefiles_list)
	imagetypes_list = np.array(imagetypes_list)

	original_weights = np.copy(model.get_weights())

	# test the network on the unmodified model as a baseline for comparison
	original_pred = test_network(model, imagefiles_list)

	# for each group of neurons, create a mask dropping out that group
	mask_list = get_masks(original_weights)

	# calculate the actions for each mask
	# create a list with an inner list for each image
	new_pred_array = [] # actions for each mask after it's applied
	for image_idx in range(len(original_pred)):
		new_pred_array.append([])

	trial_results = []
	# calculate each new action, then save in the arrays
	for mask_num, mask in enumerate(mask_list):
		trial_results.append([])

		new_weights = np.copy(original_weights)
		new_weights *= mask

		model.set_weights(new_weights)

		for imgtype_num, image_type in enumerate(IMAGE_TYPES):
			local_imagefiles = imagefiles_list[imagetypes_list == image_type]
			trial_results[mask_num].append([])

			new_preds = test_network(model, local_imagefiles)

			differences = []
			for image_idx, new in enumerate(new_preds):
				new_pred_array[image_idx + imgtype_num * num_images].append(new.tolist())
				differences.append(new.tolist() - original_pred[image_idx])

			trial_results[mask_num][imgtype_num] = new_preds

	trial_results = np.array(trial_results)
	original_pred = np.array(original_pred)
	return (trial_results, original_pred.reshape(trial_results.shape[1:]))
		
def plot_3_2d(x, colors):
	fig = plt.figure(figsize=plt.figaspect(1/3))
	ax = fig.add_subplot(1, 3, 1)
	ax.scatter(x[:, 0], x[:, 1], c=colors)
	ax.set_xlabel('left')
	ax.set_ylabel('left_rotation')
	ax.set_title(IMAGE_TYPES[0])
	ax = fig.add_subplot(1, 3, 2)
	ax.scatter(x[:, 2], x[:, 3], c=colors)
	ax.set_xlabel('left')
	ax.set_ylabel('left_rotation')
	ax.set_title(IMAGE_TYPES[1])
	ax = fig.add_subplot(1, 3, 3)
	ax.scatter(x[:, 4], x[:, 5], c=colors)
	ax.set_xlabel('left')
	ax.set_ylabel('left_rotation')
	ax.set_title(IMAGE_TYPES[2])

	plt.show()

def plot_3d(x_list, c_list, t_list):
	if len(x_list) != len(c_list) or len(c_list) != len(t_list):
		print('Length of x, color, and title lists must be the same')

	fig = plt.figure(figsize=plt.figaspect(1 / len(x_list)))

	for i, (x, colors, title) in enumerate(zip(x_list, c_list, t_list)):
		ax = fig.add_subplot(1, len(x_list), i+1, projection='3d')
		ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=colors)

		ax.set_title(title)

	plt.show()

def plot_results(x, cluster_labels):
	pca = PCA(n_components=3)
	x = pca.fit_transform(x)
	# print(pca.explained_variance_ratio_.cumsum())
	# print(pca.components_)

	x_list = [x, x]
	c_list = [cluster_labels]
	t_list = ['cluster labels']
	plot_3d(x_list, c_list, t_list)


def match_data(trial_results, original_pred):
	print(np.array(trial_results).shape)

	NUM_CLUSTERS = 4

	trial_results = np.array(trial_results).reshape(NUM_SLICES, trial_results.size // NUM_SLICES)
	trial_results_scaled = np.copy(trial_results)

	trial_results_scaled -= np.mean(trial_results_scaled, axis=0)
	trial_results_scaled /= np.max(np.abs(trial_results_scaled), axis=0)
	# trial_results_scaled[trial_labels == trial_num] = StandardScaler().fit(trial_results_scaled[trial_labels == trial_num]).transform(trial_results_scaled[trial_labels == trial_num])
	
	kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=23589).fit(trial_results_scaled)
	cluster_labels = kmeans.labels_
	# print(np.column_stack((trial_labels, cluster_labels)))

	plot_results(trial_results_scaled, cluster_labels)

	# calculate stats for each cluster
	print(trial_results.shape)
	for i in range(NUM_CLUSTERS):
		cluster_data = trial_results[(cluster_labels == i)]
		cluster_data = cluster_data.reshape(cluster_data.shape[0], len(IMAGE_TYPES), num_images, len(IMAGE_IDX))
		if len(cluster_data) > 0:
			avg = np.mean(cluster_data, axis=2)
			avg = avg.reshape(avg.shape[0], len(IMAGE_TYPES) * len(IMAGE_IDX))[:, [0, 5, 10]]
			avg = np.mean(avg, axis=0)
			print('cluster', i, cluster_data.shape)
			print(np.round(avg, 2))
		else:
			print('cluster', i, 'empty')

	print('original_pred', original_pred.shape)
	avg_original_pred = np.mean(original_pred, axis=1)
	avg_original_pred = avg_original_pred.reshape(len(IMAGE_TYPES) * len(IMAGE_IDX))[[0, 5, 10]]
	print(avg_original_pred)

if __name__ == "__main__":

	trials = []

	if len(sys.argv) < 2 or '-s' not in sys.argv:

		K.set_learning_phase(0)
		model = Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

		# for network_file_path in glob.iglob('test_networks/' + FILENAME_PATTERN + '*.h5'):
		network_file_path = 'test_networks/imagenet.pdf'

		# test the network!
		trial_results, original_pred = test_network_knife(model, network_file_path, image_file_list, image_type_list)

		with open("trial_results_imagenet_64.p", "wb") as tr:
			pkl.dump((trial_results, original_pred), tr, protocol=pkl.HIGHEST_PROTOCOL)

		K.clear_session()

	else:
		with open("trial_results_imagenet_64.p", "rb") as tr:
			trial_results, original_pred = pkl.load(tr)

	match_data(trial_results, original_pred)
