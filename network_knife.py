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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

if len(sys.argv) < 2 or '-s' not in sys.argv:
	from imageio import imread

	import tensorflow as tf
	from keras.models import load_model
	from keras import backend as K

	import nnFactory as Factory

# silence tf compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ACTIONS = 3
STATE_DIMENSIONS = (40, 40, 3)

NUM_SLICES = 10

PLOT_WIDTH = 0.25
FILENAME_PATTERN = 'actor_trial_'

IMAGE_TYPES =  ['left', 'right', 'forward']

# populate image lists so we can test these images later
image_type_list = []
image_file_list = []
image_dict = {}
for image_type in IMAGE_TYPES:
	image_dict[image_type] = []
	for image_filename in glob.iglob('camera/' + image_type + '/*.png'):
			image_type_list.append(image_type)
			image_file_list.append(image_filename)
			image_dict[image_type].append(image_filename)

# given a model, actually compute the actions for each image
def test_network(model, imagefiles_list):
	action_list = []
	for image_filename in imagefiles_list:

		img = imread(image_filename, as_gray=False, pilmode="RGB")
		img = img / np.max(img)

		# size = (40, 40, 3)
		# img = np.array(img.getdata(), np.uint8).reshape(frame.size[1], frame.size[0], 3)
		# img = (imresize(img, size) / 127.5 - 1)

		graph = tf.get_default_graph()

		with graph.as_default():
				# print(np.expand_dims(img, axis=0))
				model._make_predict_function()
				K.set_learning_phase(0)

				#print(self.actor.predict(np.expand_dims(img,axis=0)).shape)
				action = np.reshape(model.predict(np.expand_dims(img, axis=0)), ACTIONS)

		if np.any(np.isnan(action)):
				raise ValueError("Net is broken!")

		# Actions:
		# Forward, Left, Left Rotation
		# x0.03, x0.03, x90*

		image_filename = image_filename.replace('\\', '/')
		image_filename = image_filename.replace('camera/img', '')
		image_filename = image_filename.replace('.png', '')

		action_list.append(action)

	return action_list

def get_masks(original_weights):
	mask_list = []

	# for each group of neurons, create a mask dropping out that group
	SLICE_SIZE = 100 // NUM_SLICES
	for n in range(NUM_SLICES):
		mask_ = []
		for layer in original_weights:
			if layer.shape == (200, 100):
				# set weights to zero
				layer = np.ones_like(layer)
				n_ = n*SLICE_SIZE
				layer[:, n_:(n_ + SLICE_SIZE)] = 0
				mask_.append(layer)
			elif layer.shape == (100,):
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

	model.load_weights(network_file_path)

	original_weights = np.copy(model.get_weights())

	# test the network on the unmodified model as a baseline for comparison
	old_action_array = test_network(model, imagefiles_list)

	# for each group of neurons, create a mask dropping out that group
	mask_list = get_masks(original_weights)

	# calculate the actions for each mask
	# create a list with an inner list for each image
	new_action_array = [] # actions for each mask after it's applied
	for image_idx in range(len(old_action_array)):
				new_action_array.append([])

	all_differences = []
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

			new_actions = test_network(model, local_imagefiles)

			differences = []
			for image_idx, new in enumerate(new_actions):
				new_action_array[image_idx + imgtype_num * 8].append(new.tolist())
				differences.append(new.tolist() - old_action_array[image_idx])

			# remove the 'forward' action, just focusing on the turns
			differences = np.array(differences)[:, 1:]
			# calculate the mean across all the images (in the group)
			# trial_results[mask_num][imgtype_num] = np.mean(differences, axis=0).tolist()
			trial_results[mask_num][imgtype_num] = differences.tolist()
			all_differences.append(differences)

	action_max = np.max(np.abs(np.array(all_differences)))

	# visualize the new actions compared to the old
	with PdfPages(re.sub('\..+', '.pdf', network_file_path)) as pdf:
		for image_idx, image_filename in enumerate(image_file_list):

			old_action = old_action_array[image_idx]
			new_actions = new_action_array[image_idx]

			# calculate the norm between each new action and the old action for coloring
			differences = [[0, 0, 0]]
			for new in new_actions:
				differences.append(new - old_action)

			actions = np.array([old_action] + new_actions)
			colors = np.array(differences)

			plot_actions(pdf, actions, colors, action_max, "Image " + str(image_idx + 1) + " (" + imagetypes_list[image_idx] + ")")
			plot_actions(pdf, colors, colors, action_max, "Image " + str(image_idx + 1) + " Differences")

	# for the entire trial, we've calculated the differences between the unmodified network and each network with different parts cut out.
	# trial_results is the differences averaged over all of the input images in each group
	return np.array(trial_results)
		
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
		ax.set_xlabel(IMAGE_TYPES[0])
		ax.set_ylabel(IMAGE_TYPES[1])
		ax.set_zlabel(IMAGE_TYPES[2])
		ax.set_title(title)

	plt.show()

def match_data(trials, trial_results_arr):
	print(np.array(trial_results_arr).shape)
	NUM_TRIALS = len(trials)
	num_features = 48

	trial_labels = np.zeros((NUM_TRIALS, NUM_SLICES))
	for i in range(NUM_TRIALS):
		trial_labels[i, :] = i
	trial_labels = trial_labels.reshape(NUM_TRIALS * NUM_SLICES,)

	trial_results_arr = np.array(trial_results_arr).reshape(NUM_TRIALS * NUM_SLICES, num_features)
	trial_results_scaled = np.copy(trial_results_arr)

	for trial_num in range(NUM_TRIALS):
		trial_results_scaled[trial_labels == trial_num] -= np.mean(trial_results_scaled[trial_labels == trial_num], axis=0)
		trial_results_scaled[trial_labels == trial_num] /= np.max(np.abs(trial_results_scaled[trial_labels == trial_num]), axis=0)
		# trial_results_scaled[trial_labels == trial_num] = StandardScaler().fit(trial_results_scaled[trial_labels == trial_num]).transform(trial_results_scaled[trial_labels == trial_num])
	
	# pca = PCA(n_components=3)
	x = np.mean(trial_results_arr.reshape(NUM_TRIALS * NUM_SLICES, 6, 8), axis=2)
	# x = pca.fit_transform(x)
	# print(pca.explained_variance_ratio_.cumsum())
	# print(pca.components_)

	kmeans = KMeans(n_clusters=4, random_state=38599).fit(trial_results_scaled)
	# print(np.column_stack((trial_labels, kmeans.labels_)))

	plot_3_2d(x, kmeans.labels_)
	plot_3_2d(x, trial_labels)

	x_list = [x[:, :3], x[:, :3], x[:, 3:], x[:, 3:]]
	c_list = [kmeans.labels_, trial_labels, kmeans.labels_, trial_labels]
	t_list = ['kmeans labels, x 0-2', 'trial labels, x 0-2', 'kmeans labels, x 3-6', 'trial labels, x 3-6']
	plot_3d(x_list, c_list, t_list)

	# calculate stats for each cluster
	print(trial_results_arr.shape)
	for i in range(4):
		cluster_data = trial_results_arr[kmeans.labels_ == i]
		avg = np.mean(cluster_data, axis=0).reshape(3, 8, 2)
		avg = np.mean(avg, axis=1)
		print('cluster', i)
		print(avg)

if __name__ == "__main__":

	trial_results_arr = []
	trials = []

	if len(sys.argv) < 2 or '-s' not in sys.argv:

		K.set_learning_phase(0)
		actor_model = Factory.actor_network(STATE_DIMENSIONS)

		for network_file_path in glob.iglob('test_networks/' + FILENAME_PATTERN + '*.h5'):
			# extract the trial number from the filename
			trial_num = int(re.search(r'\d+', re.sub('.*(\\\/)', '', network_file_path)).group(0))

			# test the network!
			trial_results = test_network_knife(actor_model, network_file_path, image_file_list, image_type_list)

			trial_results_arr.append(trial_results)

			trials.append(trial_num)

		with open("trial_results.p", "wb") as tr:
			pkl.dump((trial_results_arr, trials), tr, protocol=pkl.HIGHEST_PROTOCOL)

		K.clear_session()

	else:
		with open("trial_results.p", "rb") as tr:
			trial_results_arr, trials = pkl.load(tr)

	match_data(trials, trial_results_arr)
