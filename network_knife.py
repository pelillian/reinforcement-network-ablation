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

	import nnFactory as Factory

# silence tf compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ACTIONS = 3
STATE_DIMENSIONS = (40, 40, 3)

NUM_SLICES = 10

PLOT_WIDTH = 0.25
FILENAME_PATTERN = 'actor_trial_'

IMAGE_TYPES =  ['left', 'right', 'forward']

num_images = 0
# populate image lists so we can test these images later
# image lists must all have the same number of images
image_type_list = []
image_file_list = []
image_dict = {}
for image_type in IMAGE_TYPES:
	image_dict[image_type] = []
	num_images = 0
	for image_filename in glob.iglob('camera/' + image_type + '/*.png'):
			image_type_list.append(image_type)
			image_file_list.append(image_filename)
			image_dict[image_type].append(image_filename)
			num_images += 1

# given a model, actually compute the actions for each image
def test_network(model, imagefiles_list):
	action_list = []
	for image_filename in imagefiles_list:

		img = imread(image_filename, as_gray=False, pilmode="RGB")
		img = img / np.max(img)

		graph = tf.get_default_graph()

		with graph.as_default():
				model._make_predict_function()
				K.set_learning_phase(0)

				action = np.reshape(model.predict(np.expand_dims(img, axis=0)), ACTIONS)

		if np.any(np.isnan(action)):
				raise ValueError("Net is broken!")

		# Actions:
		# Forward, Left, Left Rotation
		# x0.03, x0.03, x90*

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

def plot_actions(pdf, actions, action_max, title, colors=None, zero_index=True):
	font = {'family' : 'League Spartan',
		'weight' : 'bold'}
	plt.rc('font', **font)

	ind = np.arange(len(actions))
	if not zero_index:
		actions = np.delete(actions, (0), axis=0)
		if colors is not None:
			colors = np.delete(colors, (0), axis=0)
		ind = np.arange(len(actions)) + 1


	if colors is None:
		color_0 = ['#ea4626'] * len(actions)
		color_1 = ['#0060ff'] * len(actions)
	else:
		color_0 = num_to_color( np.abs(colors[:, 0] / action_max) )
		color_1 = num_to_color( np.abs(colors[:, 1] / action_max) )

	ax = plt.subplot(111)

	# use subplots2grid to add the images

	# plt.figure().imshow(plt.imread(image_filename), extent=[-0.5, 0.5, np.max(old_action) + PLOT_WIDTH, np.max(old_action) + 4*PLOT_WIDTH])

	ax.grid(color='#bbbbbb', linestyle='solid', linewidth=1)
	ax.set_axisbelow(True)

	ax.bar(ind - (PLOT_WIDTH / 2), actions[:, 0], width=PLOT_WIDTH, align='center', color=color_0)
	ax.bar(ind + (PLOT_WIDTH / 2), actions[:, 1], width=PLOT_WIDTH, align='center', color=color_1)

	ax.set_xlabel('Ablation Group')
	ax.set_ylabel('Output Change (Action Space)')
	ax.set_xticks(range(1, 11))

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
	orig_actions = test_network(model, imagefiles_list)

	# for each group of neurons, create a mask dropping out that group
	mask_list = get_masks(original_weights)

	# calculate the actions for each mask
	# create a list with an inner list for each image
	new_action_array = [] # actions for each mask after it's applied
	for image_idx in range(len(orig_actions)):
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
				new_action_array[image_idx + imgtype_num * num_images].append(new.tolist())
				differences.append(new.tolist() - orig_actions[image_idx])

			# remove the 'forward' action, just focusing on the turns
			differences = np.array(differences)[:, 1:]
			# calculate the mean across all the images (in the group)
			# trial_results[mask_num][imgtype_num] = np.mean(differences, axis=0).tolist()
			trial_results[mask_num][imgtype_num] = np.array(new_actions)[:, 1:].tolist()
			all_differences.append(differences)

	action_max = np.max(np.abs(np.array(all_differences)))

	# visualize the new actions compared to the old
	with PdfPages(re.sub('\..+', '.pdf', network_file_path)) as pdf:
		for image_idx, image_filename in enumerate(image_file_list):

			old_action = orig_actions[image_idx]
			new_actions = new_action_array[image_idx]

			# calculate the norm between each new action and the old action for coloring
			differences = [[0, 0, 0]]
			for new in new_actions:
				differences.append(new - old_action)

			actions = np.delete(np.array([old_action] + new_actions), 0, axis=1)
			differences =  np.delete(np.array(differences), 0, axis=1)

			plot_actions(pdf, actions, action_max, "Image " + str(image_idx + 1) + " (" + imagetypes_list[image_idx] + ")", colors=differences)
			plot_actions(pdf, differences, action_max, "Image " + str(image_idx + 1), zero_index=False)

	# for the entire trial, we've calculated the differences between the unmodified network and each network with different parts cut out.
	return (np.array(trial_results), np.array(orig_actions))

def colors(labels):
	colors = []
	colormap = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
	for i in range(len(labels)):
		colors.append(colormap[(int(labels[i])) % 10])
	return colors

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

def plot_2d(x_list, c_list, t_list):
	if len(x_list) != len(c_list) or len(c_list) != len(t_list):
		print('Length of x, color, and title lists must be the same')

	if len(x_list) > 1:
		fig = plt.figure(figsize=plt.figaspect(1 / len(x_list)))
	else:
		fig = plt.figure(figsize=(1024/128, 1024/128), dpi=128)

	for i, (x, colors, title) in enumerate(zip(x_list, c_list, t_list)):
		if len(x[0]) > 2: # use pca to remove extra dims
			pca = PCA(n_components=2)
			x = pca.fit_transform(x)
			use_pca = True

		ax = fig.add_subplot(1, len(x_list), i+1)
		ax.scatter(x[:, 0], x[:, 1], c=colors)
		if use_pca:
			ax.set_xlabel('First Principal Component')
			ax.set_ylabel('Second Principal Component')
		ax.set_title(title)

	plt.savefig('graphs/network_knife_ours.png', dpi=128)

def plot_3d(x_list, c_list, t_list, use_pca):
	if len(x_list) != len(c_list) or len(c_list) != len(t_list):
		print('Length of x, color, and title lists must be the same')

	if len(x_list) > 1:
		fig = plt.figure(figsize=plt.figaspect(1 / len(x_list)))
	else:
		fig = plt.figure(figsize=(1024, 1024))

	for i, (x, colors, title) in enumerate(zip(x_list, c_list, t_list)):
		ax = fig.add_subplot(1, len(x_list), i+1, projection='3d')
		ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=colors)
		if not use_pca:
			ax.set_xlabel(IMAGE_TYPES[0])
			ax.set_ylabel(IMAGE_TYPES[1])
			ax.set_zlabel(IMAGE_TYPES[2])
		ax.set_title(title)

	plt.show()

def plot_results(x, cluster_labels, trial_labels, use_pca=True):
	font = {'family' : 'League Spartan',
		'weight' : 'bold',
		'size' : '28'}
	plt.rc('font', **font)
	from matplotlib import rcParams
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()

	cluster_labels = colors(cluster_labels)
	trial_labels = colors(trial_labels)

	if use_pca:
		pca = PCA(n_components=3)
		x = pca.fit_transform(x)
		# print(pca.explained_variance_ratio_.cumsum())
		# print(pca.components_)
	else:
		x = np.mean(x.reshape(len(trial_labels), 6, num_images), axis=2)

	if use_pca:
		x_list = [x, x]
		c_list = [cluster_labels, trial_labels]
		t_list = ['cluster labels', 'trial labels']
	else:
		plot_3_2d(x, cluster_labels)
		plot_3_2d(x, trial_labels)

		x_list = [x[:, :3], x[:, :3], x[:, 3:], x[:, 3:]]
		c_list = [cluster_labels, trial_labels, cluster_labels, trial_labels]
		t_list = ['cluster labels, x 0-2', 'trial labels, x 0-2', 'cluster labels, x 3-6', 'trial labels, x 3-6']
	plot_3d(x_list, c_list, t_list, use_pca)
	plot_2d([x], [cluster_labels], [''])


def match_data(trial_results_arr, orig_actions_arr, trials):
	print(np.array(trial_results_arr).shape)
	NUM_TRIALS = len(trials)
	N_CLUSTERS = 4

	trial_labels = np.zeros((NUM_TRIALS, NUM_SLICES))
	for i in range(NUM_TRIALS):
		trial_labels[i, :] = i
	trial_labels = trial_labels.reshape(NUM_TRIALS * NUM_SLICES,)

	trial_results_arr = np.array(trial_results_arr).reshape(NUM_TRIALS * NUM_SLICES, 6 * num_images)
	trial_results_scaled = np.copy(trial_results_arr)

	for trial_num in range(NUM_TRIALS):
		trial_results_scaled[trial_labels == trial_num] -= np.mean(trial_results_scaled[trial_labels == trial_num], axis=0)
		trial_results_scaled[trial_labels == trial_num] /= np.max(np.abs(trial_results_scaled[trial_labels == trial_num]), axis=0)
		# trial_results_scaled[trial_labels == trial_num] = StandardScaler().fit(trial_results_scaled[trial_labels == trial_num]).transform(trial_results_scaled[trial_labels == trial_num])
	
	kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=23588).fit(trial_results_scaled)
	cluster_labels = kmeans.labels_
	# print(np.column_stack((trial_labels, cluster_labels)))

	plot_results(trial_results_scaled, cluster_labels, trial_labels, use_pca=True)

	# calculate stats for each cluster
	print(trial_results_scaled.shape)
	for i in range(N_CLUSTERS):
		cluster_data = trial_results_scaled[
			(cluster_labels == i)
		 	# & (trial_labels  == 0)
		 ]
		if len(cluster_data) > 0:
			avg = np.mean(cluster_data, axis=0).reshape(3, num_images, 2)
			avg = np.mean(avg, axis=1)
			print('cluster', i)
			print(avg)
		else:
			print('cluster', i, 'empty')

if __name__ == "__main__":

	trial_results_arr = []
	orig_actions_arr = []
	trials = []

	if len(sys.argv) < 2 or '-s' not in sys.argv:

		K.set_learning_phase(0)
		actor_model = Factory.actor_network(STATE_DIMENSIONS)

		for network_file_path in glob.iglob('test_networks/' + FILENAME_PATTERN + '*.h5'):
			# extract the trial number from the filename
			trial_num = int(re.search(r'\d+', re.sub('.*(\\\/)', '', network_file_path)).group(0))

			# test the network!
			trial_results, orig_actions = test_network_knife(actor_model, network_file_path, image_file_list, image_type_list)

			trial_results_arr.append(trial_results)
			orig_actions_arr.append(orig_actions)

			trials.append(trial_num)

		with open("trial_results.p", "wb") as tr:
			pkl.dump((trial_results_arr, orig_actions_arr, trials), tr, protocol=pkl.HIGHEST_PROTOCOL)

		K.clear_session()

	else:
		with open("trial_results.p", "rb") as tr:
			trial_results_arr, orig_actions_arr, trials = pkl.load(tr)

	match_data(trial_results_arr, orig_actions_arr, trials)
