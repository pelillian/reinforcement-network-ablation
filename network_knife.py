import os
import glob
import re

from imageio import imread
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas as pd
from numpy import linalg
from sklearn.decomposition import PCA

import tensorflow as tf
from keras.models import load_model
from keras import backend as K

import nnFactory as Factory

# silence tf compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ACTIONS = 3
STATE_DIMENSIONS = (40, 40, 3)

PLOT_WIDTH = 0.25
FILENAME_PATTERN = 'actor_trial_'

# populate image lists so we can test these images later
image_type_list = []
image_file_list = []
# image_dict = {}
for image_type in ['left', 'right', 'forward']:
	# image_dict[image_type] = []
	for image_filename in glob.iglob('camera/' + image_type + '/*.png'):
			image_type_list.append(image_type)
			image_file_list.append(image_filename)
			# image_dict[image_type].append(image_filename)

# given a model, actually compute the actions for each image
def test_network(model, imagefiles):
	action_list = []
	for image_filename in imagefiles:

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
	for n in range(10):
		mask_ = []
		for layer in original_weights:
			if layer.shape == (200, 100):
				# set weights to zero
				layer = np.ones_like(layer)
				n_ = n*10
				layer[:, n_:(n_ + 10)] = 0
				mask_.append(layer)
			elif layer.shape == (100,):
				# set biases to zero
				layer = np.ones_like(layer)
				n_ = n*10
				layer[n_:(n_ + 10)] = 0
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

def test_network_knife(model, network_file_path, imagefiles):

	model.load_weights(network_file_path)

	original_weights = np.copy(model.get_weights())

	# test the network on the unmodified model as a baseline for comparison
	old_action_array = test_network(model, imagefiles)

	# for each group of neurons, create a mask dropping out that group
	mask_list = get_masks(original_weights)

	# calculate the actions for each mask
	# create a list with an inner list for each image
	new_action_array = [] # actions for each mask after it's applied

	all_differences = []
	# calculate each new action, then save in the arrays
	for i, mask in enumerate(mask_list):
		new_weights = np.copy(original_weights)
		new_weights *= mask

		model.set_weights(new_weights)
		new_actions = test_network(model, imagefiles)

		if len(new_action_array) == 0:
			for image_idx in range(len(new_actions)):
				new_action_array.append([])

		differences = []
		for image_idx, new in enumerate(new_actions):
			new_action_array[image_idx].append(new.tolist())
			differences.append(new.tolist() - old_action_array[image_idx])
		all_differences.append(differences)

	trial_avg_differences = np.mean(all_differences, axis=1)
	action_max = np.max(np.abs(np.array(all_differences)))

	# for the entire trial, we've calculated the differences between the unmodified network and the network with different parts cut out.
	# we return the differences averaged over all of the input images
	return trial_avg_differences
		
def match_data(trials, avg_differences):
	# match up each group of neurons with a similar function

	with PdfPages("final_graph.pdf") as pdf:
		data = np.column_stack(tuple(both_differences))
		df = pd.DataFrame(data, index=np.arange(len( both_differences[0])) + 1)
		df.plot.bar()

		plt.title('Sorted by PCA')
		pdf.savefig()
		plt.clf()

		data = np.column_stack(tuple(trial_diff_sums))
		df = pd.DataFrame(data, index=np.arange(len(trial_diff_sums[0])) + 1)
		df.plot.bar()

		plt.title('Sorted by PCA')
		pdf.savefig()
		plt.clf()

if __name__ == "__main__":

	K.set_learning_phase(0)
	actor_model = Factory.actor_network(STATE_DIMENSIONS)

	avg_differences = []
	trials = []
	for network_file_path in glob.iglob('test_networks/' + FILENAME_PATTERN + '*.h5'):
		# extract the trial number from the filename
		trial_num = int(re.search(r'\d+', re.sub('.*(\\\/)', '', network_file_path)).group(0))

		# test the network!
		avg_differences.append(test_network_knife(actor_model, network_file_path, image_file_list))
		trials.append(trial_num)

	np.save('avg_differences.npy', avg_differences)

	match_data(trials, avg_differences)

	K.clear_session()