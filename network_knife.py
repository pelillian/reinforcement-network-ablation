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

image_type_list = []
image_file_list = []
for image_type in ['left', 'right', 'forward']:
  for image_filename in glob.iglob('camera/' + image_type + '/*.png'):
      image_type_list.append(image_type)
      image_file_list.append(image_filename)


def test_network(model):
  action_list = []
  for image_filename in image_file_list:

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


def test_network_knife(model, network_file_path):

  model.load_weights(network_file_path)

  original_weights = np.copy(model.get_weights())
  # np.save('weights.npy', original_weights)

  old_action_array = test_network(model)
  # print('Original Actions:', old_action_array)

  # calculate the difference when the entire layer is zero
  mask0 = []
  for layer in original_weights:
    if layer.shape == (200, 100):
      layer = np.zeros_like(layer)
      mask0.append(layer)
    elif layer.shape == (100,):
      layer = np.zeros_like(layer)
      mask0.append(layer)
    else:
      mask0.append(np.ones_like(layer))

  new_weights = np.copy(original_weights)
  new_weights *= mask0

  model.set_weights(new_weights)
  new_actions = test_network(model)

  # for each group of neurons, create a mask dropping out that group
  mask_list = get_masks(original_weights)

  # create a list with an inner list for each image
  new_action_array = [] # actions after mask is applied
  for image_idx in range(len(old_action_array)):
    new_action_array.append([])

  all_differences = []
  # calculate each new action, then save in the arrays
  for i, mask in enumerate(mask_list):
    new_weights = np.copy(original_weights)
    new_weights *= mask

    model.set_weights(new_weights)
    new_actions = test_network(model)

    differences = []
    for image_idx, new in enumerate(new_actions):
      new_action_array[image_idx].append(new.tolist())
      differences.append(new.tolist() - old_action_array[image_idx])
    all_differences.append(differences)

  action_max = np.max(np.abs(np.array(all_differences)))
  trial_avg_differences = np.mean(all_differences, axis=1)


  # visualize the new actions compared to the old
  with PdfPages(re.sub('\..+', '.pdf', network_file_path)) as pdf:

    plot_actions(pdf, trial_avg_differences, trial_avg_differences, action_max, "Average Differences", zero_index=False)

    for image_idx, image_filename in enumerate(image_file_list):

      old_action = old_action_array[image_idx]
      new_actions = new_action_array[image_idx]

      # calculate the norm between each new action and the old action for coloring
      differences = [[0, 0, 0]]
      for new in new_actions:
        differences.append(new - old_action)

      actions = np.array([old_action] + new_actions)
      colors = np.array(differences)

      plot_actions(pdf, actions, colors, action_max, "Image " + str(image_idx + 1))
      plot_actions(pdf, colors, colors, action_max, "Image " + str(image_idx + 1) + " Differences")

  # for the entire trial, we've calculated the differences between the unmodified network and the network with different parts cut out.
  # we return the differences averaged over all of the input images
  return trial_avg_differences
    
def match_data(trials, avg_differences):
  # match up each group of neurons with a similar function
  both_differences = []
  trial_diff_sums = []
  pca_outputs = []
  with PdfPages("avg_differences.pdf") as pdf:
    for trial_num, trial_avg_differences in zip(trials, avg_differences):
      # remove first column
      trial_avg_differences = np.delete(trial_avg_differences, 0, axis=1)

      # plot diff sum
      trial_diff_sum = np.sum(trial_avg_differences, axis=1)

      ax = plt.subplot(111)
      ax.bar(np.arange(len(trial_diff_sum)) + 1, trial_diff_sum, PLOT_WIDTH)

      plt.title(FILENAME_PATTERN + str(trial_num) + ' difference sum')
      pdf.savefig()
      plt.clf()

      # plot pca
      pca = PCA(n_components=1)
      output = pca.fit_transform(trial_avg_differences)

      # make sure pca has the same sign
      for diff, p in zip(trial_diff_sum, output):
        if (diff > 0 and p < 0 and diff - p > np.max(trial_diff_sum)) or (diff < 0 and p > 0 and p - diff > np.max(trial_diff_sum)):
          output *= -1
          continue


      ax = plt.subplot(111)
      ax.bar(np.arange(len(output)) + 1, output, PLOT_WIDTH)

      plt.title(FILENAME_PATTERN + str(trial_num) + ' PCA')
      pdf.savefig()
      plt.clf()

      # sort by pca for later plotting
      data = np.column_stack((trial_diff_sum, output))
      data = data[data[:,1].argsort()]
      trial_diff_sum = data[:,0]
      output = data[:,1]

      both_differences.append(trial_avg_differences[:, 0])
      both_differences.append(trial_avg_differences[:, 1])
      trial_diff_sums.append(trial_diff_sum)
      pca_outputs.append(output)

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
    avg_differences.append(test_network_knife(actor_model, network_file_path))
    trials.append(trial_num)

  np.save('avg_differences.npy', avg_differences)

  match_data(trials, avg_differences)

  K.clear_session()