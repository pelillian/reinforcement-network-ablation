import os
import re
import glob

from imageio import imread
from matplotlib.pyplot import imshow

import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras import backend as K

import nnFactory as Factory

# silence tf compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ACTIONS = 3
STATE_DIMENSIONS = (40, 40, 3)
ACTOR_LOG_FILE = 'actor_log.txt'

try:
    os.remove(ACTOR_LOG_FILE)
except OSError:
    pass
actor_log = open(ACTOR_LOG_FILE, "a+")
print('Trial', 'Wire', 'NetType', 'Run', 'RunSteps', 'NetworkFilename', 'Image', 'Forward', 'Left', 'LeftRotation', file=actor_log, sep=',')

K.set_learning_phase(0)
actor_model = Factory.actor_network(STATE_DIMENSIONS)

for run_folder_path in glob.iglob('networks/*'):
	run_folder_path = run_folder_path.replace('\\', '/')

	for wire_folder_path in glob.iglob(run_folder_path + '/*'):
		wire_folder_path = wire_folder_path.replace('\\', '/')

		for network_file_path in glob.iglob(wire_folder_path + '/actor*.h5'):
			network_file_path = network_file_path.replace('\\', '/')

			if 'goal' in network_file_path:
				continue

			_, trial_num, wire_num, network_filename = network_file_path.split('/')
			trial_num = int(re.search(r'\d+', trial_num).group(0))
			nettype, run_num, run_steps = network_filename.split('_')
			run_steps = run_steps.replace('.h5', '')

			actor_model.load_weights(network_file_path)

			for image_filename in glob.iglob('camera/*.png'):


				img = imread(image_filename, as_gray=False, pilmode="RGB")
				img = img / np.max(img)

				# size = (40, 40, 3)
				# img = np.array(img.getdata(), np.uint8).reshape(frame.size[1], frame.size[0], 3)
				# img = (imresize(img, size) / 127.5 - 1)

				graph = tf.get_default_graph()

				with graph.as_default():
				    # print(np.expand_dims(img, axis=0))
				    actor_model._make_predict_function()
				    K.set_learning_phase(0)

				    #print(self.actor.predict(np.expand_dims(img,axis=0)).shape)
				    action = np.reshape(actor_model.predict(np.expand_dims(img, axis=0)), ACTIONS)

				if np.any(np.isnan(action)):
				    raise ValueError("Net is broken!")

				# Actions:
				# Forward, Left, Left Rotation
				# x0.03, x0.03, x90*

				image_filename = image_filename.replace('\\', '/')
				image_filename = image_filename.replace('camera/img', '')
				image_filename = image_filename.replace('.png', '')

				print(trial_num, wire_num, nettype, run_num, run_steps,
					network_filename, image_filename, action[0], action[1], action[2], file=actor_log, sep=',')