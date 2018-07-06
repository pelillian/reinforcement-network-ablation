import numpy as np
import pandas as pd

actor_df = pd.read_csv('actor_log.txt', sep=',')

actor_df.sort_values(['Image', 'Trial', 'Wire', 'NetType', 'Run', 'RunSteps'], inplace=True)

# remove wires that failed to train & were marked with 'fail'
actor_df = actor_df[~actor_df['Wire'].str.contains("fail")]
# only look at actor data
actor_df = actor_df[actor_df['NetType'].str.contains("actor")]

for image in np.sort(actor_df['Image'].unique()):

	# this array is used to compute the average action of all the trained models
	action_arr = []

	for trial in np.sort(actor_df['Trial'].unique()):
		trial_df = actor_df[(actor_df['Image'] == image) & (actor_df['Trial'] == trial)]
		trial_df = trial_df[trial_df['Wire'] == trial_df['Wire'].max()]

		trained_result = trial_df.loc[trial_df['Run'].idxmax()]
		# print(trained_result)
		actions = [trained_result.Forward, trained_result.Left, trained_result.LeftRotation]
		print('Image', image, 'Trial', trial, 'Wire', trained_result.Wire, actions)

		action_arr.append(actions)

	avg_action = np.mean(action_arr, axis=0)



	print('Image', image, 'Average', avg_action)

