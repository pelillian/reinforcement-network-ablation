# This file is taken from RWTH Aachen Cybernetics Lab CENSE Project

from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Conv2D, Dropout, Flatten, Reshape, MaxPooling2D, Lambda, \
    RepeatVector
from keras.initializers import RandomUniform


def actor_network(input_shape):
    state_input = Input(shape=input_shape)
    # conv_module = Reshape(input_shape + (1,))(state_input)
    conv_module = Conv2D(30, (5, 5), activation="relu", name='conv_1')(state_input)  # 36x36x30
    conv_module = MaxPooling2D(pool_size=(2, 2))(conv_module)  # 18x18x30
    # conv_module = Dropout(.5)(conv_module)
    conv_module = Conv2D(15, (5, 5), activation="relu", name='conv_2')(conv_module)  # 14x14x30
    conv_module = MaxPooling2D(pool_size=(2, 2))(conv_module)  # 7x7x15
    # conv_module = Dropout(.5)(conv_module)
    conv_module = Conv2D(10, (3, 3), activation="relu", name='conv_3')(conv_module)  # 5x5x10
    # conv_module = MaxPooling2D(pool_size=(2, 2))(conv_module)  # 7x7x15
    # conv_module = Dropout(.5)(conv_module)
    conv_module = Flatten()(conv_module)
    conv_module = Dropout(.2)(conv_module)

    mlp_module = Dense(400, activation='relu', name='dense_1')(conv_module)
    mlp_module = Dropout(.2)(mlp_module)
    mlp_module = Dense(200, activation='relu', name='dense_2')(mlp_module)
    mlp_module = Dropout(.2)(mlp_module)
    mlp_module = Dense(100, activation='relu', name='dense_3')(mlp_module)
    mlp_module = Dropout(.2)(mlp_module)

    forward_action = Dense(1, activation='sigmoid', name='forward')(mlp_module)
    sideways_action = Dense(1, activation='tanh', name='sideways')(mlp_module)
    rotation_action = Dense(1, activation='tanh', name='rotation')(mlp_module)

    action_output = Concatenate()([forward_action, sideways_action, rotation_action])

    model = Model(inputs=[state_input], outputs=[action_output])

    model.compile(loss='mse',
                  optimizer='adam')

    return model


def critic_network(input_shape):
    state_input = Input(shape=input_shape)
    action_input = Input(shape=(3,))

    # conv_module = Reshape(input_shape + (1,))(state_input)
    conv_module = Conv2D(30, (5, 5), activation="relu", name='conv_1')(state_input)  # 36x36x30
    conv_module = MaxPooling2D(pool_size=(2, 2))(conv_module)  # 18x18x30
    # conv_module = Dropout(.2)(conv_module)
    conv_module = Conv2D(15, (5, 5), activation="relu", name='conv_2')(conv_module)  # 14x14x30
    conv_module = MaxPooling2D(pool_size=(2, 2))(conv_module)  # 7x7x15
    # conv_module = Dropout(.2)(conv_module)
    conv_module = Conv2D(10, (3, 3), activation="relu", name='conv_3')(conv_module)  # 5x5x10
    conv_module = Flatten()(conv_module)
    conv_module = Dropout(.2)(conv_module)

    state_module = Dense(200, activation='relu', name='state_1')(conv_module)

    action_module = Dense(200, activation='relu', name='action_1')(action_input)

    mlp_module = Concatenate()([state_module, action_module])
    mlp_module = Dropout(.2)(mlp_module)
    mlp_module = Dense(200, activation='relu', name='mlp_1')(mlp_module)
    mlp_module = Dropout(.2)(mlp_module)
    mlp_module = Dense(100, activation='relu', name='mlp_2')(mlp_module)
    mlp_module = Dropout(.2)(mlp_module)

    q_value_output = Dense(1, activation="linear", name='critic_q_value',
                           kernel_initializer=RandomUniform(-.0003, .0003))(mlp_module)
    # q_value_output = Lambda(lambda x: 2*x, name='scaled_q_value')(q_value_output)

    model = Model(inputs=[state_input, action_input], outputs=[q_value_output])

    model.compile(loss='mse', optimizer='adam')

    return model