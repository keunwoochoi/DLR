import tensorflow as tf
import numpy as np
import librosa

from config import SAVE_DIR
from models import dlr13

def get_DLR_1D(input_):
    '''
    Args:
        input_ - 1D numpy array
    Return:
        ouput - 2D numpy array DLR
    '''
    batch_size = 1
    length = input_.shape[0]
    # normalize 
    normalize = np.max(np.abs(input_))
    if normalize!=0:
        input_/=normalize
    # abs
    input_ = np.abs(input_)
    # build model
    tf.reset_default_graph()
    y = tf.placeholder(tf.float32, shape=[batch_size, length])
    with tf.variable_scope("DLR"):
        dlr = dlr13(y)
    with tf.Session() as sess:
        # restore model
        print("Restoring model starts...")
        saver = tf.train.Saver()
        print("Restoring from {}".format(tf.train.latest_checkpoint(SAVE_DIR)))
        saver.restore(sess, tf.train.latest_checkpoint(SAVE_DIR))
        print("Restoring model done.")
        # run model
        output = sess.run(dlr, feed_dict = {y : np.reshape(input_, [1, -1])})
    output = np.squeeze(output)
    return output 

def get_DLR_2D(input_):
    '''
    Args:
        input_ - 2D numpy array [ndata, length]
    Return:
        output - 3D numpy array DLR
    '''
    bath_size = 20
    ndata, length = input_.shape
    # normalize
    for i in range(ndata):
        normalize = np.max(np.abs(input_[i]))
        if normalize != 0:
            input_[i] /= normalize
    # abs      
    input_ = np.abs(input_)

    output = list()
    if ndata % batch_size != 0:
        output = np.concatenate([input_, np.zeros([batch_size - ndata % batch_size, length])], axis=0)

    nbatch = len(input_)//batch_size
    # build model
    tf.reset_default_graph()
    y = tf.placeholder(tf.float32, shape=[batch_size, length])
    with tf.variable_scope("DLR"):
        dlr = dlr13(y)
    print(vars_info("trainable_variables"))
    with tf.Session() as sess:
        # restore model
        print("Restoring model starts...")
        saver = tf.train.Saver()
        print("Restoring from {}".format(tf.train.latest_checkpoint(BALLROOM_SAVE_DIR)))
        saver.restore(sess, tf.train.latest_checkpoint(BALLROOM_SAVE_DIR))
        print("Restoring model done.")
        # run model
        for b in range(nbatch):
            feed_dict = {y: input_[b * batch_size:batch_size * (b + 1)]}
            output.append(sess.run(dlr, feed_dict=feed_dict))

    output = np.concatenate(output, axis = 0)
    output = output[:ndata]
    output = np.squeeze(output) # 4D to 3D
    return output

def get_mel_stft(y, n_fft=1024, win_len=512, hop_len=256, sr=8000, n_mels=128):
    '''
    Args : y - 1D signal
    customized mel spectogram
    I prefer [time, freq]
    '''
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_len, win_length=win_len)
    melspec = librosa.feature.melspectrogram(sr=sr, S=S, n_fft=n_fft, hop_length=hop_len, power=2.0, n_mels=n_mels)
    melspec_t = np.transpose(melspec)
    melspec_db = librosa.amplitude_to_db(melspec_t)
    std = melspec_db.std()
    if std != 0:
        melspec_db = (melspec_db - melspec_db.mean())/melspec_db.std()
    else:
        melspec_db = melspec_db - melspec_db.mean()
    return melspec_db

def my_tempogram(y=None, new_hop=1, sr=22050, onset_envelope=None, hop_length=512,
              win_length=384, center=True, window='hann', norm=np.inf):

    if win_length < 1:
        raise librosa.util.exceptions.ParameterError('win_length must be a positive integer')

    ac_window = librosa.filters.get_window(window, win_length, fftbins=True)

    if onset_envelope is None:
        if y is None:
            raise librosa.util.exceptions.ParameterError('Either y or onset_envelope must be provided')

        onset_envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Center the autocorrelation windows
    n = len(onset_envelope)

    if center:
        onset_envelope = np.pad(onset_envelope, int(win_length // 2),
                                mode='linear_ramp', end_values=[0, 0])

    # Carve onset envelope into frames
    odf_frame = librosa.util.frame(onset_envelope,
                           frame_length=win_length,
                           hop_length=new_hop)

    # Truncate to the length of the original signal
    if center:
        odf_frame = odf_frame[:, :n]

    # Window, autocorrelate, and normalize
    return librosa.util.normalize(librosa.core.audio.autocorrelate(odf_frame * ac_window[:, np.newaxis], axis=0), norm=norm, axis=0)

def get_tempogram(y):
    tempogram = my_tempogram(y, new_hop=2, hop_length=128, win_length=256, sr=8000)
    tempogram_t = np.transpose(tempogram, [1,0])
    return tempogram_t
