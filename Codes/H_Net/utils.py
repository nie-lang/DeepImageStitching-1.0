import tensorflow as tf
import numpy as np
from collections import OrderedDict
import os
import glob
import cv2


#rng = np.random.RandomState(2017)


def np_load_frame(filename, resize_height, resize_width):
    image_decoded = cv2.imread(filename)
    #image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_decoded.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized


class InputLoader(object):
    def __init__(self, video_folder, resize_height=128, resize_width=128):
        self.dir = video_folder
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self.setup()

    def __call__(self, batch_size):
        video_info_list = list(self.videos.values())
        num_videos = len(video_info_list)
        resize_height, resize_width = self._resize_height, self._resize_width

        def video_clip_generator():
            frame_id = 0
            while True:
                video_clip = []
                for i in range(0, num_videos):
                    video_clip.append(np_load_frame(video_info_list[i]['frame'][frame_id], resize_height, resize_width))
                video_clip = np.concatenate(video_clip, axis=2)
                frame_id = (frame_id + 1)%video_info_list[0]['length'] 
                yield video_clip

        # video clip paths
        dataset = tf.data.Dataset.from_generator(generator=video_clip_generator,
                                                 output_types=tf.float32,
                                                 output_shapes=[resize_height, resize_width, 2 * 3])
        print('generator dataset, {}'.format(dataset))
        dataset = dataset.prefetch(buffer_size=batch_size)
        dataset = dataset.batch(batch_size)
        print('epoch dataset, {}'.format(dataset))

        return dataset

    def __getitem__(self, video_name):
        assert video_name in self.videos.keys(), 'video = {} is not in {}!'.format(video_name, self.videos.keys())
        return self.videos[video_name]

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            if video_name == 'input1' or video_name == 'input2':
                self.videos[video_name] = {}
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
                self.videos[video_name]['frame'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])

        print(self.videos.keys())

    def get_video_clips(self, index):
        batch = []
        video_info_list = list(self.videos.values())
        for i in range(0, 2):
            image = np_load_frame(video_info_list[i]['frame'][index], self._resize_height, self._resize_width)
            batch.append(image)
        return np.concatenate(batch, axis=2)
        
        
def np_load_h(filename):
    shift = np.load(filename)
    return shift


class HLoader(object):
    def __init__(self, video_folder):
        self.dir = video_folder
        self.videos = OrderedDict()
        self.setup()

    def __call__(self, batch_size):
        video_info_list = list(self.videos.values())
        def video_clip_generator():
            frame_id = 0
            while True:
                video_clip = []
                video_clip.append(np_load_h(video_info_list[0]['frame'][frame_id]))
                video_clip = np.concatenate(video_clip, axis=1)
                #video_clip = np_load_frame(video_info_list[0]['frame'][frame_id])
                frame_id = (frame_id + 1)%video_info_list[0]['length'] 
                yield video_clip
        # video clip paths
        dataset = tf.data.Dataset.from_generator(generator=video_clip_generator,
                                                 output_types=tf.float32,
                                                 output_shapes=[8, 1])
        print('generator dataset, {}'.format(dataset))
        dataset = dataset.prefetch(buffer_size=batch_size)
        dataset = dataset.batch(batch_size)
        print('epoch dataset, {}'.format(dataset))

        return dataset

    def __getitem__(self, video_name):
        assert video_name in self.videos.keys(), 'video = {} is not in {}!'.format(video_name, self.videos.keys())
        return self.videos[video_name]

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            if video_name == 'shift':
                self.videos[video_name] = {}
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.npy'))
                self.videos[video_name]['frame'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
        print(self.videos.keys())

    def get_video_clips(self, index):
        batch = []
        video_info_list = list(self.videos.values())
        for i in range(0, 1):
            image = np_load_h(video_info_list[i]['frame'][index])
            batch.append(image)
        return np.concatenate(batch, axis=1)





def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')




