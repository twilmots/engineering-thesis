########################################################################
#
# File:   DataLoader.py
# Author: Tom Wilmots
# Date:   April 14, 2017
#
# Written for ENGR 90
#
########################################################################
#
# This script is a simple XML parser for our E90 data

import os
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pickle

class DataLoader():
  def __init__(self, batch_size=50, seq_length=300, scale_factor = 10, limit = 500):
    self.data_dir = "./Data"
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.scale_factor = scale_factor # divide data by this factor
    self.limit = limit # removes large noisy gaps in the data

    data_file = os.path.join(self.data_dir, "strokes_training_data.cpkl")
    raw_data_dir = self.data_dir+"/lineStrokes"

    if not (os.path.exists(data_file)) :
        print("creating training data pkl file from raw source")
        self.preprocess(raw_data_dir, data_file)

    self.load_preprocessed(data_file)
    self.reset_batch_pointer()

  def preprocess(self, data_dir, data_file):
    # create data file from raw xml files from iam handwriting source.

    # build the list of xml files
    filelist = []
    # Set the directory you want to start from
    rootDir = data_dir
    for dirName, subdirList, fileList in os.walk(rootDir):
      #print('Found directory: %s' % dirName)
      for fname in fileList:
        #print('\t%s' % fname)
        filelist.append(dirName+"/"+fname)

    # function to read each individual xml file
    def getStrokes(filename):
      tree = ET.parse(filename)
      root = tree.getroot()

      result = []

      x_offset = 1e20
      y_offset = 1e20
      y_height = 0
      for i in range(1, 4):
        x_offset = min(x_offset, float(root[0][i].attrib['x']))
        y_offset = min(y_offset, float(root[0][i].attrib['y']))
        y_height = max(y_height, float(root[0][i].attrib['y']))
      y_height -= y_offset
      x_offset -= 100
      y_offset -= 100

      for stroke in root[1].findall('Stroke'):
        points = []
        for point in stroke.findall('Point'):
          points.append([float(point.attrib['x'])-x_offset,float(point.attrib['y'])-y_offset])
        result.append(points)

      return result

    # converts a list of arrays into a 2d numpy int16 array
    def convert_stroke_to_array(stroke):

      n_point = 0
      for i in range(len(stroke)):
        n_point += len(stroke[i])
      stroke_data = np.zeros((n_point, 3), dtype=np.int16)

      prev_x = 0
      prev_y = 0
      counter = 0

      for j in range(len(stroke)):
        for k in range(len(stroke[j])):
          stroke_data[counter, 0] = int(stroke[j][k][0]) - prev_x
          stroke_data[counter, 1] = int(stroke[j][k][1]) - prev_y
          prev_x = int(stroke[j][k][0])
          prev_y = int(stroke[j][k][1])
          stroke_data[counter, 2] = 0
          if (k == (len(stroke[j])-1)): # end of stroke
            stroke_data[counter, 2] = 1
          counter += 1
      return stroke_data

    # build stroke database of every xml file inside iam database
    strokes = []
    for i in range(len(filelist)):
      if (filelist[i][-3:] == 'xml'):
        print('processing '+filelist[i])
        strokes.append(convert_stroke_to_array(getStrokes(filelist[i])))

    f = open(data_file,"wb")
    pickle.dump(strokes, f, protocol=2)
    f.close()

  def load_preprocessed(self, data_file):
    f = open(data_file,"rb")
    self.raw_data = pickle.load(f)
    f.close()

    # goes thru the list, and only keeps the text entries that have more than seq_length points
    self.data = []
    self.valid_data =[]
    counter = 0

    # every 1 in 20 (5%) will be used for validation data
    cur_data_counter = 0
    for data in self.raw_data:
      if len(data) > (self.seq_length+2):
        # removes large gaps from the data
        data = np.minimum(data, self.limit)
        data = np.maximum(data, -self.limit)
        data = np.array(data,dtype=np.float32)
        data[:,0:2] /= self.scale_factor
        cur_data_counter = cur_data_counter + 1
        if cur_data_counter % 20 == 0:
          self.valid_data.append(data)
        else:
          self.data.append(data)
          counter += int(len(data)/((self.seq_length+2))) # number of equiv batches this datapoint is worth

    print("train data: {}, valid data: {}".format(len(self.data), len(self.valid_data)))
    print('')
    print('')
    # minus 1, since we want the ydata to be a shifted version of x data
    self.num_batches = int(counter / self.batch_size)

  def validation_data(self):
    # returns validation data
    x_batch = []
    y_batch = []
    for i in range(self.batch_size):
      data = self.valid_data[i%len(self.valid_data)]
      idx = 0
      x_batch.append(np.copy(data[idx:idx+self.seq_length]))
      y_batch.append(np.copy(data[idx+1:idx+self.seq_length+1]))
    return x_batch, y_batch

  def next_batch(self):
    # returns a randomised, seq_length sized portion of the training data
    x_batch = []
    y_batch = []
    for i in range(self.batch_size):
      data = self.data[self.pointer]
      n_batch = int(len(data)/((self.seq_length+2))) # number of equiv batches this datapoint is worth
      idx = random.randint(0, len(data)-self.seq_length-2)
      x_batch.append(np.copy(data[idx:idx+self.seq_length]))
      y_batch.append(np.copy(data[idx+1:idx+self.seq_length+1]))
      if random.random() < (1.0/float(n_batch)): # adjust sampling probability.
        #if this is a long datapoint, sample this data more with higher probability
        self.tick_batch_pointer()
    return x_batch, y_batch

  def tick_batch_pointer(self):
    self.pointer += 1
    if (self.pointer >= len(self.data)):
      self.pointer = 0

  def reset_batch_pointer(self):
    self.pointer = 0


