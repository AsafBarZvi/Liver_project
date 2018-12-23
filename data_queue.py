#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   17.09.2017
#-------------------------------------------------------------------------------
# This file is part of SSD-TensorFlow.
#
# SSD-TensorFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SSD-TensorFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SSD-Tensorflow.  If not, see <http://www.gnu.org/licenses/>.
#-------------------------------------------------------------------------------

import Queue as q
import numpy as np
import multiprocessing as mp

#-------------------------------------------------------------------------------
class DataQueue:
    #---------------------------------------------------------------------------
    def __init__(self, images_template, segs_template, maxsize):
        #-----------------------------------------------------------------------
        # Figure out the data tupes, sizes and shapes of both arrays
        #-----------------------------------------------------------------------
        self.images_dtype = images_template.dtype
        self.images_shape = images_template.shape
        self.images_bc = len(images_template.tobytes())
        self.segs_dtype = segs_template.dtype
        self.segs_shape = segs_template.shape
        self.segs_bc = len(segs_template.tobytes())

        #-----------------------------------------------------------------------
        # Make an array pool and queue
        #-----------------------------------------------------------------------
        self.array_pool = []
        self.array_queue = mp.Queue(maxsize)
        for i in range(maxsize):
            images_buff = mp.Array('c', self.images_bc, lock=False)
            images_arr = np.frombuffer(images_buff, dtype=self.images_dtype)
            images_arr = images_arr.reshape(self.images_shape)

            segs_buff = mp.Array('c', self.segs_bc, lock=False)
            segs_arr = np.frombuffer(segs_buff, dtype=self.segs_dtype)
            segs_arr = segs_arr.reshape(self.segs_shape)

            self.array_pool.append((images_arr, segs_arr))
            self.array_queue.put(i)

        self.queue = mp.Queue(maxsize)

    #---------------------------------------------------------------------------
    def put(self, images, segs, *args, **kwargs):
        #-----------------------------------------------------------------------
        # Check whether the params are consistent with the data we can store
        #-----------------------------------------------------------------------
        def check_consistency(name, arr, dtype, shape, byte_count):
            if type(arr) is not np.ndarray:
                raise ValueError(name + ' needs to be a numpy array')
            if arr.dtype != dtype:
                raise ValueError('{}\'s elements need to be of type {} but is {}' \
                                 .format(name, str(dtype), str(arr.dtype)))
            if arr.shape != shape:
                raise ValueError('{}\'s shape needs to be {} but is {}' \
                                 .format(name, shape, arr.shape))
            if len(arr.tobytes()) != byte_count:
                raise ValueError('{}\'s byte count needs to be {} but is {}' \
                                 .format(name, byte_count, len(arr.data)))

        check_consistency('img', images, self.images_dtype, self.images_shape, self.images_bc)
        check_consistency('segs', segs, self.segs_dtype, self.segs_shape, self.segs_bc)

        #-----------------------------------------------------------------------
        # If we can not get the slot within timeout we are actually full, not
        # empty
        #-----------------------------------------------------------------------
        try:
            arr_id = self.array_queue.get(*args, **kwargs)
        except q.Empty:
            raise q.Full()

        #-----------------------------------------------------------------------
        # Copy the arrays into the shared pool
        #-----------------------------------------------------------------------
        self.array_pool[arr_id][0][:] = images
        self.array_pool[arr_id][1][:] = segs
        self.queue.put((arr_id), *args, **kwargs)

    #---------------------------------------------------------------------------
    def get(self, *args, **kwargs):
        item = self.queue.get(*args, **kwargs)
        arr_id = item

        images = np.copy(self.array_pool[arr_id][0])
        segs = np.copy(self.array_pool[arr_id][1])

        self.array_queue.put(arr_id)

        return images, segs

    #---------------------------------------------------------------------------
    def empty(self):
        return self.queue.empty()
