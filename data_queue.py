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
    def __init__(self, frames_template, lines_template, semantics_template, clipNames_template, origins_template, maxsize):
        #-----------------------------------------------------------------------
        # Figure out the data tupes, sizes and shapes of both arrays
        #-----------------------------------------------------------------------
        self.frames_dtype = frames_template.dtype
        self.frames_shape = frames_template.shape
        self.frames_bc = len(frames_template.tobytes())
        self.lines_dtype = lines_template.dtype
        self.lines_shape = lines_template.shape
        self.lines_bc = len(lines_template.tobytes())
        self.semantics_dtype = semantics_template.dtype
        self.semantics_shape = semantics_template.shape
        self.semantics_bc = len(semantics_template.tobytes())
        self.clipNames_dtype = clipNames_template.dtype
        self.clipNames_shape = clipNames_template.shape
        self.clipNames_bc = len(clipNames_template.tobytes())
        self.origins_dtype = origins_template.dtype
        self.origins_shape = origins_template.shape
        self.origins_bc = len(origins_template.tobytes())

        #-----------------------------------------------------------------------
        # Make an array pool and queue
        #-----------------------------------------------------------------------
        self.array_pool = []
        self.array_queue = mp.Queue(maxsize)
        for i in range(maxsize):
            frames_buff = mp.Array('c', self.frames_bc, lock=False)
            frames_arr = np.frombuffer(frames_buff, dtype=self.frames_dtype)
            frames_arr = frames_arr.reshape(self.frames_shape)

            lines_buff = mp.Array('c', self.lines_bc, lock=False)
            lines_arr = np.frombuffer(lines_buff, dtype=self.lines_dtype)
            lines_arr = lines_arr.reshape(self.lines_shape)

            semantics_buff = mp.Array('c', self.semantics_bc, lock=False)
            semantics_arr = np.frombuffer(semantics_buff, dtype=self.semantics_dtype)
            semantics_arr = semantics_arr.reshape(self.semantics_shape)

            clipNames_buff = mp.Array('c', self.clipNames_bc, lock=False)
            clipNames_arr = np.frombuffer(clipNames_buff, dtype=self.clipNames_dtype)
            clipNames_arr = clipNames_arr.reshape(self.clipNames_shape)

            origins_buff = mp.Array('c', self.origins_bc, lock=False)
            origins_arr = np.frombuffer(origins_buff, dtype=self.origins_dtype)
            origins_arr = origins_arr.reshape(self.origins_shape)

            self.array_pool.append((frames_arr, lines_arr, semantics_arr, clipNames_arr, origins_arr))
            self.array_queue.put(i)

        self.queue = mp.Queue(maxsize)

    #---------------------------------------------------------------------------
    def put(self, frames, lines, semantics, clipNames, origins, *args, **kwargs):
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

        check_consistency('img', frames, self.frames_dtype, self.frames_shape, self.frames_bc)
        check_consistency('lines', lines, self.lines_dtype, self.lines_shape, self.lines_bc)
        check_consistency('semantics', semantics, self.semantics_dtype, self.semantics_shape, self.semantics_bc)
        #check_consistency('clipNames', clipNames, self.clipNames_dtype, self.clipNames_shape, self.clipNames_bc)
        check_consistency('origins', origins, self.origins_dtype, self.origins_shape, self.origins_bc)

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
        self.array_pool[arr_id][0][:] = frames
        self.array_pool[arr_id][1][:] = lines
        self.array_pool[arr_id][2][:] = semantics
        self.array_pool[arr_id][3][:] = clipNames
        self.array_pool[arr_id][4][:] = origins
        self.queue.put((arr_id), *args, **kwargs)

    #---------------------------------------------------------------------------
    def get(self, *args, **kwargs):
        item = self.queue.get(*args, **kwargs)
        arr_id = item

        frames = np.copy(self.array_pool[arr_id][0])
        lines = np.copy(self.array_pool[arr_id][1])
        semantics = np.copy(self.array_pool[arr_id][2])
        clipNames = np.copy(self.array_pool[arr_id][3])
        origins = np.copy(self.array_pool[arr_id][4])

        self.array_queue.put(arr_id)

        return frames, lines, semantics, clipNames, origins

    #---------------------------------------------------------------------------
    def empty(self):
        return self.queue.empty()
