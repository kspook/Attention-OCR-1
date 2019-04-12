__author__ = 'moonkey'

import io
import os
import numpy as np
from PIL import Image
from collections import Counter
import pickle as cPickle
import random, math
from data_util.bucketdata import BucketData


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  #'../labels/bank_labelsSW.txt')
                                  '../labels/bank_labelsS.txt')								  

class DataGen(object):
    GO = 1
    EOS = 2

    def __init__(self,
                 data_root, annotation_fn,
                 evaluate = False,
                 valid_target_len = float('inf'),
                 img_width_range = (12, 320),
                 word_len = 30):
        """
        :param data_root:
        :param annotation_fn:
        :param lexicon_fn:
        :param img_width_range: only needed for training set
        :return:
        """

        img_height = 32
        self.data_root = data_root
        if os.path.exists(annotation_fn):
            self.annotation_path = annotation_fn
        else:
            self.annotation_path = os.path.join(data_root, annotation_fn)

        if evaluate:
            self.bucket_specs = [(int(math.floor(64 / 4)), int(word_len + 2)), (int(math.floor(108 / 4)), int(word_len + 2)),
                                 (int(math.floor(140 / 4)), int(word_len + 2)), (int(math.floor(256 / 4)), int(word_len + 2)),
                                 (int(math.floor(img_width_range[1] / 4)), int(word_len + 2))]
        else:
            self.bucket_specs = [(int(64 / 4), 9 + 2), (int(108 / 4), 15 + 2),
                             (int(140 / 4), 17 + 2), (int(256 / 4), 20 + 2),
                             (int(math.ceil(img_width_range[1] / 4)), word_len + 2)]

        self.bucket_min_width, self.bucket_max_width = img_width_range
        self.image_height = img_height
        self.valid_target_len = valid_target_len

        self.bucket_data = {i: BucketData()
                            for i in range(self.bucket_max_width + 1)}

    def clear(self):
        self.bucket_data = {i: BucketData()
                            for i in range(self.bucket_max_width + 1)}

    def get_size(self):
        with open(self.annotation_path, 'r', encoding='utf-8') as ann_file:
            return len(ann_file.readlines())

    def gen(self, batch_size):
        valid_target_len = self.valid_target_len


        with open(self.annotation_path, 'r', encoding='utf-8') as ann_file:
            lines = ann_file.readlines()
            print('annotation_path ', self.annotation_path)
            random.shuffle(lines)
            for l in lines:
                img_path, lex = l.rstrip().split('  ')
                #img_path, lex = l.strip().split()				
                #print('img_path, lex', img_path, lex)
                try:
                    img_bw, word = self.read_data(img_path, lex)
                    #print('data gen : img_bw,  word=self.read_data',  word)
                    if valid_target_len < float('inf'):
                        word = word[:valid_target_len + 1]
                    width = img_bw.shape[-1]
                    #print('width', width)

                    # TODO:resize if > 320
                    b_idx = min(width, self.bucket_max_width)
                    bs = self.bucket_data[b_idx].append(img_bw, word, os.path.join(self.data_root,img_path))
                    if bs >= batch_size:
                        b = self.bucket_data[b_idx].flush_out(
                                self.bucket_specs,
                                valid_target_length=valid_target_len,
                                go_shift=1)
                        if b is not None:
                            yield b
                        else:
                            assert False, 'no valid bucket of width %d'%width
                except IOError:
                    pass # ignore error images
                    #with open('error_img.txt', 'a') as ef:
                    #    ef.write(img_path + '\n')
        self.clear()

    def read_data(self, img_path, lex):
        assert 0 < len(lex) < self.bucket_specs[-1][1]
        #print('read_data:self.data_root', self.data_root)
        # L = R * 299/1000 + G * 587/1000 + B * 114/1000
        with open(os.path.join(self.data_root, img_path), 'rb') as img_file:
            img = Image.open(img_file)
            w, h = img.size
            #print('w,h : ', w,h)
            aspect_ratio = float(w) / float(h)
            if aspect_ratio < float(self.bucket_min_width) / self.image_height:
                img = img.resize(
                    (self.bucket_min_width, self.image_height),
                    Image.ANTIALIAS)
            elif aspect_ratio > float(
                    self.bucket_max_width) / self.image_height:
                img = img.resize(
                    (self.bucket_max_width, self.image_height),
                    Image.ANTIALIAS)
            elif h != self.image_height:
                img = img.resize(
                    (int(aspect_ratio * self.image_height), self.image_height),
                    Image.ANTIALIAS)

            img_bw = img.convert('L')
            img_bw = np.asarray(img_bw, dtype=np.uint8)
            img_bw = img_bw[np.newaxis, :]

        # 'a':97, '0':48
        word = [self.GO]
        #num_char = 123+26+256+3+1
        # , 44 . 46    0~9 :48-58  A-Z : 65-90 a-z :97~122  

        try:
            fp=open('outputs.txt', 'w+', encoding='utf-8')
        except:
            print('could not open file'+outputs.txt)
            quit()

        '''        
        for c in lex:
            #assert 96 < ord(c) < 123 or 47 < ord(c) < 58
            #print('c', ord(c))
            fp.write(img_path+'\t'+lex+'\t')
            fp.write(str(ord(c)))
            word.append(
                ord(c) - 44032- 43 +43 + 117 +3   if ord(c) > 44031   
                else ord(c) - 43 + 3)
                #else ord(c) - 43 + 3)))))
            fp.write('\n')
        fp.close()
        '''        
        '''        
        for c in lex:
            #0assert 96 < ord(c) < 123 or 47 < ord(c) < 58
            print('c', ord(c))
            word.append(
                ord(c) - 97 + 13 if ord(c) > 96 else ord(c) - 48 + 3)
        '''
        label_file = DEFAULT_LABEL_FILE
        with io.open(label_file, 'r', encoding='utf-8') as f:
           labels = f.read().splitlines()

        for c in lex:
            print('c ord(c)', c, ord(c))
            for i, l in enumerate(labels):
                #print('i  l', i , l)
                if c== l:
                   n=i+3
                   print('data gen c ord(c) l i n : ', c, ord(c), l, i, n)
                   word.append(n)
                '''
                else:
                   print('exit')
                   exit()
                '''

        word.append(self.EOS)
        '''
        word = np.array(word, dtype=np.int32)
        word = np.array( [self.GO] +
        [ord(c) -44032- 43 + 43 + 117 +3   if ord(c) > 44031   
        #else ord(c) - 43 + 3
        #else ord(c) - 43 + 3))))
        #[ord(c) - 97 + 13 if ord(c) > 96 else ord(c) - 48 + 3
        for c in lex] + [self.EOS], dtype=np.int32)   
        word = np.array( [self.GO] +
        word + [self.EOS], dtype=np.int32)   
        '''
        '''
        fp=open(outputs.txt, 'w', endconding='utf8')
        fp.truncate()
        count=0
        for (i in range(0, len(word)):
            #for(j in range(o, word[i]):
            fp.write(word[i])    
            fp.write('\n')
        fp.close()
        '''
        #print('ord(c), c: ', ord(c), c)
        #print('word:', word)

        return img_bw, word


def test_gen():
    print('testing gen_valid')
    # s_gen = EvalGen('../../data/evaluation_data/svt', 'test.txt')
    # s_gen = EvalGen('../../data/evaluation_data/iiit5k', 'test.txt')
    # s_gen = EvalGen('../../data/evaluation_data/icdar03', 'test.txt')
    s_gen = EvalGen('../../data/evaluation_data/icdar13', 'test.txt')
    count = 0
    for batch in s_gen.gen(1):
        count += 1
        print(str(batch['bucket_id']) + ' ' + str(batch['data'].shape[2:]))
        assert batch['data'].shape[2] == img_height
    print(count)


if __name__ == '__main__':
    test_gen()
