import * as tf from '@tensorflow/tfjs-node';

tf.tensor4d([1,2,3,4], [1,2,2,1]).print();

console.log(tf.memory())