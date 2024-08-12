import * as tf from '@tensorflow/tfjs-node';

interface KcHouseDataRecord {
	price: number,
	sqft_living: number,
}

const data = tf.data.csv("http://127.0.0.1:8080/kc_house_data.csv")

export async function main() {
	const dataset = await data.map(
		// @ts-expect-error: Expect the columns to exist
		(record: KcHouseDataRecord) => ({
			x: record.sqft_living,
			y: record.price,
		})
	).toArray()

	if(dataset.length % 2 !== 0) {
		dataset.pop();
	}
	tf.util.shuffle(dataset);

	const values = dataset.map(p => p.x)
	const tensorX = tf.tensor2d(values, [values.length, 1])
	const { tensor: tensorXnormalized, min: minX, max: maxX } = normalize(tensorX)
	const [trainingX, testingX] = tf.split(tensorXnormalized, 2);
	
	denormalize(tensorXnormalized, minX, maxX).print()
	
	const labels = dataset.map(p => p.y)
	const tensorY = tf.tensor2d(labels, [labels.length, 1])
	const { tensor: tensorYnormalized, min: minY, max: maxY } = normalize(tensorY)
	const [trainingY, testingY] = tf.split(tensorYnormalized, 2);

	denormalize(tensorYnormalized, minY, maxY).print()
}


const normalize = (tensor: tf.Tensor2D): { tensor: tf.Tensor2D, min: tf.Tensor2D, max: tf.Tensor2D } => {
	const min = tensor.min()
	const max = tensor.max()
	return {
		tensor: tensor.sub(min).div(max.sub(min)),
		min: (min as tf.Tensor2D),
		max: (max as tf.Tensor2D)
	}
}

const denormalize = (tensor: tf.Tensor2D, min: tf.Tensor2D, max: tf.Tensor2D) => tf.tidy(() => {
	return tensor.mul(max.sub(min)).add(min)
})