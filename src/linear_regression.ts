import * as tf from '@tensorflow/tfjs-node';
import readline from "readline"

interface KcHouseDataRecord {
	price: number,
	sqft_living: number,
}

const rl = readline.createInterface({
	input: process.stdin,
	output: process.stdout
});

const data = tf.data.csv("http://127.0.0.1:8080/kc_house_data.csv")

export async function main() {
	const dataset = await data.map(
		// @ts-expect-error: Expect the columns to exist
		(record: KcHouseDataRecord) => ({
			x: record.sqft_living,
			y: record.price,
		})
	).toArray()

	if (dataset.length % 2 !== 0) {
		dataset.pop();
	}
	tf.util.shuffle(dataset);

	const values = dataset.map(p => p.x)
	const tensorX = tf.tensor2d(values, [values.length, 1])
	const { tensor: tensorXnormalized, min: minX, max: maxX } = normalize(tensorX)
	const [trainingX, testingX] = tf.split(tensorXnormalized, 2);

	const labels = dataset.map(p => p.y)
	const tensorY = tf.tensor2d(labels, [labels.length, 1])
	const { tensor: tensorYnormalized, min: minY, max: maxY } = normalize(tensorY)
	const [trainingY, testingY] = tf.split(tensorYnormalized, 2);

	const model = createModel();
	const result = await trainModel(model, trainingX, trainingY);
	const trainingLoss = result.history.loss.pop();
	const validationLoss = result.history.val_loss.pop();
	const lossTensor = model.evaluate(testingX, testingY) as tf.Scalar;
	const loss = await lossTensor.dataSync();
	console.log("\nloss", loss, "\ntrainingLoss", trainingLoss, "\nvalidationLoss", validationLoss);

	model.save(`file://dist/model-kc-house-regression`)

	rl.question("\nEnter a house square footage to get a price estimate: ", (answer) => {
		const inputTensor = tf.tensor1d([Number(answer)])
		const normalizedInput = normalize(inputTensor, minX, maxX);
		const normalizedOutputTensor = model.predict(normalizedInput.tensor) as tf.Tensor
		const output = denormalize(normalizedOutputTensor, minY, maxY).dataSync()
		console.log("Prediction: $", Number(output[0]).toLocaleString());
		rl.close();
	});
}


const normalize = (
	tensor: tf.Tensor, previousMin: tf.Tensor | null = null, previousMax: tf.Tensor | null = null
): { tensor: tf.Tensor, min: tf.Tensor, max: tf.Tensor } => {
	const min = previousMin || tensor.min()
	const max = previousMax || tensor.max()
	return {
		tensor: tensor.sub(min).div(max.sub(min)),
		min: (min as tf.Tensor),
		max: (max as tf.Tensor)
	}
}

const denormalize = (tensor: tf.Tensor, min: tf.Tensor, max: tf.Tensor) => tf.tidy(() => {
	return tensor.mul(max.sub(min)).add(min)
})

function createModel() {
	const model = tf.sequential();

	model.add(tf.layers.dense({
		units: 1,
		useBias: true,
		activation: 'linear',
		inputDim: 1,

	}))

	model.compile({
		loss: "meanSquaredError",
		optimizer: tf.train.sgd(0.1)
	})

	return model;
}

function trainModel(model: tf.Sequential, tensorX: tf.Tensor, tensorY: tf.Tensor) {
	return model.fit(tensorX, tensorY, {
		batchSize: 32,
		validationSplit: 0.2,
		epochs: 20,
		callbacks: {
			onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss=${log?.loss}`)
		}
	})
}
