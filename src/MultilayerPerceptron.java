import org.jblas.DoubleMatrix;

import java.util.Random;
import java.util.function.Predicate;

class MultilayerPerceptron {
    private final Layer[] layers;
    private final Random random = new Random();

    MultilayerPerceptron(int inputSize, int outputSize) {
        this(inputSize, inputSize * inputSize, outputSize);
    }

    MultilayerPerceptron(int inputSize, int hiddenNodes, int outputSize) {
        this(inputSize, 1, hiddenNodes, outputSize);
    }

    MultilayerPerceptron(int inputSize, int hiddenLayers, int hiddenNodes, int outputSize) {
        this.layers = new Layer[1 + hiddenLayers + 1];
        this.layers[0] = new Layer(inputSize, hiddenNodes, random);
        for (int i = 0; i < hiddenLayers; i++) {
            this.layers[i + 1] = new Layer(hiddenNodes, hiddenNodes, random);
        }
        this.layers[this.layers.length - 1] = new Layer(hiddenNodes, outputSize, random);
    }

    MultilayerPerceptron(Layer... layers) {
        this.layers = layers;
    }

    double[] classify(double[] input) {
        DoubleMatrix in = new DoubleMatrix(input.length, 1, input);
        for (Layer layer : this.layers) {
            in = layer.feedForward(in);
        }
        return in.toArray();
    }

    void train(double[][] inputs, double[][] outputs, Predicate<Double> predicate) {
        if (inputs.length != outputs.length) {
            throw new IllegalArgumentException("Inputs size != outputs size.");
        }
        double error;
        do {
            error = 0;
            for (int index = 0; index < inputs.length; index++) {
                error += this.train(inputs[index], outputs[index]);
            }
        } while (predicate.test(error / outputs.length));
    }

    private double train(double[] input, double[] output) {
        DoubleMatrix[] cache = new DoubleMatrix[this.layers.length + 1];
        DoubleMatrix in = cache[0] = new DoubleMatrix(input.length, 1, input);

        // 1. Feed forward
        for (int i = 0; i < this.layers.length; i++) {
            cache[i + 1] = in = this.layers[i].feedForward(in);
        }

        // 2. Back propagate though layers in reverse order
        DoubleMatrix out, copy = out = new DoubleMatrix(output.length, 1, output);
        for (int i = this.layers.length - 1; i >= 0; i--) {
            out = this.layers[i].backPropagate(cache[i + 1], out, cache[i]);
        }

        // 3. Calculate error
        return crossEntropy(copy, cache[cache.length - 1]);
    }

    private double crossEntropy(DoubleMatrix a, DoubleMatrix b) {
        DoubleMatrix temp = new DoubleMatrix(b.rows, b.columns);
        for (int j = 0; j < b.columns; j++) {
            for (int i = 0; i < b.rows; i++) {
                temp.put(i, j, (a.get(i, j) * Math.log(b.get(i, j))) + ((1 - a.get(i, j)) * Math.log(1 - b.get(i, j))));
            }
        }
        return -temp.sum();
    }
}
