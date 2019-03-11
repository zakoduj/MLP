import org.jblas.DoubleMatrix;

import java.util.function.Supplier;

/**
 * Holds layer learning logic.
 */
abstract class AbstractLayer {
    /**
     * Layer weights.
     */
    private DoubleMatrix weights;
    /**
     * Layer biases.
     */
    private DoubleMatrix biases;
    /**
     * Learning rate.
     */
    private final double learningRate = 0.1;

    /**
     * Default and only c-tor, which takes layer dimensions, as well as initial value provider for weights and biases.
     * @param inputSize inputSize size
     * @param outputSize outputSize size
     * @param init initial value provider
     */
    AbstractLayer(int inputSize, int outputSize, Supplier<Double> init) {
        this.weights = new DoubleMatrix(outputSize, inputSize);
        for (int i = 0; i < this.weights.length; i++) {
            this.weights.put(i, init.get());
        }
        this.biases = new DoubleMatrix(outputSize, 1);
        for (int i = 0; i < this.biases.length; i++) {
            this.biases.put(i, init.get());
        }
    }

    /**
     * Feed forward routine.
     * @param matrix
     * @return
     */
    DoubleMatrix feedForward(DoubleMatrix matrix) {
        return this.activate(this.weights.mmul(matrix).add(this.biases));
    }

    /**
     * Back propagation routine.
     * @param predicted result from current layer
     * @param actual actual value
     * @param previous result from previous layer
     * @return result for next layer
     */
    DoubleMatrix backPropagate(DoubleMatrix predicted, DoubleMatrix actual, DoubleMatrix previous) {
        // Calculate error
        DoubleMatrix error = actual.sub(predicted);
        // Calculate gradient
        DoubleMatrix gradient = this.deactivate(predicted).mul(error).mul(this.learningRate);
        // Calculate delta
        DoubleMatrix delta = gradient.mmul(previous.transpose());
        // Update weights
        this.weights = this.weights.add(delta);
        // Update biases
        this.biases = this.biases.add(gradient);
        // Calculate target for next layer
        return this.weights.transpose().mmul(error).add(previous);
    }

    /**
     * Activation function.
     * @param matrix
     * @return activated matrix
     */
    abstract DoubleMatrix activate(DoubleMatrix matrix);

    /**
     * Derivative of activation function to use in back propagation. Not really sure how to name it.
     * @param matrix
     * @return deactivated matrix
     */
    abstract DoubleMatrix deactivate(DoubleMatrix matrix);
}
