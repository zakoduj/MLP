import org.jblas.DoubleMatrix;

import java.util.function.Supplier;

/**
 * More on activation functions here:
 * https://www.jeremyjordan.me/neural-networks-activation-functions/
 * Extension class of base layer class, which defines what activation function to use for current layer.
 */
class SigmoidLayer extends AbstractLayer {
    SigmoidLayer(int inputSize, int outputSize, Supplier<Double> supplier) {
        super(inputSize, outputSize, supplier);
    }

    @Override
    DoubleMatrix activate(DoubleMatrix matrix) {
        DoubleMatrix r = matrix.dup();
        for (int i = 0; i < r.length; i++) {
            r.put(i, this.sigmoid(r.get(i)));
        }
        return r;
    }

    @Override
    DoubleMatrix deactivate(DoubleMatrix matrix) {
        DoubleMatrix r = matrix.dup();
        for (int i = 0; i < r.length; i++) {
            r.put(i, this.digmoid(r.get(i)));
        }
        return r;
    }

    /**
     * Activation function.
     */
    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * Derivate of activation function.
     */
    private double digmoid(double x) {
        return x * (1 - x);
    }
}
