import java.util.Random;

/**
 * Network layer.
 */
class Layer extends SigmoidLayer {
    Layer(int inputSize, int outputSize, Random random) {
        super(inputSize, outputSize, random::nextGaussian);
    }
}
