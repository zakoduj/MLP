import java.util.Arrays;
import java.util.Random;

public class Main {
    public static void main(String [] args) {
        double start = System.currentTimeMillis();
        double[][] inputs = new double[][] {
                new double[] {1, 1, 0, 0},
                new double[] {0, 0, -1,-1}
        };
        double[][] outputs = new double[][] {
                new double[] {1, 0},
                new double[] {0, 1}
        };
        Random random = new Random();

        // Time = 0.181s
        MultilayerPerceptron mlp = new MultilayerPerceptron(
                new Layer(inputs[0].length, 16, random),
                new Layer(16, 8, random),
                new Layer(8, 32, random),
                new Layer(32, outputs[0].length, random)
        );
//        MultilayerPerceptron mlp = new MultilayerPerceptron(inputs[0].length, outputs[0].length); // Time = 0.656s
        mlp.train(inputs, outputs, error -> error > 0.01);
        System.out.println("Time = " + (System.currentTimeMillis() - start) / 1000 + "s");

        System.out.println(Arrays.toString(mlp.classify(inputs[0])));
        System.out.println(Arrays.toString(mlp.classify(inputs[1])));

    }
}
