package ca.germuth.machine_learning;

import java.util.ArrayList;
import ca.germuth.machine_learning.Main;

public class GradientDescent {
//	private static final double LEARNING_RATE = 0.33; //set 1
//	private static final double LEARNING_RATE = 0.33; //set 2
	private static final double LEARNING_RATE = 0.0242; // set 3
	
//	private static final double LEARNING_RATE = 0.0000001; 	//set 4 no norm
//	private static final double LEARNING_RATE = 0.40;		//set 4 with norm
	
	public static double[] run(ArrayList<TrainingData> training, boolean printError) {
		int numFeatures = training.get(0).input.length;
		// y = Th0 + Th1*x1 + Th2*x2 + Th3*x3 + etc...
		
		// initialize theta to be all ones, just need a starting state
		double[] theta = new double[numFeatures];
		for (int i = 0; i < theta.length; i++) {
			theta[i] = 1.0;
		}

		double currError = Main.error(theta, training);
		// difficult to test for convergence, so rather
		// just do 100 fixed iterations
		for (int i = 0; i < 1000; i++) {
			// take derivative of cost function
			// see https://chrisjmccormick.wordpress.com/2014/03/04/gradient-descent-derivation/
			// where x(0) = 1.0

			double[] newTheta = new double[numFeatures];
			// for each feature, calc partial derivative
			// dJ/d(theta) = 1/m * sum( predicted - actual ) * x(i)
			for (int j = 0; j < numFeatures; j++) {
				double deriv = 0.0;
				for (int k = 0; k < training.size(); k++) {
					TrainingData curr = training.get(k);
					deriv += (Main.predict(theta, curr.input) - curr.output) * curr.input[j];
				}
				deriv /= training.size();
				newTheta[j] = theta[j] - LEARNING_RATE * deriv;
			}

			theta = newTheta;

			currError = Main.error(theta, training);
			System.out.println(currError);
		}
		return theta;
	}
}
