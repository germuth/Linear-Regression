package ca.germuth.machine_learning;

import Jama.Matrix;

public class GradientDescent {
	//TODO these should be parameters
	public static Matrix run(Matrix X, Matrix y, final int NUM_ITERATIONS, final double LEARNING_RATE, boolean printError) {
		//n = number of features (including fake feature)
		int n = X.getColumnDimension();
		//m = number of training tuples
		int m = X.getRowDimension();
		
		// initialize theta to be all ones, just need a starting state
		double[][] thetaArr = new double[n][1];
		for (int i = 0; i < thetaArr.length; i++) {
			thetaArr[i][0] = 0.0;
		}
		Matrix theta = new Matrix(thetaArr);

		double currError = Main.error(theta, X, y);
		if(printError){
			System.out.println("Error before: " + currError);
		}
		// difficult to test for convergence, so rather
		// just do 100 fixed iterations
		for (int i = 0; i < NUM_ITERATIONS; i++) {
			// where x(0) = 1.0

			double[][] newTheta = new double[n][1];
			// for each feature, calc partial derivative
			// dJ/d(theta) = 1/m * sum( predicted - actual ) * x(i)
			for (int j = 0; j < n; j++) {
				double deriv = 0.0;
				for (int row = 0; row < m; row++) {
					deriv += (Main.predict(theta, X.getMatrix(row, row, 0, n-1)) - y.getArray()[row][0]) * X.getArray()[row][j];
				}
				newTheta[j][0] = theta.getArray()[j][0] - LEARNING_RATE * deriv / m;
			}

			theta = new Matrix(newTheta);

			currError = Main.error(theta, X, y);
			if(printError){
				System.out.println(currError);				
			}
		}
		return theta;
	}
}
