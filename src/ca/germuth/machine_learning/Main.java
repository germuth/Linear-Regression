package ca.germuth.machine_learning;

import java.io.File;
import java.io.FileNotFoundException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Scanner;

import Jama.Matrix;

/**
 * Multi Variate Linear Regression
 * 	Currently has Potential to use Gradient Descent or the Normal Equation
 * 	to solve multi variable linear regression
 * @author Aaron
 *
 */
public class Main {
	private static final String TRAINING_FILE = "training_data_3.txt";
	
	//input training examples
	//each row is all one training tuple
	//each column is one feature
	private static Matrix X;
	//one column of outputs for each training tuple
	private static Matrix y;
	
	public static void main(String[] args) {
		DecimalFormat df = new DecimalFormat();
		df.setMaximumFractionDigits(3);
		
		//initialize X and y
		readTrainingData();
		
		//normalize so all features have the same distribution
		//approximately -2.5 <-> 2.5
		//gradient descent converges poorly when features are out of proportion
//		normalize(training);
		
		//add fake feature to bias so that we can matrix multiple
		//for ex. 
		// y = th1 * x1 + th0 -> y = th1 * x1 + th0 * x0
		//x0, the fake feature, is always 1
		//so we add a column of ones to input
		addFakeFeature();

		Matrix theta = GradientDescent.run(X, y, 1500, 0.01, false);
//		Matrix theta = NormalEquation.run(training);
		
		//display results
		int n = theta.getRowDimension();
		
		System.out.println("Best Estimate for Equation:");
		String str = "y = ";
		for(int i = 0; i < n; i++){
			str += df.format(theta.getArray()[i][0]) + " ";
			if(i != 0){
				str += "x" + (i) + " ";
			}
			str += "+ ";
		}
		
		System.out.println(str.substring(0, str.length() - 3));
		System.out.println("");
		System.out.println("Total Error:");
		System.out.println(df.format(error(theta, X, y)));
		System.out.println("");
		System.out.println("Press Y to predict an input, and N to close program");
		Scanner s = new Scanner(System.in);
		String token = s.next().toLowerCase();
		while(!token.contains("n")){
			System.out.println("Enter Input Features");
			double[][] in = new double[1][n];
			in[0][0] = 1.0;
			for(int i = 1; i < n; i++){
				in[0][i] = s.nextDouble();
			}
			Matrix x = new Matrix(in);
			//TODO need to normalize input features if training set is normalized
			System.out.println("Expected Output is " + predict(theta, x));
		}
		s.close();
	}

	public static double predict(Matrix theta, Matrix x){
		return x.times(theta).getArray()[0][0];
	}
	
	public static double error(Matrix theta, Matrix X, Matrix y){
		double error = 0.0;
		
		int m = X.getRowDimension();
		int n = X.getColumnDimension();
		for(int row = 0; row < m; row++){			
			//									get each row of matrix
			double predicted = predict(theta, X.getMatrix(row, row, 0, n-1));
			double actual = y.getArray()[row][0];
			double diff = predicted - actual;
			error += Math.pow(diff, 2);
		}
		return error / (2 * m);
	}
	
	private static void addFakeFeature() {
		double[][] soFar = X.getArray();
		double[][] newX = new double[soFar.length][soFar[0].length + 1];
		for(int i = 0; i < newX.length; i++){
			for(int j = 0; j < newX[i].length; j++){
				if(j == 0){
					newX[i][j] = 1.0;
				}else{
					newX[i][j] = soFar[i][j-1];					
				}
			}
		}
		X = new Matrix(newX);
	}
	
	public static ArrayList<TrainingData> normalize(ArrayList<TrainingData> training){
		int numFeatures = training.get(0).input.length;
		//for each feature
		for(int i = 0; i < numFeatures; i++){
			//get all values, find max and min
			double MAX = Double.MIN_VALUE;
			double MIN = Double.MAX_VALUE;
			ArrayList<Double> allValues = new ArrayList<Double>();
			for(int j = 0; j < training.size(); j++){
				double val = training.get(j).input[i];
				if(val > MAX){
					MAX = val;
				}
				if(val < MIN){
					MIN = val;
				}
				allValues.add(val);
			}
			
			//okay now replace values with normalized
			for(int j = 0; j < training.size(); j++){
				double val = training.get(j).input[i];
				//gives range between 0 <-> 1
				val = (val - MIN) / (MAX - MIN);
				//range between 0 <-> 2
				val *= 2;
				//range between -1 <-> 1
				val -= 1.0;
				training.get(j).input[i] = val;
			}
		}
		return training;
	}
	
	public static void readTrainingData(){ 
		ArrayList<TrainingData> list = new ArrayList<TrainingData>();
		int inputs = -1; //start at -1 to exclude last input, which is output
		try {
			Scanner s = new Scanner(new File(TRAINING_FILE));
			//read header to determine number of variables
			Scanner lineScanner = new Scanner(s.nextLine());
			while(lineScanner.hasNext()){
				inputs++;
				lineScanner.next();
			}
			lineScanner.close();
			s.nextLine();
			
			while(s.hasNextLine()){
				String line = s.nextLine();
				lineScanner = new Scanner(line);
				double[] input = new double[inputs];
				for(int i = 0; i < inputs; i++){
					input[i] = lineScanner.nextDouble();
				}
				list.add(new TrainingData(input, lineScanner.nextDouble()));
			}
			s.close();
		} catch (FileNotFoundException e) {
			System.err.println("File not Found");
			e.printStackTrace();
		}
		//convert to matrix
		double[][] input = new double[list.size()][inputs];
		double[][] output = new double[list.size()][1];
		for(int i = 0; i < input.length; i++){
			for(int j = 0; j < input[i].length + 1; j++){
				if(j == input[i].length){			
					output[i][0] = list.get(i).output;
				}else{
					input[i][j] = list.get(i).input[j];					
				}
			}
		}
		X = new Matrix(input);
		y = new Matrix(output);
	}
	
}class TrainingData{
	public double[] input;
	public double output;
	public TrainingData(double[] i, double o){
		input = i;
		output = o;
	}
}
